"""LLM access with schema validation and one retry (PLAN.md section 11)."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from pydantic import BaseModel, SecretStr, ValidationError

from claimscope.config import Settings, get_settings

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredLLM(Protocol):
    """The slice of a chat model this project uses.

    Declared as a Protocol so tests can pass a stub without importing langchain
    or touching the network.
    """

    def invoke_structured(self, prompt: str, schema: type[T]) -> T: ...


_KEY_ENV_VAR = {"anthropic": "ANTHROPIC_API_KEY", "google": "GEMINI_API_KEY"}

TRANSIENT_RETRIES = 8
"""Attempts before giving up on a transient failure.

Free-tier models are congested for minutes at a time, and a 503 is a refusal
rather than a served request, so waiting costs nothing but wall-clock. Eight
attempts with the backoff below wait about six and a half minutes in total.
"""

TRANSIENT_BACKOFF_S = 5.0
MAX_BACKOFF_S = 120.0
"""Cap on a single wait, so the delay plateaus instead of doubling forever."""

# Server-side conditions that clear on their own.
_TRANSIENT_MARKERS = (
    "503",
    "unavailable",
    "overloaded",
    "high demand",
    "429",
    "rate limit",
    "resource_exhausted",
    "internal error",
    "500",
)

# Conditions a short backoff will never clear: a zero quota, an exhausted daily
# allowance, or no credit at all.
_PERMANENT_MARKERS = (
    "limit: 0",
    "credit balance is too low",
    "generaterequestsperdayperprojectpermodel",
    "perdayperproject",
)


def _is_quota_exhausted(message: str) -> bool:
    """Whether this model's allowance is spent, as opposed to any other failure.

    Distinct from _is_transient: waiting will not help, but another model will,
    because the free tier counts per model rather than per account.
    """
    lowered = message.lower()
    # Only the per-day and zero-quota signatures: a per-minute 429 clears on its
    # own, and _is_transient already waits that out.
    return any(marker in lowered for marker in ("perdayperproject", "limit: 0"))


def _is_transient(message: str) -> bool:
    """Whether a failure is worth retrying."""
    lowered = message.lower()
    if any(marker in lowered for marker in _PERMANENT_MARKERS):
        return False
    return any(marker in lowered for marker in _TRANSIENT_MARKERS)


class ProviderStructuredLLM:
    """Calls the configured provider and validates the reply against a schema.

    The plan specifies Anthropic. Google is supported behind the same interface
    so the pipeline can run when no Anthropic credit is available; switching
    providers is a config change, not a code change.
    """

    def __init__(
        self, settings: Settings | None = None, model: BaseChatModel | None = None
    ) -> None:
        self._settings = settings or get_settings()
        self._model = model
        self._chain = self._settings.model_chain()
        """The fallback order, fixed at construction.

        It must not be recomputed after a switch: model_chain() puts the current
        model first, so re-deriving it once model_name has changed reorders the
        list and each model ends up pointing back at the one before it.
        """
        self._retired: set[str] = set()
        """Models whose daily allowance is spent. Never tried again this process."""
        self._skipped: set[str] = set()
        """Models congested during the current call. Cleared on the next one."""

    def _get_model(self) -> BaseChatModel:
        if self._model is not None:
            return self._model

        provider = self._settings.provider
        api_key = self._settings.active_api_key
        if self._settings.needs_api_key and not api_key:
            raise RuntimeError(
                f"{_KEY_ENV_VAR[provider]} is not set. Copy .env.example to .env and fill it in."
            )

        if provider == "ollama":
            from langchain_ollama import ChatOllama

            # A local model is slow but never rate limited, so the timeout is
            # generous and there is nothing to back off from.
            self._model = ChatOllama(
                model=self._settings.model_name,
                base_url=self._settings.ollama_base_url,
                temperature=0.0,
                num_ctx=self._settings.ollama_context_tokens,
            )
        elif api_key is None:
            # Unreachable: needs_api_key covers exactly these providers.
            raise RuntimeError(f"no API key available for provider {provider}")
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            # These are pydantic field aliases; ChatAnthropic types its
            # __init__ as (*args, **kwargs), so they are not statically checked.
            self._model = ChatAnthropic(
                model_name=self._settings.model_name,
                api_key=SecretStr(api_key),
                timeout=120.0,
                stop=None,
                max_retries=0,  # see the Google branch: our backoff owns this
            )
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI

            # max_retries=0 because _invoke_with_backoff decides what to retry.
            # The SDK's own retry loop burns ~40s on a daily quota error that
            # will never clear, and hides the attempts from our logging.
            self._model = ChatGoogleGenerativeAI(
                model=self._settings.model_name,
                google_api_key=SecretStr(api_key),
                timeout=120.0,
                max_retries=0,
            )
        return self._model

    def _invoke_with_backoff(self, structured: Runnable[Any, Any], prompt: str) -> object:
        """Call the model, retrying transient server-side failures.

        Providers return 503 under load and 429 when rate limited; both clear on
        their own. A quota of zero never does, so it is raised immediately.
        """
        last_error: Exception | None = None

        for attempt in range(1, TRANSIENT_RETRIES + 1):
            try:
                return structured.invoke(prompt)
            except Exception as exc:
                message = str(exc)
                if not _is_transient(message) or attempt == TRANSIENT_RETRIES:
                    raise
                last_error = exc
                delay = min(TRANSIENT_BACKOFF_S * (2 ** (attempt - 1)), MAX_BACKOFF_S)
                logger.warning(
                    "transient LLM failure (attempt %d/%d), retrying in %.0fs: %s",
                    attempt,
                    TRANSIENT_RETRIES,
                    delay,
                    message[:200],
                )
                time.sleep(delay)

        raise RuntimeError(f"unreachable: {last_error}")

    def invoke_structured(self, prompt: str, schema: type[T]) -> T:
        """Invoke the model, moving to a fallback if this one is out of quota.

        Google counts its free tier per model, so an exhausted daily allowance
        is not the end of the run: the next model in the chain has its own.
        """
        # Congestion skips last for this call only; retirements outlive it.
        self._skipped = set()
        if self._settings.model_name in self._retired and not self._advance_model(retire=True):
            raise RuntimeError(
                "every model in the chain has spent its quota: " + ", ".join(self._chain)
            )
        while True:
            try:
                return self._invoke_one_model(prompt, schema)
            except Exception as exc:
                message = str(exc)
                # Quota exhaustion is the obvious case, but a model that stays
                # congested through the whole backoff is equally unusable, and
                # another model is often serving fine at that moment.
                exhausted = _is_quota_exhausted(message)
                if not (exhausted or _is_transient(message)):
                    raise
                if not self._advance_model(retire=exhausted):
                    raise

    def _advance_model(self, retire: bool) -> bool:
        """Switch to the next model not yet tried for this call. False when none is left.

        Two failures look alike here and must not be treated alike. A spent
        daily allowance never comes back, so that model is retired for the rest
        of the process. Congestion clears in minutes, so the model is only
        skipped for the remainder of this call and is available again on the
        next one.

        The skip is what stops the ping-pong: model_chain() lists the current
        model first, so walking it by position after a switch makes each model
        point back at the one before it, and the pair alternates forever.
        """
        current = self._settings.model_name
        if retire:
            self._retired.add(current)
        self._skipped.add(current)

        blocked = self._retired | self._skipped
        nxt = next((name for name in self._chain if name not in blocked), None)
        if nxt is None:
            logger.error(
                "no model in the chain could serve the request: %s", ", ".join(self._chain)
            )
            return False

        logger.warning("%s is unusable right now; switching to %s", current, nxt)
        self._settings = self._settings.model_copy(update={"model_name": nxt})
        self._model = None  # rebuilt lazily against the new model
        return True

    def _invoke_one_model(self, prompt: str, schema: type[T]) -> T:
        """Invoke the current model, retrying once if the reply fails validation.

        The plan requires validating every LLM output and retrying once on
        failure, logging the error either way. Transient transport failures are
        handled separately, in _invoke_with_backoff.
        """
        structured = self._get_model().with_structured_output(schema)
        last_error: Exception | None = None

        for attempt in (1, 2):
            try:
                result = self._invoke_with_backoff(structured, prompt)
                if isinstance(result, schema):
                    return result
                return schema.model_validate(result)
            except (ValidationError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "LLM output failed %s validation on attempt %d: %s",
                    schema.__name__,
                    attempt,
                    exc,
                )

        raise ValueError(
            f"LLM output failed {schema.__name__} validation twice: {last_error}"
        ) from last_error
