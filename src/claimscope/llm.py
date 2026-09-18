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

TRANSIENT_RETRIES = 4
TRANSIENT_BACKOFF_S = 5.0

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

# A zero quota is a plan limit, not congestion: retrying never helps.
_PERMANENT_MARKERS = ("limit: 0", "credit balance is too low")


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

    def _get_model(self) -> BaseChatModel:
        if self._model is not None:
            return self._model

        provider = self._settings.provider
        api_key = self._settings.active_api_key
        if not api_key:
            raise RuntimeError(
                f"{_KEY_ENV_VAR[provider]} is not set. Copy .env.example to .env and fill it in."
            )

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            # These are pydantic field aliases; ChatAnthropic types its
            # __init__ as (*args, **kwargs), so they are not statically checked.
            self._model = ChatAnthropic(
                model_name=self._settings.model_name,
                api_key=SecretStr(api_key),
                timeout=120.0,
                stop=None,
            )
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._model = ChatGoogleGenerativeAI(
                model=self._settings.model_name,
                google_api_key=SecretStr(api_key),
                timeout=120.0,
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
                delay = TRANSIENT_BACKOFF_S * (2 ** (attempt - 1))
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
        """Invoke the model, retrying once if the reply fails validation.

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
