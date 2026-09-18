"""LLM access with schema validation and one retry (PLAN.md section 11)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, TypeVar

from pydantic import BaseModel, SecretStr, ValidationError

from claimscope.config import Settings, get_settings

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredLLM(Protocol):
    """The slice of a chat model this project uses.

    Declared as a Protocol so tests can pass a stub without importing langchain
    or touching the network.
    """

    def invoke_structured(self, prompt: str, schema: type[T]) -> T: ...


class AnthropicStructuredLLM:
    """Calls Claude and validates the reply against a Pydantic schema."""

    def __init__(
        self, settings: Settings | None = None, model: BaseChatModel | None = None
    ) -> None:
        self._settings = settings or get_settings()
        self._model = model

    def _get_model(self) -> BaseChatModel:
        if self._model is None:
            from langchain_anthropic import ChatAnthropic

            if not self._settings.anthropic_api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
                )
            # These are ChatAnthropic's pydantic field aliases; its __init__ is
            # typed as (*args, **kwargs), so they are not statically checked.
            self._model = ChatAnthropic(
                model_name=self._settings.model_name,
                api_key=SecretStr(self._settings.anthropic_api_key),
                timeout=120.0,
                stop=None,
            )
        return self._model

    def invoke_structured(self, prompt: str, schema: type[T]) -> T:
        """Invoke the model, retrying once if the reply fails validation.

        The plan requires validating every LLM output and retrying once on
        failure, logging the error either way.
        """
        structured = self._get_model().with_structured_output(schema)
        last_error: Exception | None = None

        for attempt in (1, 2):
            try:
                result = structured.invoke(prompt)
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
