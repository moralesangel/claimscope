"""Provider selection: model defaults, key routing, and error messages."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from claimscope.config import DEFAULT_MODELS, Settings
from claimscope.llm import ProviderStructuredLLM


class Answer(BaseModel):
    value: int


class TestModelDefaults:
    def test_anthropic_default_model(self) -> None:
        assert Settings(provider="anthropic").model_name == DEFAULT_MODELS["anthropic"]

    def test_google_default_model(self) -> None:
        assert Settings(provider="google").model_name == DEFAULT_MODELS["google"]

    def test_explicit_model_wins_over_the_default(self) -> None:
        settings = Settings(provider="google", model_name="gemini-3.1-pro-preview")
        assert settings.model_name == "gemini-3.1-pro-preview"


class TestKeyRouting:
    def test_anthropic_uses_the_anthropic_key(self) -> None:
        settings = Settings(provider="anthropic", anthropic_api_key="a", gemini_api_key="g")
        assert settings.active_api_key == "a"

    def test_google_uses_the_gemini_key(self) -> None:
        settings = Settings(provider="google", anthropic_api_key="a", gemini_api_key="g")
        assert settings.active_api_key == "g"

    def test_missing_key_is_none(self) -> None:
        assert Settings(provider="google", anthropic_api_key="a").active_api_key is None


class TestMissingKeyErrors:
    def test_names_the_anthropic_variable(self) -> None:
        llm = ProviderStructuredLLM(Settings(provider="anthropic", anthropic_api_key=None))

        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            llm.invoke_structured("prompt", Answer)

    def test_names_the_gemini_variable(self) -> None:
        llm = ProviderStructuredLLM(Settings(provider="google", gemini_api_key=None))

        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            llm.invoke_structured("prompt", Answer)

    def test_google_is_unaffected_by_a_present_anthropic_key(self) -> None:
        # Selecting google must not silently fall back to the other provider's key.
        llm = ProviderStructuredLLM(
            Settings(provider="google", anthropic_api_key="present", gemini_api_key=None)
        )

        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            llm.invoke_structured("prompt", Answer)
