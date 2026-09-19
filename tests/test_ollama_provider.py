"""The local-model provider, which needs no API key."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from claimscope.config import DEFAULT_MODELS, Settings
from claimscope.llm import ProviderStructuredLLM


class Answer(BaseModel):
    value: int


class TestConfiguration:
    def test_has_a_default_model(self) -> None:
        assert Settings(provider="ollama").model_name == DEFAULT_MODELS["ollama"]

    def test_needs_no_api_key(self) -> None:
        """A local model authenticates with nothing, unlike the hosted ones."""
        settings = Settings(provider="ollama")

        assert settings.active_api_key is None
        assert not settings.needs_api_key

    def test_hosted_providers_still_need_a_key(self) -> None:
        assert Settings(provider="google").needs_api_key
        assert Settings(provider="anthropic").needs_api_key

    def test_has_a_default_base_url(self) -> None:
        assert Settings(provider="ollama").ollama_base_url.startswith("http")

    def test_the_context_window_fits_a_paper(self) -> None:
        """Ollama defaults to 2048 tokens, which would truncate a paper silently."""
        from claimscope.nodes.extract_claims import MAX_PAPER_CHARS

        settings = Settings(provider="ollama")

        # Roughly four characters per token, plus room for the prompt itself.
        assert settings.ollama_context_tokens > MAX_PAPER_CHARS / 4


class TestModelConstruction:
    def test_builds_without_a_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The key check must not reject a provider that needs no key."""
        built: dict[str, object] = {}

        class FakeChatOllama:
            def __init__(self, **kwargs: object) -> None:
                built.update(kwargs)

        import langchain_ollama

        monkeypatch.setattr(langchain_ollama, "ChatOllama", FakeChatOllama)

        llm = ProviderStructuredLLM(Settings(provider="ollama", model_name="qwen3:4b"))
        llm._get_model()

        assert built["model"] == "qwen3:4b"

    def test_passes_the_context_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        built: dict[str, object] = {}

        class FakeChatOllama:
            def __init__(self, **kwargs: object) -> None:
                built.update(kwargs)

        import langchain_ollama

        monkeypatch.setattr(langchain_ollama, "ChatOllama", FakeChatOllama)

        settings = Settings(provider="ollama", ollama_context_tokens=16000)
        ProviderStructuredLLM(settings)._get_model()

        assert built["num_ctx"] == 16000

    def test_uses_deterministic_sampling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Reproducibility matters more than variety for this work.
        built: dict[str, object] = {}

        class FakeChatOllama:
            def __init__(self, **kwargs: object) -> None:
                built.update(kwargs)

        import langchain_ollama

        monkeypatch.setattr(langchain_ollama, "ChatOllama", FakeChatOllama)

        ProviderStructuredLLM(Settings(provider="ollama"))._get_model()

        assert built["temperature"] == 0.0


class TestKeyErrors:
    def test_ollama_never_complains_about_a_missing_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeChatOllama:
            def __init__(self, **_kwargs: object) -> None:
                pass

        import langchain_ollama

        monkeypatch.setattr(langchain_ollama, "ChatOllama", FakeChatOllama)

        # Would raise for anthropic or google.
        ProviderStructuredLLM(Settings(provider="ollama"))._get_model()

    def test_google_still_complains(self) -> None:
        llm = ProviderStructuredLLM(Settings(provider="google", gemini_api_key=None))

        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            llm.invoke_structured("prompt", Answer)
