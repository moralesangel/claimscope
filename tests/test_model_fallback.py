"""Falling back to another model when one runs out of daily quota.

Google's free tier counts requests per model, not per account, so an exhausted
model is not an exhausted key. Discovered the hard way: gemini-3.6-flash was
refusing every request while gemini-flash-latest answered fine.
"""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel

from claimscope.config import DEFAULT_FALLBACKS, Settings
from claimscope.llm import ProviderStructuredLLM, _is_quota_exhausted

DAILY_QUOTA_ERROR = (
    "429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
)


class Answer(BaseModel):
    value: int


class RecordingModel:
    """A chat model that fails or answers according to a script."""

    def __init__(self, outcomes: dict[str, object], name: str) -> None:
        self.outcomes = outcomes
        self.name = name

    def with_structured_output(self, _schema: type[BaseModel]) -> RecordingModel:
        return self

    def invoke(self, _prompt: str) -> object:
        outcome = self.outcomes[self.name]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _llm_over(outcomes: dict[str, object], settings: Settings) -> ProviderStructuredLLM:
    """A client that builds a RecordingModel for whichever model is current."""
    llm = ProviderStructuredLLM(settings)
    used: list[str] = []

    def build() -> RecordingModel:
        name = llm._settings.model_name
        used.append(name)
        return RecordingModel(outcomes, name)

    llm._get_model = build  # type: ignore[method-assign]
    llm.used = used  # type: ignore[attr-defined]
    return llm


class TestQuotaDetection:
    def test_recognises_a_daily_quota_error(self) -> None:
        assert _is_quota_exhausted(DAILY_QUOTA_ERROR)

    def test_recognises_a_zero_quota(self) -> None:
        assert _is_quota_exhausted("429 RESOURCE_EXHAUSTED ... limit: 0, model: gemini-3.1-pro")

    def test_a_per_minute_limit_is_not_exhaustion(self) -> None:
        """That clears on its own; the backoff handles it without switching."""
        assert not _is_quota_exhausted(
            "429 quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
        )

    def test_congestion_is_not_exhaustion(self) -> None:
        assert not _is_quota_exhausted("503 UNAVAILABLE high demand")


class TestChain:
    def test_the_primary_model_comes_first(self) -> None:
        chain = Settings(provider="google", model_name="gemini-3.6-flash").model_chain()

        assert chain[0] == "gemini-3.6-flash"

    def test_the_default_fallbacks_follow(self) -> None:
        chain = Settings(provider="google").model_chain()

        assert set(DEFAULT_FALLBACKS["google"]) <= set(chain)

    def test_an_explicit_list_overrides_the_defaults(self) -> None:
        settings = Settings(
            provider="google", model_name="primary", model_fallbacks=["second", "third"]
        )

        assert settings.model_chain() == ["primary", "second", "third"]

    def test_duplicates_are_removed(self) -> None:
        settings = Settings(provider="google", model_name="same", model_fallbacks=["same", "other"])

        assert settings.model_chain() == ["same", "other"]

    def test_providers_without_fallbacks_have_a_chain_of_one(self) -> None:
        assert len(Settings(provider="ollama").model_chain()) == 1


class TestSwitching:
    def test_an_exhausted_model_falls_through_to_the_next(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        settings = Settings(
            provider="google",
            gemini_api_key="k",
            model_name="first",
            model_fallbacks=["second"],
        )
        llm = _llm_over(
            {"first": RuntimeError(DAILY_QUOTA_ERROR), "second": Answer(value=7)}, settings
        )

        with caplog.at_level(logging.WARNING):
            result = llm.invoke_structured("prompt", Answer)

        assert result.value == 7
        assert llm.used == ["first", "second"]  # type: ignore[attr-defined]
        assert "switching to second" in caplog.text

    def test_it_walks_the_whole_chain(self) -> None:
        settings = Settings(
            provider="google",
            gemini_api_key="k",
            model_name="a",
            model_fallbacks=["b", "c"],
        )
        llm = _llm_over(
            {
                "a": RuntimeError(DAILY_QUOTA_ERROR),
                "b": RuntimeError(DAILY_QUOTA_ERROR),
                "c": Answer(value=3),
            },
            settings,
        )

        assert llm.invoke_structured("prompt", Answer).value == 3
        assert llm.used == ["a", "b", "c"]  # type: ignore[attr-defined]

    def test_it_gives_up_when_every_model_is_spent(self, caplog: pytest.LogCaptureFixture) -> None:
        settings = Settings(
            provider="google", gemini_api_key="k", model_name="a", model_fallbacks=["b"]
        )
        llm = _llm_over(
            {"a": RuntimeError(DAILY_QUOTA_ERROR), "b": RuntimeError(DAILY_QUOTA_ERROR)},
            settings,
        )

        with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError):
            llm.invoke_structured("prompt", Answer)

        assert "no model in the chain could serve" in caplog.text

    def test_persistent_congestion_also_switches(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A model stuck at 503 is as unusable as one out of quota.

        This is what actually blocked a real run: gemini-3.6-flash returned 503
        for the whole backoff while gemini-flash-latest was answering fine.
        """
        monkeypatch.setattr("claimscope.llm.time.sleep", lambda _s: None)
        settings = Settings(
            provider="google", gemini_api_key="k", model_name="busy", model_fallbacks=["free"]
        )
        llm = _llm_over(
            {"busy": RuntimeError("503 UNAVAILABLE high demand"), "free": Answer(value=9)},
            settings,
        )

        with caplog.at_level(logging.WARNING):
            assert llm.invoke_structured("prompt", Answer).value == 9

        assert "switching to free" in caplog.text

    def test_other_failures_do_not_switch_models(self) -> None:
        """Switching on an unrelated error would hide the real problem."""
        settings = Settings(
            provider="google", gemini_api_key="k", model_name="a", model_fallbacks=["b"]
        )
        llm = _llm_over(
            {"a": RuntimeError("401 authentication_error"), "b": Answer(value=1)}, settings
        )

        with pytest.raises(RuntimeError, match="authentication"):
            llm.invoke_structured("prompt", Answer)

        assert llm.used == ["a"]  # type: ignore[attr-defined]

    def test_a_working_model_is_left_alone(self) -> None:
        settings = Settings(
            provider="google", gemini_api_key="k", model_name="a", model_fallbacks=["b"]
        )
        llm = _llm_over({"a": Answer(value=5)}, settings)

        assert llm.invoke_structured("prompt", Answer).value == 5
        assert llm.used == ["a"]  # type: ignore[attr-defined]
