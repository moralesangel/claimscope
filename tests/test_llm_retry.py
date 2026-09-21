"""Transient failures are retried with backoff; permanent ones are not."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from claimscope.config import Settings
from claimscope.llm import TRANSIENT_RETRIES, ProviderStructuredLLM, _is_transient


class Answer(BaseModel):
    value: int


class FlakyRunnable:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def invoke(self, _prompt: str) -> object:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FlakyModel:
    def __init__(self, outcomes: list[object]) -> None:
        self.runnable = FlakyRunnable(outcomes)

    def with_structured_output(self, _schema: type[BaseModel]) -> FlakyRunnable:
        return self.runnable


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep backoff tests instant."""
    monkeypatch.setattr("claimscope.llm.time.sleep", lambda _s: None)


def _llm(outcomes: list[object]) -> tuple[ProviderStructuredLLM, FlakyModel]:
    model = FlakyModel(outcomes)
    return ProviderStructuredLLM(Settings(anthropic_api_key="k"), model=model), model  # type: ignore[arg-type]


class TestIsTransient:
    @pytest.mark.parametrize(
        "message",
        [
            "503 UNAVAILABLE. This model is currently experiencing high demand.",
            "429 RESOURCE_EXHAUSTED quota exceeded, please retry in 53s",
            "Error code: 500 - internal error",
            "overloaded_error",
        ],
    )
    def test_recognises_transient_failures(self, message: str) -> None:
        assert _is_transient(message)

    @pytest.mark.parametrize(
        "message",
        [
            # A zero quota is a plan limit; waiting never clears it.
            "429 RESOURCE_EXHAUSTED ... limit: 0, model: gemini-3.1-pro",
            # A daily allowance does not come back within a backoff either.
            "429 quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier",
            "Your credit balance is too low to access the Anthropic API",
            "401 authentication_error: invalid x-api-key",
            "404 model not found",
        ],
    )
    def test_rejects_permanent_failures(self, message: str) -> None:
        assert not _is_transient(message)

    def test_a_per_minute_limit_is_still_transient(self) -> None:
        # Unlike a daily cap, this clears while we wait.
        assert _is_transient("429 quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier")


def test_retries_a_503_then_succeeds() -> None:
    llm, model = _llm([RuntimeError("503 UNAVAILABLE high demand"), Answer(value=5)])

    assert llm.invoke_structured("prompt", Answer).value == 5
    assert model.runnable.calls == 2


def test_the_backoff_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Doubling without a cap would eventually wait longer than anyone will."""
    from claimscope.llm import MAX_BACKOFF_S

    waits: list[float] = []
    monkeypatch.setattr("claimscope.llm.time.sleep", waits.append)

    outcomes: list[object] = [RuntimeError("503 UNAVAILABLE")] * (TRANSIENT_RETRIES - 1)
    outcomes.append(Answer(value=1))
    llm, _ = _llm(outcomes)

    llm.invoke_structured("prompt", Answer)

    assert waits, "expected the run to have waited at all"
    assert max(waits) <= MAX_BACKOFF_S


def test_it_waits_long_enough_to_outlast_congestion() -> None:
    """Free-tier models stay congested for minutes, not seconds."""
    from claimscope.llm import MAX_BACKOFF_S, TRANSIENT_BACKOFF_S

    total = sum(
        min(TRANSIENT_BACKOFF_S * 2**attempt, MAX_BACKOFF_S)
        for attempt in range(TRANSIENT_RETRIES - 1)
    )

    assert total >= 300, f"only waits {total:.0f}s in total"


def test_gives_up_after_the_retry_budget() -> None:
    outcomes: list[object] = [RuntimeError("503 UNAVAILABLE")] * TRANSIENT_RETRIES
    llm, model = _llm(outcomes)

    with pytest.raises(RuntimeError, match="503"):
        llm.invoke_structured("prompt", Answer)

    assert model.runnable.calls == TRANSIENT_RETRIES


def test_does_not_retry_a_zero_quota() -> None:
    llm, model = _llm([RuntimeError("429 RESOURCE_EXHAUSTED limit: 0, model: gemini-3.1-pro")])

    with pytest.raises(RuntimeError, match="limit: 0"):
        llm.invoke_structured("prompt", Answer)

    assert model.runnable.calls == 1


def test_does_not_retry_an_auth_failure() -> None:
    llm, model = _llm([RuntimeError("401 authentication_error")])

    with pytest.raises(RuntimeError):
        llm.invoke_structured("prompt", Answer)

    assert model.runnable.calls == 1
