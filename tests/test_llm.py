"""Tests for the validate-and-retry-once contract (PLAN.md section 11)."""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel

from claimscope.config import Settings
from claimscope.llm import AnthropicStructuredLLM


class Answer(BaseModel):
    value: int


class FakeStructuredRunnable:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def invoke(self, _prompt: str) -> object:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeModel:
    def __init__(self, outcomes: list[object]) -> None:
        self.runnable = FakeStructuredRunnable(outcomes)

    def with_structured_output(self, _schema: type[BaseModel]) -> FakeStructuredRunnable:
        return self.runnable


def _llm(outcomes: list[object]) -> tuple[AnthropicStructuredLLM, FakeModel]:
    model = FakeModel(outcomes)
    return AnthropicStructuredLLM(Settings(anthropic_api_key="k"), model=model), model  # type: ignore[arg-type]


def test_returns_a_valid_first_response() -> None:
    llm, model = _llm([Answer(value=7)])

    assert llm.invoke_structured("prompt", Answer).value == 7
    assert model.runnable.calls == 1


def test_coerces_a_dict_response() -> None:
    llm, _ = _llm([{"value": 3}])

    assert llm.invoke_structured("prompt", Answer).value == 3


def test_retries_once_then_succeeds(caplog: pytest.LogCaptureFixture) -> None:
    llm, model = _llm([ValueError("bad output"), Answer(value=42)])

    with caplog.at_level(logging.WARNING):
        assert llm.invoke_structured("prompt", Answer).value == 42

    assert model.runnable.calls == 2
    assert "failed Answer validation on attempt 1" in caplog.text


def test_gives_up_after_two_failures(caplog: pytest.LogCaptureFixture) -> None:
    llm, model = _llm([ValueError("first"), ValueError("second")])

    with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match="validation twice"):
        llm.invoke_structured("prompt", Answer)

    assert model.runnable.calls == 2


def test_requires_an_api_key() -> None:
    llm = AnthropicStructuredLLM(Settings(anthropic_api_key=None))

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        llm.invoke_structured("prompt", Answer)
