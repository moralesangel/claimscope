"""The CLI turns library tracebacks into actionable one-liners."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from claimscope.cli import app

runner = CliRunner()


@pytest.mark.parametrize(
    ("error_message", "expected_hint"),
    [
        (
            "Error code: 400 - your credit balance is too low to access the Anthropic API",
            "Add credits",
        ),
        ("Error code: 401 - authentication_error: bad key", "ANTHROPIC_API_KEY"),
        ("invalid x-api-key", "ANTHROPIC_API_KEY"),
        ("ANTHROPIC_API_KEY is not set. Copy .env.example", "Copy .env.example"),
        ("Error code: 429 - rate_limit_error", "Rate limited"),
    ],
)
def test_maps_known_failures_to_a_hint(
    monkeypatch: pytest.MonkeyPatch, error_message: str, expected_hint: str
) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(error_message)

    monkeypatch.setattr("claimscope.graph.build_graph", explode)

    result = runner.invoke(app, ["analyze", "1706.03762"])

    assert result.exit_code == 1
    assert expected_hint in result.stdout
    # The raw library traceback should not be dumped by default.
    assert "Traceback" not in result.stdout


def test_unknown_failures_show_the_message_and_offer_verbose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("something unexpected broke")

    monkeypatch.setattr("claimscope.graph.build_graph", explode)

    result = runner.invoke(app, ["analyze", "1706.03762"])

    assert result.exit_code == 1
    assert "something unexpected broke" in result.stdout
    assert "--verbose" in result.stdout
