"""The doctor command: is this environment ready to run?"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from claimscope.cli import app

runner = CliRunner()


@pytest.fixture
def ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fully working environment."""
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("CLAIMSCOPE_PROVIDER", "google")
    monkeypatch.setenv("CLAIMSCOPE_SANDBOX_BACKEND", "subprocess")
    monkeypatch.setattr("claimscope.cli._first_working_model", lambda settings: "gemini-x")


def test_reports_ready_when_everything_works(ready: None) -> None:
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "Ready" in result.stdout


def test_a_missing_key_is_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAIMSCOPE_PROVIDER", "google")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "GEMINI_API_KEY" in result.stdout
    assert "Not ready" in result.stdout


def test_a_model_that_never_answers_is_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("CLAIMSCOPE_PROVIDER", "google")
    monkeypatch.setattr("claimscope.cli._first_working_model", lambda settings: None)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "no usable model" in result.stdout


def test_an_unavailable_sandbox_is_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("CLAIMSCOPE_PROVIDER", "google")
    monkeypatch.setenv("CLAIMSCOPE_SANDBOX_BACKEND", "docker")
    monkeypatch.setattr("claimscope.cli._first_working_model", lambda settings: "gemini-x")

    from claimscope.sandbox import docker_runner

    monkeypatch.setattr(docker_runner.DockerCPURunner, "available", lambda self: False)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "no sandbox" in result.stdout


def test_it_says_when_the_sandbox_does_not_contain(ready: None) -> None:
    """A reader must not assume the subprocess backend is containment."""
    result = runner.invoke(app, ["doctor"])

    assert "NOT contained" in result.stdout


def test_it_shows_the_model_chain(ready: None) -> None:
    # So a user can see which fallbacks would be tried.
    result = runner.invoke(app, ["doctor"])

    assert "model chain:" in result.stdout


def test_a_local_provider_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAIMSCOPE_PROVIDER", "ollama")
    monkeypatch.setenv("CLAIMSCOPE_SANDBOX_BACKEND", "subprocess")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "not needed" in result.stdout
