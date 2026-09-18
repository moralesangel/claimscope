"""Tracing is optional, and must never be able to break a run."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from claimscope.config import Settings
from claimscope.tracing import setup_tracing, traced_run

CONFIG: dict[str, Any] = {"configurable": {"thread_id": "t-1"}}


def _settings(**overrides: Any) -> Settings:
    return Settings(anthropic_api_key="k", **overrides)


class TestDisabled:
    def test_no_callbacks_when_tracing_is_off(self) -> None:
        assert setup_tracing(_settings(tracing_enabled=False)) == []

    def test_config_passes_through_untouched(self) -> None:
        with traced_run(_settings(tracing_enabled=False), CONFIG) as config:
            assert config == CONFIG

    def test_nothing_is_written_to_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

        setup_tracing(_settings(tracing_enabled=False))

        import os

        assert "LANGSMITH_TRACING" not in os.environ


class TestMisconfiguration:
    """A half-configured backend degrades to no tracing, with a warning."""

    def test_missing_langsmith_key_disables_tracing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

        with caplog.at_level(logging.WARNING):
            callbacks = setup_tracing(_settings(tracing_enabled=True, tracing_backend="langsmith"))

        assert callbacks == []
        assert "LANGSMITH_API_KEY" in caplog.text

    def test_missing_langfuse_keys_disable_tracing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
            monkeypatch.delenv(name, raising=False)

        with caplog.at_level(logging.WARNING):
            callbacks = setup_tracing(_settings(tracing_enabled=True, tracing_backend="langfuse"))

        assert callbacks == []
        assert "LANGFUSE_PUBLIC_KEY" in caplog.text

    def test_a_run_still_works_when_tracing_is_misconfigured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The property that matters: tracing cannot take the run down."""
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

        with traced_run(_settings(tracing_enabled=True), CONFIG) as config:
            assert config == CONFIG


class TestLangSmith:
    def test_sets_the_variables_langchain_reads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
        monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)

        setup_tracing(
            _settings(
                tracing_enabled=True, tracing_backend="langsmith", tracing_project="myproject"
            )
        )

        import os

        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert os.environ["LANGSMITH_PROJECT"] == "myproject"

    def test_does_not_override_an_existing_project(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
        monkeypatch.setenv("LANGSMITH_PROJECT", "set-by-the-user")

        setup_tracing(_settings(tracing_enabled=True, tracing_backend="langsmith"))

        import os

        assert os.environ["LANGSMITH_PROJECT"] == "set-by-the-user"

    def test_uses_no_callbacks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # LangSmith is picked up from the environment, not via callbacks.
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")

        assert setup_tracing(_settings(tracing_enabled=True, tracing_backend="langsmith")) == []


class TestLangfuse:
    def test_a_missing_package_is_a_warning_not_a_crash(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")

        with caplog.at_level(logging.WARNING):
            callbacks = setup_tracing(_settings(tracing_enabled=True, tracing_backend="langfuse"))

        # langfuse is an optional extra, absent in the default test environment.
        assert callbacks == []
        assert "uv sync --extra tracing" in caplog.text

    def test_a_failing_handler_does_not_propagate(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A bad host or credentials must not end the run."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")

        import sys
        import types

        module = types.ModuleType("langfuse.langchain")

        def explode(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("cannot reach the Langfuse host")

        module.CallbackHandler = explode  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langfuse", types.ModuleType("langfuse"))
        monkeypatch.setitem(sys.modules, "langfuse.langchain", module)

        with caplog.at_level(logging.WARNING):
            callbacks = setup_tracing(_settings(tracing_enabled=True, tracing_backend="langfuse"))

        assert callbacks == []
        assert "continuing untraced" in caplog.text


class TestFlush:
    def test_traces_are_flushed_on_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Short CLI runs exit before a background batch would be sent."""
        flushed: list[bool] = []

        class FakeClient:
            def flush(self) -> None:
                flushed.append(True)

        class FakeHandler:
            client = FakeClient()

        monkeypatch.setattr("claimscope.tracing.setup_tracing", lambda settings: [FakeHandler()])

        with traced_run(_settings(tracing_enabled=True), CONFIG) as config:
            assert "callbacks" in config

        assert flushed == [True]

    def test_a_failing_flush_is_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeClient:
            def flush(self) -> None:
                raise RuntimeError("network gone")

        class FakeHandler:
            client = FakeClient()

        monkeypatch.setattr("claimscope.tracing.setup_tracing", lambda settings: [FakeHandler()])

        # Must not raise on the way out.
        with traced_run(_settings(tracing_enabled=True), CONFIG):
            pass
