"""Phase 0 smoke tests: the package imports and the CLI responds."""

from __future__ import annotations

from typer.testing import CliRunner

from claimscope import __version__
from claimscope.cli import app

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "version" in result.stdout
    assert "config" in result.stdout


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_config_redacts_api_key() -> None:
    result = runner.invoke(app, ["config"], env={"ANTHROPIC_API_KEY": "secret-value"})
    assert result.exit_code == 0
    assert "secret-value" not in result.stdout
    assert "anthropic_api_key:" in result.stdout


class TestThreadStatus:
    """A crashed run must not be listed as done.

    The old check was binary -- awaiting review, or done -- so a run that died
    inside a node was listed as finished and the error that killed it never
    appeared anywhere.
    """

    def _snapshot(self, *, next_nodes: tuple[str, ...] = (), tasks: list[object] | None = None):
        from types import SimpleNamespace

        return SimpleNamespace(next=next_nodes, tasks=tasks or [], values={})

    def test_a_finished_run_is_done(self) -> None:
        from claimscope.cli import _thread_status

        assert _thread_status(self._snapshot()) == "done"  # type: ignore[arg-type]

    def test_a_crashed_run_names_the_node(self) -> None:
        from types import SimpleNamespace

        from claimscope.cli import _thread_status

        task = SimpleNamespace(name="design_plan", interrupts=(), error="RemoteProtocolError(...)")
        snapshot = self._snapshot(next_nodes=("design_plan",), tasks=[task])

        status = _thread_status(snapshot)  # type: ignore[arg-type]

        assert "failed" in status
        assert "design_plan" in status

    def test_a_run_awaiting_review_says_so(self) -> None:
        from types import SimpleNamespace

        from claimscope.cli import _thread_status

        task = SimpleNamespace(name="review", interrupts=("something",), error=None)
        snapshot = self._snapshot(next_nodes=("review",), tasks=[task])

        assert "awaiting review" in _thread_status(snapshot)  # type: ignore[arg-type]

    def test_an_unfinished_run_names_where_it_stopped(self) -> None:
        """Interrupted with Ctrl-C: pending work, but no error to report."""
        from types import SimpleNamespace

        from claimscope.cli import _thread_status

        task = SimpleNamespace(name="codegen", interrupts=(), error=None)
        snapshot = self._snapshot(next_nodes=("codegen",), tasks=[task])

        status = _thread_status(snapshot)  # type: ignore[arg-type]

        assert "stopped before codegen" in status
