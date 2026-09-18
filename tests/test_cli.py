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
