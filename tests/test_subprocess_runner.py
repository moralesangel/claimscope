"""The subprocess sandbox, exercised by actually running code in it.

Unlike the Docker tests, these do not need a daemon, so they run everywhere and
verify the restrictions for real rather than checking arguments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from claimscope.config import Settings
from claimscope.sandbox.preamble import build_preamble
from claimscope.sandbox.runner import ExecutionRequest, ImageSpec, SandboxError
from claimscope.sandbox.subprocess_runner import SubprocessRunner


def _workspace(tmp_path: Path, script: str) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    (ws / "run.py").write_text(script, encoding="utf-8")
    return ws


def _request(workspace: Path, args: list[str] | None = None, timeout_s: int = 60):
    return ExecutionRequest(workspace=workspace, args=args or [], timeout_s=timeout_s)


@pytest.fixture
def runner(settings: Settings) -> SubprocessRunner:
    return SubprocessRunner(settings)


class TestRunning:
    def test_runs_a_script_and_captures_output(
        self, runner: SubprocessRunner, tmp_path: Path
    ) -> None:
        workspace = _workspace(tmp_path, "print('hello from the sandbox')")

        result = runner.execute(_request(workspace))

        assert result.ok
        assert "hello from the sandbox" in result.stdout

    def test_passes_arguments_through(self, runner: SubprocessRunner, tmp_path: Path) -> None:
        script = (
            "import argparse\n"
            "p = argparse.ArgumentParser()\n"
            "p.add_argument('--arm')\n"
            "p.add_argument('--seed', type=int)\n"
            "a = p.parse_args()\n"
            "print(f'{a.arm}:{a.seed}')\n"
        )
        workspace = _workspace(tmp_path, script)

        result = runner.execute(_request(workspace, ["--arm", "treatment", "--seed", "3"]))

        assert "treatment:3" in result.stdout

    def test_the_experiment_can_write_result_json(
        self, runner: SubprocessRunner, tmp_path: Path
    ) -> None:
        script = (
            "import json\n"
            "json.dump({'arm': 'a', 'seed': 0, 'metric': 0.5, 'metric_name': 'acc'},\n"
            "          open('result.json', 'w'))\n"
        )
        workspace = _workspace(tmp_path, script)

        runner.execute(_request(workspace))

        payload = json.loads((workspace / "result.json").read_text(encoding="utf-8"))
        assert payload["metric"] == 0.5

    def test_a_failing_script_reports_its_traceback(
        self, runner: SubprocessRunner, tmp_path: Path
    ) -> None:
        workspace = _workspace(tmp_path, "raise ValueError('deliberate failure')")

        result = runner.execute(_request(workspace))

        assert not result.ok
        assert "deliberate failure" in result.stderr

    def test_a_hanging_script_is_killed(self, runner: SubprocessRunner, tmp_path: Path) -> None:
        workspace = _workspace(tmp_path, "import time; time.sleep(120)")

        result = runner.execute(_request(workspace, timeout_s=3))

        assert result.timed_out
        assert not result.ok


class TestNetworkBlocking:
    """The restriction that matters most: generated code must not download."""

    def test_socket_connections_are_blocked(self, runner: SubprocessRunner, tmp_path: Path) -> None:
        script = (
            "import socket, sys\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 53), timeout=5)\n"
            "    print('NETWORK REACHABLE')\n"
            "except OSError as e:\n"
            "    print(f'blocked: {e}')\n"
            "    sys.exit(3)\n"
        )
        workspace = _workspace(tmp_path, script)

        result = runner.execute(_request(workspace))

        assert "NETWORK REACHABLE" not in result.stdout
        assert result.exit_code == 3

    def test_name_resolution_is_blocked(self, runner: SubprocessRunner, tmp_path: Path) -> None:
        # A hostname should fail fast rather than after a DNS timeout.
        script = (
            "import socket, sys\n"
            "try:\n"
            "    socket.getaddrinfo('example.com', 80)\n"
            "    print('RESOLVED')\n"
            "except OSError:\n"
            "    sys.exit(3)\n"
        )
        workspace = _workspace(tmp_path, script)

        result = runner.execute(_request(workspace))

        assert "RESOLVED" not in result.stdout
        assert result.exit_code == 3

    def test_urllib_downloads_are_blocked(self, runner: SubprocessRunner, tmp_path: Path) -> None:
        script = (
            "import sys, urllib.request\n"
            "try:\n"
            "    urllib.request.urlopen('http://example.com')\n"
            "    print('DOWNLOADED')\n"
            "except OSError:\n"
            "    sys.exit(3)\n"
        )
        workspace = _workspace(tmp_path, script)

        result = runner.execute(_request(workspace))

        assert "DOWNLOADED" not in result.stdout
        assert result.exit_code == 3

    def test_the_error_explains_what_to_do(self, runner: SubprocessRunner, tmp_path: Path) -> None:
        """The message reaches the debug node, so it should be actionable."""
        script = "import socket\nsocket.create_connection(('1.1.1.1', 53))\n"
        workspace = _workspace(tmp_path, script)

        result = runner.execute(_request(workspace))

        assert "Network access is disabled" in result.stderr


class TestEnvironmentScrubbing:
    def test_parent_secrets_are_not_inherited(
        self, runner: SubprocessRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-value-do-not-leak")
        script = "import os; print(os.environ.get('ANTHROPIC_API_KEY', 'ABSENT'))"
        workspace = _workspace(tmp_path, script)

        result = runner.execute(_request(workspace))

        assert "secret-value-do-not-leak" not in result.stdout
        assert "ABSENT" in result.stdout

    def test_the_sandbox_marks_itself(self, runner: SubprocessRunner, tmp_path: Path) -> None:
        script = "import os; print(os.environ.get('CLAIMSCOPE_SANDBOX', 'none'))"
        workspace = _workspace(tmp_path, script)

        result = runner.execute(_request(workspace))

        assert "subprocess" in result.stdout

    def test_offline_flags_are_set_for_ml_libraries(
        self, runner: SubprocessRunner, tmp_path: Path
    ) -> None:
        script = "import os; print(os.environ.get('HF_HUB_OFFLINE'))"
        workspace = _workspace(tmp_path, script)

        assert "1" in runner.execute(_request(workspace)).stdout


@pytest.mark.skipif(sys.platform == "win32", reason="resource limits are POSIX only")
class TestResourceLimits:
    def test_memory_is_capped(self, tmp_path: Path) -> None:
        settings = Settings(runs_dir=tmp_path / "runs", sandbox_memory_mb=128)
        script = (
            "import sys\n"
            "try:\n"
            "    big = bytearray(512 * 1024 * 1024)\n"
            "    print('ALLOCATED')\n"
            "except MemoryError:\n"
            "    sys.exit(3)\n"
        )
        workspace = _workspace(tmp_path, script)

        result = SubprocessRunner(settings).execute(_request(workspace))

        assert "ALLOCATED" not in result.stdout


class TestGuards:
    def test_a_missing_workspace_raises(self, runner: SubprocessRunner, tmp_path: Path) -> None:
        with pytest.raises(SandboxError, match="workspace does not exist"):
            runner.execute(_request(tmp_path / "absent"))

    def test_a_missing_run_py_raises(self, runner: SubprocessRunner, tmp_path: Path) -> None:
        empty = tmp_path / "ws"
        empty.mkdir()

        with pytest.raises(SandboxError, match=r"no run\.py"):
            runner.execute(_request(empty))

    def test_network_is_never_granted(self, runner: SubprocessRunner, tmp_path: Path) -> None:
        workspace = _workspace(tmp_path, "print('x')")
        request = ExecutionRequest(workspace=workspace, args=[], timeout_s=60, network=True)

        with pytest.raises(SandboxError, match="never grants network access"):
            runner.execute(request)

    def test_it_is_always_available(self, runner: SubprocessRunner) -> None:
        assert runner.available()


class TestRunnerInterface:
    """A misplaced helper once pushed execute() out of the class body.

    Nothing caught it until a real run failed with "no attribute 'execute'",
    because every other test constructs the runner and calls one method.
    """

    def test_it_satisfies_the_runner_protocol(self, settings: Settings) -> None:
        from claimscope.sandbox.runner import Runner

        runner: Runner = SubprocessRunner(settings)

        for method in ("prepare", "execute", "available"):
            assert callable(getattr(runner, method, None)), f"missing {method}"

    def test_build_runner_returns_a_usable_backend(self, settings: Settings) -> None:
        from claimscope.sandbox.runner import build_runner

        runner = build_runner(settings.model_copy(update={"sandbox_backend": "subprocess"}))

        assert isinstance(runner, SubprocessRunner)
        assert callable(runner.execute)


class TestPreamble:
    def test_includes_both_restrictions(self) -> None:
        preamble = build_preamble(memory_mb=256, cpu_seconds=60)

        assert "socket" in preamble
        assert "RLIMIT_AS" in preamble

    def test_memory_is_expressed_in_bytes(self) -> None:
        assert str(256 * 1024 * 1024) in build_preamble(memory_mb=256, cpu_seconds=60)

    def test_it_is_valid_python(self) -> None:
        compile(build_preamble(memory_mb=256, cpu_seconds=60), "<preamble>", "exec")


class TestPrepare:
    def test_warns_that_this_is_not_containment(
        self, runner: SubprocessRunner, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        with caplog.at_level(logging.WARNING):
            runner.prepare(ImageSpec())

        assert "WITHOUT container isolation" in caplog.text

    def test_skips_packages_that_are_already_present(
        self, runner: SubprocessRunner, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        with caplog.at_level(logging.INFO):
            runner.prepare(ImageSpec(packages=["numpy"]))

        assert "already present" in caplog.text
