"""Real Docker containment checks, skipped when no daemon is available.

Everything else about the sandbox is tested against a fake client. These are the
tests that prove the container actually confines the code, so they must run
against a real daemon. They are skipped rather than failed where Docker is
absent, and marked ``docker`` so they can be selected or excluded:

    pytest -m docker        # only these
    pytest -m "not docker"  # everything else
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claimscope.config import Settings
from claimscope.sandbox.docker_runner import DockerCPURunner
from claimscope.sandbox.runner import ExecutionRequest, ImageSpec

pytestmark = pytest.mark.docker


def _docker_available() -> bool:
    try:
        import docker

        client = docker.from_env()
        client.ping()
    except Exception:
        return False
    return True


requires_docker = pytest.mark.skipif(not _docker_available(), reason="no reachable Docker daemon")


@pytest.fixture(scope="module")
def prepared_runner() -> DockerCPURunner:
    settings = Settings(docker_image="claimscope-test:latest", sandbox_memory_mb=512)
    runner = DockerCPURunner(settings)
    runner.prepare(ImageSpec())
    return runner


def _workspace(tmp_path: Path, script: str) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    (ws / "run.py").write_text(script, encoding="utf-8")
    return ws


@requires_docker
def test_runs_a_script_and_collects_its_result(
    prepared_runner: DockerCPURunner, tmp_path: Path
) -> None:
    script = """
import argparse, json
p = argparse.ArgumentParser()
p.add_argument("--arm")
p.add_argument("--seed", type=int)
p.add_argument("--steps", type=int, default=None)
a = p.parse_args()
json.dump(
    {"arm": a.arm, "seed": a.seed, "metric": 0.5, "metric_name": "accuracy"},
    open("result.json", "w"),
)
"""
    workspace = _workspace(tmp_path, script)

    result = prepared_runner.execute(
        ExecutionRequest(workspace=workspace, args=["--arm", "a", "--seed", "0"], timeout_s=120)
    )

    assert result.ok, result.stderr
    payload = json.loads((workspace / "result.json").read_text(encoding="utf-8"))
    assert payload["metric"] == 0.5


@requires_docker
def test_the_container_has_no_network(prepared_runner: DockerCPURunner, tmp_path: Path) -> None:
    """The central containment property: generated code cannot phone home."""
    script = """
import socket, sys
socket.setdefaulttimeout(5)
try:
    socket.create_connection(("1.1.1.1", 53))
    print("NETWORK REACHABLE")
    sys.exit(0)
except OSError:
    print("network blocked")
    sys.exit(3)
"""
    result = prepared_runner.execute(
        ExecutionRequest(workspace=_workspace(tmp_path, script), args=[], timeout_s=120)
    )

    assert "NETWORK REACHABLE" not in result.stdout
    assert result.exit_code == 3


@requires_docker
def test_the_container_does_not_run_as_root(
    prepared_runner: DockerCPURunner, tmp_path: Path
) -> None:
    script = "import os; print(os.getuid())"

    result = prepared_runner.execute(
        ExecutionRequest(workspace=_workspace(tmp_path, script), args=[], timeout_s=120)
    )

    assert result.stdout.strip() != "0"


@requires_docker
def test_the_root_filesystem_is_read_only(prepared_runner: DockerCPURunner, tmp_path: Path) -> None:
    script = """
import sys
try:
    open("/etc/evil", "w").write("x")
    print("WROTE OUTSIDE WORKSPACE")
    sys.exit(0)
except OSError:
    print("blocked")
    sys.exit(3)
"""
    result = prepared_runner.execute(
        ExecutionRequest(workspace=_workspace(tmp_path, script), args=[], timeout_s=120)
    )

    assert "WROTE OUTSIDE WORKSPACE" not in result.stdout
    assert result.exit_code == 3


@requires_docker
def test_the_workspace_stays_writable(prepared_runner: DockerCPURunner, tmp_path: Path) -> None:
    script = "open('output.txt', 'w').write('written')"
    workspace = _workspace(tmp_path, script)

    result = prepared_runner.execute(ExecutionRequest(workspace=workspace, args=[], timeout_s=120))

    assert result.ok, result.stderr
    assert (workspace / "output.txt").read_text(encoding="utf-8") == "written"


@requires_docker
def test_a_hanging_script_is_killed(prepared_runner: DockerCPURunner, tmp_path: Path) -> None:
    script = "import time; time.sleep(600)"

    result = prepared_runner.execute(
        ExecutionRequest(workspace=_workspace(tmp_path, script), args=[], timeout_s=5)
    )

    assert result.timed_out
    assert not result.ok
