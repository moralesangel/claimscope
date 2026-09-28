"""The sandbox's security properties, verified against a fake Docker client.

Docker itself is not available in every environment, so these assert the
arguments the runner *asks for*. Actually running a container is covered by the
integration test in test_sandbox_integration.py, skipped without a daemon.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from claimscope.config import Settings
from claimscope.sandbox import docker_runner
from claimscope.sandbox.docker_runner import (
    WORKSPACE_MOUNT,
    DockerCPURunner,
    render_dockerfile,
)
from claimscope.sandbox.runner import ExecutionRequest, ImageSpec, SandboxError


class FakeContainer:
    def __init__(self, exit_code: int = 0, wait_raises: bool = False) -> None:
        self.exit_code = exit_code
        self.wait_raises = wait_raises
        self.killed = False
        self.removed = False

    def wait(self, timeout: int | None = None) -> dict[str, int]:
        if self.wait_raises:
            raise TimeoutError("timed out")
        return {"StatusCode": self.exit_code}

    def logs(self, stdout: bool = True, stderr: bool = False) -> bytes:
        return b"out" if stdout else b"err"

    def kill(self) -> None:
        self.killed = True

    def remove(self, force: bool = False) -> None:
        self.removed = True


class FakeContainers:
    def __init__(self, container: FakeContainer) -> None:
        self.container = container
        self.kwargs: dict[str, Any] = {}
        self.image: str | None = None

    def run(self, image: str, **kwargs: Any) -> FakeContainer:
        self.image = image
        self.kwargs = kwargs
        return self.container


class FakeClient:
    def __init__(self, container: FakeContainer | None = None) -> None:
        self.containers = FakeContainers(container or FakeContainer())
        self.images = FakeImages()

    def ping(self) -> bool:
        return True


class FakeImages:
    def __init__(self) -> None:
        self.build_kwargs: dict[str, Any] = {}

    def build(self, **kwargs: Any) -> tuple[object, list[dict[str, str]]]:
        self.build_kwargs = kwargs
        return object(), [{"stream": "Successfully built"}]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "run.py").write_text("print('hi')", encoding="utf-8")
    return ws


def _request(workspace: Path, **overrides: Any) -> ExecutionRequest:
    kwargs: dict[str, Any] = {
        "workspace": workspace,
        "args": ["--arm", "treatment", "--seed", "0"],
        "timeout_s": 60,
    }
    kwargs.update(overrides)
    return ExecutionRequest(**kwargs)


class TestSecurityLimits:
    """Every one of these is a containment property the plan requires."""

    def _run_and_capture(self, settings: Settings, workspace: Path) -> dict[str, Any]:
        client = FakeClient()
        DockerCPURunner(settings, client).execute(_request(workspace))
        return client.containers.kwargs

    def test_runs_as_a_non_root_user(self, settings: Settings, workspace: Path) -> None:
        # Either the image's own user, or this process's uid:gid so the
        # bind-mounted workspace stays writable. Never root, either way.
        user = self._run_and_capture(settings, workspace)["user"]
        assert user == "runner" or user.split(":")[0] not in ("0", "")

    def test_never_hands_the_experiment_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Running claimscope itself as root must not make the sandbox root too.
        monkeypatch.setattr(docker_runner.os, "getuid", lambda: 0, raising=False)
        monkeypatch.setattr(docker_runner.os, "getgid", lambda: 0, raising=False)

        assert docker_runner._container_user() == "runner"

    def test_matches_the_host_uid_so_the_workspace_is_writable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The bind mount keeps the host's ownership, so a mismatched uid cannot
        # write result.json, which is how every run under Docker would fail.
        monkeypatch.setattr(docker_runner.os, "getuid", lambda: 1001, raising=False)
        monkeypatch.setattr(docker_runner.os, "getgid", lambda: 1002, raising=False)

        assert docker_runner._container_user() == "1001:1002"

    def test_network_is_disabled(self, settings: Settings, workspace: Path) -> None:
        assert self._run_and_capture(settings, workspace)["network_disabled"] is True

    def test_root_filesystem_is_read_only(self, settings: Settings, workspace: Path) -> None:
        assert self._run_and_capture(settings, workspace)["read_only"] is True

    def test_all_capabilities_are_dropped(self, settings: Settings, workspace: Path) -> None:
        assert self._run_and_capture(settings, workspace)["cap_drop"] == ["ALL"]

    def test_privilege_escalation_is_blocked(self, settings: Settings, workspace: Path) -> None:
        assert "no-new-privileges" in self._run_and_capture(settings, workspace)["security_opt"]

    def test_memory_and_cpu_are_capped(self, settings: Settings, workspace: Path) -> None:
        kwargs = self._run_and_capture(settings, workspace)
        assert kwargs["mem_limit"] == f"{settings.sandbox_memory_mb}m"
        assert kwargs["nano_cpus"] == int(settings.sandbox_cpus * 1_000_000_000)

    def test_process_count_is_capped(self, settings: Settings, workspace: Path) -> None:
        assert self._run_and_capture(settings, workspace)["pids_limit"] == 256

    def test_only_the_workspace_is_mounted(self, settings: Settings, workspace: Path) -> None:
        volumes = self._run_and_capture(settings, workspace)["volumes"]

        assert len(volumes) == 1
        assert volumes[str(workspace.resolve())] == {"bind": WORKSPACE_MOUNT, "mode": "rw"}

    def test_network_can_be_enabled_explicitly(self, settings: Settings, workspace: Path) -> None:
        # Only dependency installation should ever ask for this.
        client = FakeClient()
        DockerCPURunner(settings, client).execute(_request(workspace, network=True))

        assert client.containers.kwargs["network_disabled"] is False


class TestExecution:
    def test_passes_arguments_to_run_py(self, settings: Settings, workspace: Path) -> None:
        client = FakeClient()
        DockerCPURunner(settings, client).execute(_request(workspace))

        assert client.containers.kwargs["command"] == [
            "python",
            "run.py",
            "--arm",
            "treatment",
            "--seed",
            "0",
        ]

    def test_successful_run_reports_ok(self, settings: Settings, workspace: Path) -> None:
        result = DockerCPURunner(settings, FakeClient()).execute(_request(workspace))

        assert result.ok
        assert result.exit_code == 0
        assert result.stdout == "out"
        assert result.stderr == "err"

    def test_failing_run_is_not_ok(self, settings: Settings, workspace: Path) -> None:
        client = FakeClient(FakeContainer(exit_code=1))

        result = DockerCPURunner(settings, client).execute(_request(workspace))

        assert not result.ok
        assert result.failure_summary() == "exit code 1"

    def test_timeout_kills_the_container(self, settings: Settings, workspace: Path) -> None:
        container = FakeContainer(wait_raises=True)
        client = FakeClient(container)

        result = DockerCPURunner(settings, client).execute(_request(workspace))

        assert result.timed_out
        assert not result.ok
        assert container.killed
        assert "timed out" in result.failure_summary()

    def test_container_is_always_removed(self, settings: Settings, workspace: Path) -> None:
        container = FakeContainer(exit_code=1)
        client = FakeClient(container)

        DockerCPURunner(settings, client).execute(_request(workspace))

        assert container.removed

    def test_missing_workspace_is_a_sandbox_error(self, settings: Settings, tmp_path: Path) -> None:
        runner = DockerCPURunner(settings, FakeClient())

        with pytest.raises(SandboxError, match="workspace does not exist"):
            runner.execute(_request(tmp_path / "absent"))


class TestDockerfile:
    def test_creates_a_non_root_user(self) -> None:
        assert "useradd" in render_dockerfile(ImageSpec())
        assert "USER runner" in render_dockerfile(ImageSpec())

    def test_installs_requested_packages(self) -> None:
        dockerfile = render_dockerfile(ImageSpec(packages=["numpy", "scikit-learn"]))

        assert "pip install --no-cache-dir numpy scikit-learn" in dockerfile

    def test_handles_no_extra_packages(self) -> None:
        assert "no extra packages" in render_dockerfile(ImageSpec())

    def test_package_order_is_stable(self) -> None:
        # Same packages in any order must produce the same image definition.
        first = render_dockerfile(ImageSpec(packages=["b", "a"]))
        second = render_dockerfile(ImageSpec(packages=["a", "b"]))

        assert first == second


class TestAvailability:
    def test_available_when_the_daemon_answers(self, settings: Settings) -> None:
        assert DockerCPURunner(settings, FakeClient()).available()

    def test_unavailable_when_the_daemon_is_unreachable(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import docker

        def explode(*_args: object, **_kwargs: object) -> None:
            raise OSError("cannot connect to the Docker daemon")

        monkeypatch.setattr(docker, "from_env", explode)

        assert DockerCPURunner(settings).available() is False

    def test_execute_without_a_daemon_raises_sandbox_error(
        self, settings: Settings, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import docker

        def explode(*_args: object, **_kwargs: object) -> None:
            raise OSError("cannot connect to the Docker daemon")

        monkeypatch.setattr(docker, "from_env", explode)

        with pytest.raises(SandboxError, match="Docker daemon"):
            DockerCPURunner(settings).execute(_request(workspace))
