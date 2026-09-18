"""Docker CPU sandbox (PLAN.md section 7).

Generated code is untrusted, so the container runs as a non-root user with no
network, a read-only root filesystem, dropped capabilities, and limits on CPU,
memory, processes and wall-clock time. Only the claim's workspace is mounted.
"""

from __future__ import annotations

import io
import logging
import tarfile
import time
from typing import TYPE_CHECKING, Any

from claimscope.sandbox.runner import (
    ExecutionRequest,
    ExecutionResult,
    ImageSpec,
    SandboxError,
)

if TYPE_CHECKING:
    from claimscope.config import Settings

logger = logging.getLogger(__name__)

WORKSPACE_MOUNT = "/workspace"

# Written into the image at build time. Kept here rather than as a file on disk
# so the image definition travels with the code that depends on it.
DOCKERFILE_CPU = """\
FROM python:3.11-slim

# A non-root user owns the workspace; generated code never runs as root.
RUN useradd --create-home --uid 1000 runner

RUN pip install --no-cache-dir --upgrade pip
{package_install}

USER runner
WORKDIR /workspace
"""


def render_dockerfile(spec: ImageSpec) -> str:
    """Produce the Dockerfile for an image with these packages preinstalled.

    Dependencies are installed at build time, so the experiment itself can run
    with the network disabled.
    """
    if spec.packages:
        packages = " ".join(sorted(spec.packages))
        install = f"RUN pip install --no-cache-dir {packages}"
    else:
        install = "# no extra packages"
    return DOCKERFILE_CPU.format(package_install=install)


class DockerCPURunner:
    """Runs experiments in a locked-down CPU container."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client
        self._image_tag = settings.docker_image

    def _get_client(self) -> Any:
        if self._client is None:
            import docker

            try:
                self._client = docker.from_env()
                self._client.ping()
            except Exception as exc:
                raise SandboxError(
                    "Cannot reach the Docker daemon. Is Docker Desktop running?"
                ) from exc
        return self._client

    def available(self) -> bool:
        """Whether Docker is reachable, without raising."""
        try:
            self._get_client()
        except SandboxError:
            return False
        return True

    def prepare(self, spec: ImageSpec) -> None:
        """Build the experiment image. This is the only step allowed network access."""
        client = self._get_client()
        dockerfile = render_dockerfile(spec)
        logger.info("building image %s with packages=%s", self._image_tag, spec.packages)

        context = _tar_context({"Dockerfile": dockerfile})
        try:
            _image, logs = client.images.build(
                fileobj=context,
                custom_context=True,
                tag=self._image_tag,
                rm=True,
                forcerm=True,
            )
        except Exception as exc:
            raise SandboxError(f"failed to build sandbox image: {exc}") from exc

        for entry in logs:
            if stream := entry.get("stream", "").strip():
                logger.debug("docker build: %s", stream)

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Run one experiment under the sandbox's limits."""
        client = self._get_client()
        workspace = request.workspace.resolve()
        if not workspace.is_dir():
            raise SandboxError(f"workspace does not exist: {workspace}")

        settings = self._settings
        command = ["python", "run.py", *request.args]

        started = time.monotonic()
        container = None
        try:
            container = client.containers.run(
                self._image_tag,
                command=command,
                detach=True,
                # Only the claim's workspace is visible to the experiment.
                volumes={str(workspace): {"bind": WORKSPACE_MOUNT, "mode": "rw"}},
                working_dir=WORKSPACE_MOUNT,
                user="runner",
                network_disabled=not request.network,
                mem_limit=f"{settings.sandbox_memory_mb}m",
                nano_cpus=int(settings.sandbox_cpus * 1_000_000_000),
                pids_limit=256,
                read_only=True,
                # The workspace mount stays writable; everything else does not.
                tmpfs={"/tmp": "size=256m"},
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                environment={"PYTHONUNBUFFERED": "1", "HOME": "/tmp"},
            )

            timed_out = False
            try:
                status = container.wait(timeout=request.timeout_s)
                exit_code = int(status.get("StatusCode", 1))
            except Exception:
                # wait() raises on timeout; kill the container and report it.
                timed_out = True
                exit_code = 124
                _safe_kill(container)

            stdout = _decode(container.logs(stdout=True, stderr=False))
            stderr = _decode(container.logs(stdout=False, stderr=True))
            duration = time.monotonic() - started

            return ExecutionResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_s=duration,
                timed_out=timed_out,
            )
        finally:
            if container is not None:
                _safe_remove(container)


def _tar_context(files: dict[str, str]) -> io.BytesIO:
    """Build an in-memory tar to use as a Docker build context."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, content in files.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buffer.seek(0)
    return buffer


def _decode(raw: bytes | str) -> str:
    return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw


def _safe_kill(container: Any) -> None:
    try:
        container.kill()
    except Exception:  # the container may already be gone
        logger.debug("could not kill container", exc_info=True)


def _safe_remove(container: Any) -> None:
    try:
        container.remove(force=True)
    except Exception:  # cleanup is best effort
        logger.debug("could not remove container", exc_info=True)


def build_runner(settings: Settings, client: Any | None = None) -> DockerCPURunner:
    """The runner for this configuration.

    Only a CPU backend exists today; a GPU runner would be selected here.
    """
    return DockerCPURunner(settings, client)
