"""The sandbox contract (PLAN.md section 7).

Generated code is never executed outside a sandbox, so every backend implements
this interface. Only ``DockerCPURunner`` exists today; the interface is shaped to
accommodate a future GPU runner, local or remote, without changing callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from claimscope.config import Settings


class SandboxError(RuntimeError):
    """The sandbox itself failed, as opposed to the experiment failing."""


@dataclass(frozen=True)
class ExecutionRequest:
    """One invocation of a generated experiment."""

    workspace: Path
    """Host directory holding run.py and its outputs. The only mount."""

    args: list[str]
    """Arguments to run.py, e.g. ["--arm", "treatment", "--seed", "0"]."""

    timeout_s: int

    network: bool = False
    """Experiments run without network. Only dependency installation needs it."""


@dataclass
class ExecutionResult:
    """What came back from one sandboxed run."""

    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def failure_summary(self) -> str:
        """A short description of why this run failed, for logs and the debug node."""
        if self.timed_out:
            return f"timed out after {self.duration_s:.0f}s"
        return f"exit code {self.exit_code}"


@dataclass
class ExecutionFailure:
    """A failed run, carried to the debug node so it can propose a fix."""

    claim_id: str
    arm: str
    seed: int
    result: ExecutionResult

    def traceback_text(self) -> str:
        """The part of the output a debugger needs."""
        return (self.result.stderr or self.result.stdout or "").strip()


@dataclass
class ImageSpec:
    """What the experiment needs installed before it can run."""

    packages: list[str] = field(default_factory=list)
    """Extra pip packages beyond the base image."""


class Runner(Protocol):
    """A sandbox that can prepare an environment and execute experiments in it."""

    def prepare(self, spec: ImageSpec) -> None:
        """Build or pull the environment. This step may use the network."""
        ...

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Run one experiment. Raises SandboxError if the sandbox is unusable."""
        ...

    def available(self) -> bool:
        """Whether this backend can run right now."""
        ...


def build_runner(settings: Settings, client: Any | None = None) -> Runner:
    """The sandbox for this configuration.

    Docker is the default and the only backend that truly contains what it runs.
    The subprocess backend is for environments without Docker, such as Colab,
    and has to be asked for explicitly.
    """
    if settings.sandbox_backend == "subprocess":
        from claimscope.sandbox.subprocess_runner import SubprocessRunner

        return SubprocessRunner(settings)

    from claimscope.sandbox.docker_runner import DockerCPURunner

    return DockerCPURunner(settings, client)


def is_contained(settings: Settings) -> bool:
    """Whether the configured sandbox actually contains what it runs.

    The report says so when it does not, since a result produced by unconfined
    execution carries a caveat the reader deserves to see.
    """
    return settings.sandbox_backend == "docker"
