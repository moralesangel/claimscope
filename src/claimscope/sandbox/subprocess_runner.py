"""A weaker sandbox for environments without Docker, such as Google Colab.

**This does not contain hostile code.** It runs the experiment in a separate
process with network calls blocked, memory and CPU capped, a scrubbed
environment, and only the claim's workspace as its working directory. A
determined script can defeat every one of those from inside the process.

It exists so the pipeline can be exercised end to end where Docker is
unavailable. It is opt-in, never the default, and every run through it is
recorded as unconfined so the report can say so.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from claimscope.sandbox.preamble import build_preamble
from claimscope.sandbox.runner import (
    ExecutionRequest,
    ExecutionResult,
    ImageSpec,
    SandboxError,
)

if TYPE_CHECKING:
    from claimscope.config import Settings

logger = logging.getLogger(__name__)

# Python imports sitecustomize automatically at startup if it is on the path,
# which is how the restrictions get applied before the experiment's own code.
PREAMBLE_FILENAME = "sitecustomize.py"

WARNING = (
    "Running generated code WITHOUT container isolation. This backend blocks "
    "network access and caps resources, but cannot contain hostile code. Use it "
    "only for code you are willing to run on this machine."
)

# Environment variables a scientific stack legitimately needs. Everything else
# is dropped, so credentials in the parent environment are not inherited.
_ENV_ALLOWLIST = (
    "PATH",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "TEMP",
    "TMP",
    "SYSTEMROOT",
    "PYTHONHASHSEED",
)


class SubprocessRunner:
    """Runs experiments in a restricted subprocess."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._warned = False

    def available(self) -> bool:
        """Always usable: it needs nothing but a Python interpreter."""
        return True

    def prepare(self, spec: ImageSpec) -> None:
        """Install the experiment's dependencies into the current environment.

        Unlike the Docker backend there is no image to build, so packages land
        in the environment this process is running in. That is a real side
        effect, and another reason this backend is opt-in.
        """
        if not self._warned:
            logger.warning(WARNING)
            self._warned = True

        if not spec.packages:
            return

        missing = [name for name in spec.packages if not _is_installed(name)]
        if not missing:
            logger.info("all sandbox packages already present")
            return

        logger.info("installing into the current environment: %s", ", ".join(missing))
        for command in _install_commands(missing):
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, check=False, timeout=900
                )
            except (OSError, subprocess.SubprocessError) as exc:
                logger.debug("%s failed to start: %s", command[0], exc)
                continue

            if result.returncode == 0:
                still_missing = [name for name in missing if not _is_installed(name)]
                if still_missing:
                    logger.warning(
                        "installed but not importable: %s; the experiment must work without them",
                        ", ".join(still_missing),
                    )
                else:
                    logger.info("installed %s", ", ".join(missing))
                return

            logger.debug("%s: %s", command[0], (result.stderr or result.stdout).strip()[:200])

        # Research repos pin versions that often will not resolve, and some
        # environments have neither pip nor uv. The codegen prompt tells the
        # model to work with what is available, so this is a warning rather than
        # a dead run -- but a loud one, because an experiment that silently
        # falls back to random data measures nothing.
        logger.warning(
            "COULD NOT INSTALL %s. The experiment will have to work without them, which "
            "usually means falling back to synthetic data and measuring nothing. Install "
            "them yourself, or use the Docker backend, which builds its own image.",
            ", ".join(missing),
        )

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Run one experiment with the restrictions this backend can apply."""
        workspace = request.workspace.resolve()
        if not workspace.is_dir():
            raise SandboxError(f"workspace does not exist: {workspace}")

        script = workspace / "run.py"
        if not script.is_file():
            raise SandboxError(f"no run.py in {workspace}")

        settings = self._settings
        if request.network:
            # Only dependency installation asks for this, and prepare() does not
            # go through execute(), so treat it as a mistake rather than obeying.
            raise SandboxError("the subprocess backend never grants network access to experiments")

        # Kept outside the workspace so the experiment does not see it as a
        # local module, and so it is never mistaken for generated code.
        preamble_dir = workspace.parent / f".sandbox_{workspace.name}"
        preamble_dir.mkdir(parents=True, exist_ok=True)
        preamble_path = preamble_dir / PREAMBLE_FILENAME
        preamble_path.write_text(
            build_preamble(settings.sandbox_memory_mb, request.timeout_s), encoding="utf-8"
        )

        started = time.monotonic()
        try:
            completed = subprocess.run(
                [sys.executable, "run.py", *request.args],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=request.timeout_s,
                check=False,
                env=_child_env(preamble_path),
                start_new_session=True,  # a timeout kills the whole group
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                exit_code=124,
                stdout=_decode(exc.stdout),
                stderr=_decode(exc.stderr),
                duration_s=time.monotonic() - started,
                timed_out=True,
            )
        finally:
            preamble_path.unlink(missing_ok=True)

        return ExecutionResult(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_s=time.monotonic() - started,
        )


def _install_commands(packages: list[str]) -> list[list[str]]:
    """Ways to install into this environment, best first.

    uv-managed virtualenvs ship no pip at all, so pip alone is not enough --
    this project's own venv is one of them.
    """
    commands: list[list[str]] = []

    if uv := shutil.which("uv"):
        # --python pins the target to this interpreter, not uv's default.
        commands.append([uv, "pip", "install", "--python", sys.executable, "--quiet", *packages])
    commands.append([sys.executable, "-m", "pip", "install", "--quiet", *packages])
    return commands


def _child_env(preamble_path: Path) -> dict[str, str]:
    """A minimal environment that preloads the restriction preamble.

    PYTHONSTARTUP only applies to interactive sessions, so the preamble is
    injected via sitecustomize on the path instead.
    """
    env = {name: os.environ[name] for name in _ENV_ALLOWLIST if name in os.environ}
    env["PYTHONPATH"] = str(preamble_path.parent)
    env["PYTHONUNBUFFERED"] = "1"
    env["CLAIMSCOPE_SANDBOX"] = "subprocess"
    # Offline flags the common ML libraries respect, as a second line of defence.
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def _is_installed(requirement: str) -> bool:
    """Whether a requirement's distribution is already importable."""
    from importlib.metadata import PackageNotFoundError, version

    name = _requirement_name(requirement)
    try:
        version(name)
    except PackageNotFoundError:
        return False
    return True


def _requirement_name(requirement: str) -> str:
    """Strip version specifiers and extras from a requirement line."""
    for separator in ("==", ">=", "<=", "~=", ">", "<", "!=", "["):
        requirement = requirement.split(separator)[0]
    return requirement.strip()


def _decode(raw: bytes | str | None) -> str:
    if raw is None:
        return ""
    return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
