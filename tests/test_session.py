"""Checkpoint storage details that are easy to break silently."""

from __future__ import annotations

import dataclasses
import inspect

from pydantic import BaseModel

from claimscope import schemas
from claimscope.config import Settings
from claimscope.sandbox import runner as runner_module
from claimscope.session import _ALLOWED_MODULES, checkpoint_path, checkpointer, new_thread_id

# Sandbox dataclasses the graph state stores. The others (ExecutionRequest,
# ImageSpec) are inputs to the runner and never reach a checkpoint.
_SERIALIZED_SANDBOX_TYPES = {"ExecutionFailure", "ExecutionResult"}


def _state_carrying_types() -> set[str]:
    """Every project type that can end up inside a checkpoint."""
    names = {
        name
        for name, obj in inspect.getmembers(schemas, inspect.isclass)
        if issubclass(obj, BaseModel) and obj.__module__ == "claimscope.schemas"
    }
    return names | _SERIALIZED_SANDBOX_TYPES


def test_serialized_sandbox_types_still_exist() -> None:
    """Keep the hand-maintained list above honest."""
    for name in _SERIALIZED_SANDBOX_TYPES:
        obj = getattr(runner_module, name, None)
        assert obj is not None, f"{name} no longer exists in sandbox.runner"
        assert dataclasses.is_dataclass(obj), f"{name} is no longer a dataclass"


def test_every_schema_is_registered_for_deserialization() -> None:
    """Guard against adding a type and breaking resume.

    LangGraph will refuse to deserialize unregistered types in a future version,
    which would make interrupted runs unresumable. Anything that can reach the
    graph state must be in the allowlist.
    """
    registered = {name for _module, name in _ALLOWED_MODULES}
    defined = _state_carrying_types()

    assert defined <= registered, (
        f"types missing from the allowlist: {sorted(defined - registered)}"
    )


def test_allowlist_has_no_stale_entries() -> None:
    registered = {name for _module, name in _ALLOWED_MODULES}
    defined = _state_carrying_types()

    assert registered <= defined, (
        f"allowlist names no longer defined: {sorted(registered - defined)}"
    )


def test_checkpoint_lives_under_the_runs_directory(settings: Settings) -> None:
    assert checkpoint_path(settings) == settings.runs_dir / "checkpoints.sqlite"


def test_opening_the_checkpointer_creates_its_directory(settings: Settings) -> None:
    assert not settings.runs_dir.exists()

    with checkpointer(settings):
        pass

    assert checkpoint_path(settings).exists()


def test_thread_ids_are_unique_and_readable() -> None:
    first = new_thread_id("1706.03762")
    second = new_thread_id("1706.03762")

    assert first.startswith("1706.03762-")
    assert first != second
