"""Checkpoint storage details that are easy to break silently."""

from __future__ import annotations

import inspect

from pydantic import BaseModel

from claimscope import schemas
from claimscope.config import Settings
from claimscope.session import _ALLOWED_MODULES, checkpoint_path, checkpointer, new_thread_id


def test_every_schema_is_registered_for_deserialization() -> None:
    """Guard against adding a schema and breaking resume.

    LangGraph will refuse to deserialize unregistered types in a future version,
    which would make interrupted runs unresumable. Anything that can reach the
    graph state must be in the allowlist.
    """
    defined = {
        name
        for name, obj in inspect.getmembers(schemas, inspect.isclass)
        if issubclass(obj, BaseModel) and obj.__module__ == "claimscope.schemas"
    }
    registered = {name for _module, name in _ALLOWED_MODULES}

    assert defined <= registered, (
        f"schemas missing from the allowlist: {sorted(defined - registered)}"
    )


def test_allowlist_has_no_stale_entries() -> None:
    registered = {name for _module, name in _ALLOWED_MODULES}
    defined = {
        name
        for name, obj in inspect.getmembers(schemas, inspect.isclass)
        if issubclass(obj, BaseModel) and obj.__module__ == "claimscope.schemas"
    }

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
