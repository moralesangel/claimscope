"""Checkpoint storage, so an interrupted run can be resumed in a new process."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

if TYPE_CHECKING:
    from claimscope.config import Settings

CHECKPOINT_FILENAME = "checkpoints.sqlite"

# Our Pydantic models are stored inside checkpoints. LangGraph warns about
# deserializing unregistered types today and will refuse them in a future
# version, so declare them explicitly rather than relying on the permissive
# default. Add every schema that can reach the graph state.
_ALLOWED_MODULES: tuple[tuple[str, str], ...] = (
    ("claimscope.schemas", "Claim"),
    ("claimscope.schemas", "ClaimList"),
    ("claimscope.schemas", "ClaimVerdict"),
    ("claimscope.schemas", "PaperSource"),
    ("claimscope.schemas", "ReductionPlan"),
    ("claimscope.schemas", "RunResult"),
    ("claimscope.schemas", "TriageDecision"),
    ("claimscope.schemas", "TriageResult"),
)


def _serializer() -> JsonPlusSerializer:
    """A serializer that accepts this project's schemas and nothing else new."""
    return JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MODULES)


def checkpoint_path(settings: Settings) -> Path:
    """Where the checkpoint database lives."""
    return Path(settings.runs_dir) / CHECKPOINT_FILENAME


def new_thread_id(paper_id: str) -> str:
    """A fresh thread id, prefixed with the paper so listings are readable."""
    return f"{paper_id}-{uuid.uuid4().hex[:8]}"


def thread_config(thread_id: str) -> RunnableConfig:
    """The config LangGraph uses to address one conversation thread."""
    return {"configurable": {"thread_id": thread_id}}


@contextmanager
def checkpointer(settings: Settings) -> Iterator[SqliteSaver]:
    """Open the checkpoint database, creating its directory if needed."""
    path = checkpoint_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        saver.serde = _serializer()
        yield saver


def list_threads(saver: SqliteSaver) -> list[str]:
    """Every thread id present in the checkpoint database, newest first."""
    seen: dict[str, None] = {}
    for checkpoint in saver.list(None):
        thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
        if thread_id and thread_id not in seen:
            seen[thread_id] = None
    return list(seen)
