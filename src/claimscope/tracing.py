"""Optional LLM tracing (PLAN.md section 3).

Tracing is a debugging aid, never a dependency: ClaimScope must run identically
with it switched off, and a misconfigured or unreachable tracing backend must
degrade to no tracing rather than failing the run. Everything here is written
around that.

Two backends are supported. LangSmith ships with langchain and needs only an
API key. Langfuse is an optional extra (``uv sync --extra tracing``) and can be
self-hosted.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from claimscope.config import Settings

logger = logging.getLogger(__name__)

# Environment variables each backend needs. Checked before enabling tracing so a
# half-configured backend is reported clearly rather than failing mid-run.
_REQUIRED_ENV = {
    "langsmith": ("LANGSMITH_API_KEY",),
    "langfuse": ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"),
}


def _missing_env(backend: str) -> list[str]:
    return [name for name in _REQUIRED_ENV.get(backend, ()) if not os.environ.get(name)]


def _enable_langsmith(settings: Settings) -> bool:
    """Turn on LangSmith by setting the variables langchain reads itself."""
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", settings.tracing_project)
    logger.info("tracing to LangSmith, project %s", os.environ["LANGSMITH_PROJECT"])
    return True


def _langfuse_handler(settings: Settings) -> Any | None:
    """Build the Langfuse callback handler, or None if it is unusable."""
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        logger.warning(
            "tracing_backend is langfuse but the package is not installed; "
            "install it with: uv sync --extra tracing"
        )
        return None

    try:
        handler = CallbackHandler()
    except Exception:
        # A bad host or credentials must not take the run down with it.
        logger.warning("could not start Langfuse tracing; continuing untraced", exc_info=True)
        return None

    logger.info("tracing to Langfuse, project %s", settings.tracing_project)
    return handler


def setup_tracing(settings: Settings) -> list[Any]:
    """Configure tracing and return callbacks to pass to the graph.

    Returns an empty list when tracing is off or unavailable, which is the
    normal case and not an error.
    """
    if not settings.tracing_enabled:
        return []

    backend = settings.tracing_backend
    if missing := _missing_env(backend):
        logger.warning(
            "tracing is enabled for %s but %s is not set; continuing untraced",
            backend,
            ", ".join(missing),
        )
        return []

    if backend == "langsmith":
        _enable_langsmith(settings)
        return []  # langchain picks LangSmith up from the environment

    handler = _langfuse_handler(settings)
    return [handler] if handler is not None else []


@contextmanager
def traced_run(settings: Settings, config: RunnableConfig) -> Iterator[RunnableConfig]:
    """Yield the run config, with tracing callbacks attached if any are active.

    Used as::

        with traced_run(settings, thread_config(tid)) as config:
            graph.invoke(state, config)

    Yields the config unchanged when tracing is off, so callers need no branch.
    """
    callbacks = setup_tracing(settings)
    try:
        yield {**config, "callbacks": callbacks} if callbacks else config
    finally:
        _flush(callbacks)


def _flush(callbacks: list[Any]) -> None:
    """Push buffered spans before the process exits.

    Tracing clients batch in the background, so a short CLI run can finish and
    exit before anything is sent.
    """
    for callback in callbacks:
        client = getattr(callback, "client", None)
        flush = getattr(client, "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception:
                logger.debug("could not flush traces", exc_info=True)
