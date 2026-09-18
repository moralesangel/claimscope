"""Prompt loading. Prompts live as .md files here, never inline in code."""

from __future__ import annotations

from pathlib import Path

_PROMPT_DIR = Path(__file__).parent


def load_prompt(name: str, **variables: object) -> str:
    """Load ``<name>.md`` and substitute ``{placeholder}`` variables.

    Raises KeyError if the template needs a variable that was not supplied, so a
    typo surfaces here rather than as a confusing prompt sent to the model.
    """
    template = (_PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")
    return template.format(**variables)
