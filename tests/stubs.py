"""Test doubles shared across test modules."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class StubLLM:
    """A StructuredLLM that replays canned responses and records its prompts."""

    def __init__(self, responses: list[BaseModel]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def invoke_structured(self, prompt: str, schema: type[T]) -> T:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("StubLLM ran out of canned responses")
        response = self.responses.pop(0)
        if not isinstance(response, schema):
            raise AssertionError(f"StubLLM was asked for {schema.__name__}, has {type(response)}")
        return response
