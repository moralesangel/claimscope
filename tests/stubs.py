"""Test doubles shared across test modules."""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel

from claimscope.sandbox.runner import ExecutionRequest, ExecutionResult

T = TypeVar("T", bound=BaseModel)


class StubLLM:
    """A StructuredLLM that replays canned responses and records its prompts."""

    def __init__(self, responses: list[BaseModel]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def invoke_structured(self, prompt: str, schema: type[T]) -> T:
        self.prompts.append(prompt)
        if not self.responses:
            # Running dry almost always means the graph took an unplanned turn
            # into the debug loop, and the interesting part is the failure that
            # sent it there -- which the prompt carries and a bare message
            # throws away. This cost a CI round-trip once; it should not again.
            raise AssertionError(
                f"StubLLM ran out of canned responses; it was asked for "
                f"{schema.__name__}. Last prompt:\n{prompt[-2000:]}"
            )
        response = self.responses.pop(0)
        if not isinstance(response, schema):
            raise AssertionError(f"StubLLM was asked for {schema.__name__}, has {type(response)}")
        return response


class FakeRunner:
    """A Runner that executes run.py in-process instead of in Docker.

    Docker is not available on every development machine, and the graph should be
    testable without it. This keeps the observable contract -- arguments in,
    result.json out, exit codes and durations -- so the nodes are exercised for
    real; only the isolation is simulated.
    """

    def __init__(
        self,
        *,
        seconds_per_call: float = 0.5,
        fail_times: int = 0,
        fail_message: str = "Traceback: ValueError: boom",
        metric_by_arm: dict[str, float] | None = None,
    ) -> None:
        self.seconds_per_call = seconds_per_call
        self.remaining_failures = fail_times
        self.fail_message = fail_message
        self.metric_by_arm = metric_by_arm or {"treatment": 0.8, "control": 0.5}
        self.calls: list[list[str]] = []
        self.prepared: list[object] = []

    def available(self) -> bool:
        return True

    def prepare(self, spec: object) -> None:
        self.prepared.append(spec)

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.calls.append(list(request.args))

        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            return ExecutionResult(
                exit_code=1,
                stdout="",
                stderr=self.fail_message,
                duration_s=self.seconds_per_call,
            )

        arm = _arg_value(request.args, "--arm") or "treatment"
        seed = int(_arg_value(request.args, "--seed") or 0)
        metric = self.metric_by_arm.get(arm, 0.0) + seed * 0.001

        (request.workspace / "result.json").write_text(
            json.dumps({"arm": arm, "seed": seed, "metric": metric, "metric_name": "accuracy"}),
            encoding="utf-8",
        )
        return ExecutionResult(
            exit_code=0,
            stdout=f"ran {arm} seed {seed}\n",
            stderr="",
            duration_s=self.seconds_per_call,
        )


def _arg_value(args: list[str], flag: str) -> str | None:
    """Read the value following ``flag`` in an argument list."""
    if flag in args:
        index = args.index(flag)
        if index + 1 < len(args):
            return args[index + 1]
    return None
