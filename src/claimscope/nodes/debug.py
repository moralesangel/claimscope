"""Repair failing experiment scripts (PLAN.md section 6).

Attempts are capped per claim. When the cap is reached the claim is abandoned
with a reason rather than retried forever; analyze reports it as inconclusive.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from claimscope.config import Settings, get_settings
from claimscope.integrity import integrity_warnings, plan_text
from claimscope.llm import ProviderStructuredLLM, StructuredLLM
from claimscope.nodes.codegen import RUN_SCRIPT, SANDBOX_PACKAGES
from claimscope.prompts import load_prompt
from claimscope.schemas import Claim, CodePatch
from claimscope.state import GraphState

logger = logging.getLogger(__name__)

MAX_TRACEBACK_CHARS = 4_000


def propose_patch(
    claim: Claim,
    code: str,
    arm: str,
    seed: int,
    failure_summary: str,
    traceback: str,
    llm: StructuredLLM,
) -> CodePatch:
    """Ask the model to diagnose and fix the failure."""
    prompt = load_prompt(
        "debug",
        claim=json.dumps(claim.model_dump(mode="json"), indent=2, ensure_ascii=False),
        arm=arm,
        seed=seed,
        failure_summary=failure_summary,
        traceback=traceback[-MAX_TRACEBACK_CHARS:],
        code=code,
        packages=", ".join(SANDBOX_PACKAGES),
    )
    return llm.invoke_structured(prompt, CodePatch)


def debug(
    state: GraphState,
    settings: Settings | None = None,
    llm: StructuredLLM | None = None,
) -> GraphState:
    """Graph node: patch the script for each claim whose run failed."""
    settings = settings or get_settings()
    llm = llm or ProviderStructuredLLM(settings)

    claims_by_id = {claim.id: claim for claim in state["claims"]}
    workspaces = state.get("workspace_dirs", {})
    plans = state.get("plans", {})
    failures = dict(state.get("execution_failures", {}))
    attempts = dict(state.get("debug_attempts", {}))
    abandoned = dict(state.get("abandoned_claim_ids", {}))
    errors = list(state.get("errors", []))

    for claim_id, failure in list(failures.items()):
        used = attempts.get(claim_id, 0)
        if used >= settings.max_debug_attempts:
            reason = (
                f"Gave up after {used} debug attempts; last failure was "
                f"{failure.result.failure_summary()}."
            )
            logger.info("abandoning %s: %s", claim_id, reason)
            abandoned[claim_id] = reason
            errors.append(f"{claim_id}: {reason}")
            failures.pop(claim_id)
            continue

        claim = claims_by_id.get(claim_id)
        workspace_dir = workspaces.get(claim_id)
        if claim is None or workspace_dir is None:
            failures.pop(claim_id)
            continue

        script = Path(workspace_dir) / RUN_SCRIPT
        patch = propose_patch(
            claim=claim,
            code=script.read_text(encoding="utf-8"),
            arm=failure.arm,
            seed=failure.seed,
            failure_summary=failure.result.failure_summary(),
            traceback=failure.traceback_text(),
            llm=llm,
        )

        if patch.unfixable:
            # No rewriting reaches a dataset the sandbox cannot download, and
            # substituting one is forbidden. Retrying would spend the remaining
            # attempts on incidental errors and bury the real cause.
            reason = f"The plan cannot be carried out here: {patch.diagnosis}"
            logger.info("abandoning %s: %s", claim_id, reason)
            abandoned[claim_id] = reason
            errors.append(f"{claim_id}: {reason}")
            failures.pop(claim_id)
            continue

        # The patch rewrites the whole script, so it can reintroduce exactly
        # what codegen was told to avoid: a stand-in dataset, or label noise
        # on real data. The failure being fixed is usually "cannot download
        # the dataset", which is the very pressure that produces a substitute.
        plan = plans.get(claim_id)
        if plan is not None:
            for warning in integrity_warnings(patch.code, plan_text(plan), claim_id):
                logger.warning("%s", warning)
                errors.append(warning)

        script.write_text(patch.code + "\n", encoding="utf-8")
        attempts[claim_id] = used + 1
        failures.pop(claim_id)
        logger.info(
            "patched %s (attempt %d/%d): %s",
            claim_id,
            attempts[claim_id],
            settings.max_debug_attempts,
            patch.diagnosis,
        )

    return {
        "execution_failures": failures,
        "debug_attempts": attempts,
        "abandoned_claim_ids": abandoned,
        "errors": errors,
    }


def needs_debugging(state: GraphState) -> bool:
    """Whether any claim is waiting for a fix."""
    return bool(state.get("execution_failures"))
