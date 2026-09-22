"""Generate the experiment script for each approved plan (PLAN.md section 6).

Two modes. ``from_scratch`` writes a self-contained experiment from the paper's
description. ``official_repo`` clones the paper's code, surveys it, and asks for
an adapter that drives it at reduced scale -- preferable when it exists, because
the method is then the authors' own rather than our reading of the paper.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from claimscope.config import Settings, get_settings
from claimscope.integrity import integrity_warnings
from claimscope.llm import ProviderStructuredLLM, StructuredLLM
from claimscope.prompts import load_prompt
from claimscope.repo import EntrypointInfo, RepoError, RepoSurvey, survey
from claimscope.schemas import Claim, GeneratedCode, ReductionPlan
from claimscope.state import GraphState

logger = logging.getLogger(__name__)

RUN_SCRIPT = "run.py"
RESULT_FILE = "result.json"
REPO_DIR = "repo"
"""Where the official repository is cloned inside the claim's workspace."""

# Preinstalled in the sandbox image. Kept small: every package is install time
# on every build, and generated code should not reach for heavy dependencies.
SANDBOX_PACKAGES = ("numpy", "scikit-learn")

_FEEDBACK_TEMPLATE = """
## Your previous attempt failed

{feedback}

Fix the cause. Return the complete corrected script, not a diff.
"""


def workspace_for(settings: Settings, paper_id: str, claim_id: str) -> Path:
    """Where one claim's code and outputs live. This is the only mounted directory."""
    return Path(settings.runs_dir) / paper_id / "workspaces" / claim_id


DEFAULT_ARMS = ("treatment", "control")


def arms_for(claim: Claim) -> list[str]:
    """The arm names to run. A comparison needs two; fall back if the claim lacks them."""
    return list(claim.arms) if len(claim.arms) >= 2 else list(DEFAULT_ARMS)


def _per_run_seconds(arm_count: int, seeds: int, settings: Settings) -> float:
    """Rough time budget for a single (arm, seed) run."""
    return (settings.budget_minutes_total * 60) / max(1, arm_count * seeds)


def generate_code(
    claim: Claim,
    plan: ReductionPlan,
    settings: Settings,
    llm: StructuredLLM,
    feedback: str = "",
) -> GeneratedCode:
    """Ask the model for a run.py implementing this plan from scratch."""
    arms = arms_for(claim)

    prompt = load_prompt(
        "codegen",
        claim=json.dumps(claim.model_dump(mode="json"), indent=2, ensure_ascii=False),
        plan=json.dumps(plan.model_dump(mode="json"), indent=2, ensure_ascii=False),
        arms=", ".join(arms),
        packages=", ".join(SANDBOX_PACKAGES),
        total_runs=len(arms) * plan.seeds,
        budget_minutes=settings.budget_minutes_total,
        per_run_seconds=_per_run_seconds(len(arms), plan.seeds, settings),
        feedback_section=_FEEDBACK_TEMPLATE.format(feedback=feedback) if feedback else "",
    )
    return llm.invoke_structured(prompt, GeneratedCode)


def _format_entrypoints(entrypoints: list[EntrypointInfo]) -> str:
    """Render the surveyed entrypoints for the prompt."""
    if not entrypoints:
        return "_None found. You will have to locate the training code yourself._"

    lines = []
    for entry in entrypoints:
        flags = ", ".join(f"`{flag}`" for flag in entry.arguments) or "none declared"
        warning = (
            "  **This script downloads data at runtime, which the sandbox blocks. "
            "You must prevent that.**"
            if entry.downloads_data
            else ""
        )
        lines.append(f"- `{entry.path}` — accepts: {flags}{warning}")
    return "\n".join(lines)


def _bullets(items: list[str], empty: str) -> str:
    return "\n".join(f"- `{item}`" for item in items) if items else f"_{empty}_"


def generate_code_from_repo(
    claim: Claim,
    plan: ReductionPlan,
    repo_survey: RepoSurvey,
    settings: Settings,
    llm: StructuredLLM,
    feedback: str = "",
) -> GeneratedCode:
    """Ask the model for a run.py that drives the paper's own repository."""
    arms = arms_for(claim)

    prompt = load_prompt(
        "codegen_repo",
        claim=json.dumps(claim.model_dump(mode="json"), indent=2, ensure_ascii=False),
        plan=json.dumps(plan.model_dump(mode="json"), indent=2, ensure_ascii=False),
        repo_url=repo_survey.url,
        repo_commit=repo_survey.commit,
        entrypoints=_format_entrypoints(repo_survey.entrypoints),
        config_files=_bullets(repo_survey.config_files, "No config files found."),
        dependencies=_bullets(repo_survey.dependencies, "None declared."),
        imported_packages=_bullets(repo_survey.imported_packages, "None detected."),
        python_files=_bullets(repo_survey.python_files[:30], "None found."),
        readme_excerpt=repo_survey.readme_excerpt or "_No README._",
        arms=", ".join(arms),
        packages=", ".join(sorted({*SANDBOX_PACKAGES, *repo_survey.imported_packages})),
        per_run_seconds=_per_run_seconds(len(arms), plan.seeds, settings),
        feedback_section=_FEEDBACK_TEMPLATE.format(feedback=feedback) if feedback else "",
    )
    return llm.invoke_structured(prompt, GeneratedCode)


def _plan_text(plan: ReductionPlan) -> str:
    """The plan's prose, for checking what the script was supposed to use."""
    return " ".join([plan.original_setup, plan.reduced_setup, *plan.changes, *plan.preserved])


def codegen(
    state: GraphState,
    settings: Settings | None = None,
    llm: StructuredLLM | None = None,
) -> GraphState:
    """Graph node: write run.py into each approved claim's workspace."""
    settings = settings or get_settings()
    llm = llm or ProviderStructuredLLM(settings)

    claims_by_id = {claim.id: claim for claim in state["claims"]}
    plans = state.get("plans", {})
    workspaces = dict(state.get("workspace_dirs", {}))

    errors = list(state.get("errors", []))
    commits = dict(state.get("repo_commits", {}))
    extra_packages = set(state.get("extra_packages", []))

    for claim_id in state.get("approved_plan_ids", []):
        if claim_id in workspaces:
            continue  # already generated
        claim = claims_by_id.get(claim_id)
        plan = plans.get(claim_id)
        if claim is None or plan is None:
            logger.warning("cannot generate code for %s: missing claim or plan", claim_id)
            continue

        workspace = workspace_for(settings, state["paper_id"], claim_id)
        workspace.mkdir(parents=True, exist_ok=True)

        generated = _generate_for(
            claim, plan, state, workspace, settings, llm, errors, commits, extra_packages
        )

        # The report names the plan's dataset, so a script that trains on
        # something else makes the report describe an experiment that never
        # ran. The prompt forbids it; these warnings catch the times it does
        # happen, and reach the reader through the report's errors section.
        for warning in integrity_warnings(generated.code, _plan_text(plan), claim_id):
            logger.warning("%s", warning)
            errors.append(warning)

        # Validation strips surrounding whitespace; restore the trailing newline
        # so the file is a well-formed text file.
        (workspace / RUN_SCRIPT).write_text(generated.code + "\n", encoding="utf-8")
        workspaces[claim_id] = str(workspace)
        logger.info("generated %s for %s: %s", RUN_SCRIPT, claim_id, generated.summary)

    return {
        "workspace_dirs": workspaces,
        "errors": errors,
        "repo_commits": commits,
        "extra_packages": sorted(extra_packages),
    }


def _generate_for(
    claim: Claim,
    plan: ReductionPlan,
    state: GraphState,
    workspace: Path,
    settings: Settings,
    llm: StructuredLLM,
    errors: list[str],
    commits: dict[str, str],
    extra_packages: set[str],
) -> GeneratedCode:
    """Generate the script, using the paper's repository when the plan asks for it.

    A repository that cannot be cloned or surveyed falls back to from_scratch
    rather than failing the claim: a reduced experiment written from the paper's
    description is still worth running.
    """
    repo_url = state.get("repo_url")
    if plan.code_source != "official_repo" or not repo_url:
        return generate_code(claim, plan, settings, llm)

    try:
        repo_survey = survey(repo_url, workspace / REPO_DIR)
    except (RepoError, subprocess.SubprocessError, OSError) as exc:
        message = f"{claim.id}: could not use the official repo ({exc}); writing from scratch"
        logger.warning(message)
        errors.append(message)
        return generate_code(claim, plan, settings, llm)

    commits[claim.id] = repo_survey.commit
    # The sandbox image must carry whatever the repo imports, or every run dies
    # on the first import. Many research repos declare nothing, so the inferred
    # imports are usually the only signal.
    extra_packages.update(repo_survey.imported_packages)
    return generate_code_from_repo(claim, plan, repo_survey, settings, llm)
