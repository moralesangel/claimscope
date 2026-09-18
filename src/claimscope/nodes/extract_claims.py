"""Extract falsifiable claims from the paper text (PLAN.md section 6)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from claimscope.config import Settings, get_settings
from claimscope.llm import AnthropicStructuredLLM, StructuredLLM
from claimscope.prompts import load_prompt
from claimscope.schemas import Claim, ClaimList
from claimscope.state import GraphState

logger = logging.getLogger(__name__)

# Papers run long and the claims we want sit in the abstract, results and
# ablation sections. Truncate rather than pay for the full text plus appendices.
MAX_PAPER_CHARS = 60_000


def _dedupe_ids(claims: list[Claim]) -> list[Claim]:
    """Make claim ids unique, since the model occasionally repeats a slug."""
    seen: dict[str, int] = {}
    for claim in claims:
        if claim.id in seen:
            seen[claim.id] += 1
            claim.id = f"{claim.id}_{seen[claim.id]}"
        else:
            seen[claim.id] = 1
    return claims


def extract_claims(
    state: GraphState,
    settings: Settings | None = None,
    llm: StructuredLLM | None = None,
) -> GraphState:
    """Graph node: turn the paper text into a validated list of claims."""
    settings = settings or get_settings()
    llm = llm or AnthropicStructuredLLM(settings)

    text = state["paper_text"]
    if len(text) > MAX_PAPER_CHARS:
        logger.info("truncating paper text from %d to %d chars", len(text), MAX_PAPER_CHARS)
        text = text[:MAX_PAPER_CHARS]

    prompt = load_prompt(
        "extract_claims",
        title=state.get("paper_title", ""),
        text=text,
        max_claims=settings.max_claims_extracted,
    )

    result = llm.invoke_structured(prompt, ClaimList)
    claims = _dedupe_ids(result.claims)
    logger.info("extracted %d claims", len(claims))

    _write_claims(claims, Path(settings.runs_dir) / state["paper_id"])
    return {"claims": claims}


def _write_claims(claims: list[Claim], out_dir: Path) -> None:
    """Persist claims.json next to the cached paper, for inspection and eval."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = [claim.model_dump() for claim in claims]
    (out_dir / "claims.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
