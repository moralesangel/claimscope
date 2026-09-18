"""Graph construction (PLAN.md section 6).

Phase 2 covers:

    ingest -> extract_claims -> triage -> design_plan -> review (interrupt)
        review --rejected--> design_plan
        review --approved--> END

Execution, analysis and reporting are appended in later phases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from langgraph.graph import START, StateGraph

from claimscope.config import Settings, get_settings
from claimscope.llm import StructuredLLM
from claimscope.nodes.design_plan import design_plan
from claimscope.nodes.extract_claims import extract_claims
from claimscope.nodes.ingest import ingest
from claimscope.nodes.review import needs_redesign, review
from claimscope.nodes.triage import triage
from claimscope.state import GraphState

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph


def _route_after_review(state: GraphState) -> Literal["design_plan", "__end__"]:
    """Rejected plans go back to design; otherwise the phase is done."""
    return "design_plan" if needs_redesign(state) else "__end__"


def _route_after_triage(state: GraphState) -> Literal["design_plan", "__end__"]:
    """Nothing testable means there is nothing to plan."""
    return "design_plan" if state.get("selected_claim_ids") else "__end__"


def build_graph(
    settings: Settings | None = None,
    llm: StructuredLLM | None = None,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[GraphState]:
    """Build and compile the graph.

    ``settings`` and ``llm`` are injected so tests can run the whole graph
    against a stub model with no network. A ``checkpointer`` is required for the
    review interrupt to be resumable.
    """
    settings = settings or get_settings()

    builder: StateGraph[GraphState] = StateGraph(GraphState)
    builder.add_node("ingest", lambda state: ingest(state, settings))
    builder.add_node("extract_claims", lambda state: extract_claims(state, settings, llm))
    builder.add_node("triage", lambda state: triage(state, settings, llm))
    builder.add_node("design_plan", lambda state: design_plan(state, settings, llm))
    builder.add_node("review", review)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "extract_claims")
    builder.add_edge("extract_claims", "triage")
    builder.add_conditional_edges("triage", _route_after_triage)
    builder.add_edge("design_plan", "review")
    builder.add_conditional_edges("review", _route_after_review)

    return builder.compile(checkpointer=checkpointer)
