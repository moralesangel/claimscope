"""Graph construction (PLAN.md section 6).

The full pipeline (PLAN.md section 6):

    ingest -> extract_claims -> triage -> design_plan -> review (interrupt)
        review --rejected--> design_plan
        review --approved--> codegen -> execute
        execute --failed--> debug -> execute   (capped per claim)
        execute --ok--> analyze -> report -> END

Every path reaches analyze, so a claim that was never testable, was abandoned,
or ran out of budget still appears in the report with its reason.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from langgraph.graph import END, START, StateGraph

from claimscope.config import Settings, get_settings
from claimscope.llm import StructuredLLM
from claimscope.nodes.analyze import analyze
from claimscope.nodes.codegen import codegen
from claimscope.nodes.debug import debug, needs_debugging
from claimscope.nodes.design_plan import design_plan
from claimscope.nodes.execute import execute
from claimscope.nodes.extract_claims import extract_claims
from claimscope.nodes.ingest import ingest
from claimscope.nodes.report import report
from claimscope.nodes.review import needs_redesign, review
from claimscope.nodes.triage import triage
from claimscope.session import use_project_serializer
from claimscope.state import GraphState

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from claimscope.sandbox.runner import Runner


def _route_after_review(state: GraphState) -> Literal["design_plan", "codegen"]:
    """Rejected plans go back to design; approved ones move to code generation."""
    return "design_plan" if needs_redesign(state) else "codegen"


def _route_after_execute(state: GraphState) -> Literal["debug", "analyze"]:
    """Failures go to the debug loop; otherwise move on to analysis."""
    return "debug" if needs_debugging(state) else "analyze"


def _route_after_debug(state: GraphState) -> Literal["execute", "analyze"]:
    """Retry patched scripts; analyse what we have once nothing is retryable.

    A claim abandoned after exhausting its debug attempts still reaches analyze,
    which records it as inconclusive with the reason.
    """
    abandoned = set(state.get("abandoned_claim_ids", {}))
    completed = set(state.get("run_results", {}))
    retryable = [
        cid
        for cid in state.get("approved_plan_ids", [])
        if cid not in abandoned and cid not in completed
    ]
    return "execute" if retryable else "analyze"


def _route_after_triage(state: GraphState) -> Literal["design_plan", "analyze"]:
    """Nothing testable means nothing to plan, but the paper still gets a report."""
    return "design_plan" if state.get("selected_claim_ids") else "analyze"


def build_graph(
    settings: Settings | None = None,
    llm: StructuredLLM | None = None,
    checkpointer: BaseCheckpointSaver[str] | None = None,
    runner: Runner | None = None,
) -> CompiledStateGraph[GraphState]:
    """Build and compile the graph.

    ``settings``, ``llm`` and ``runner`` are injected so tests can run the whole
    graph with no network and no Docker. A ``checkpointer`` is required for the
    review interrupt to be resumable.
    """
    settings = settings or get_settings()

    builder: StateGraph[GraphState] = StateGraph(GraphState)
    builder.add_node("ingest", lambda state: ingest(state, settings))
    builder.add_node("extract_claims", lambda state: extract_claims(state, settings, llm))
    builder.add_node("triage", lambda state: triage(state, settings, llm))
    builder.add_node("design_plan", lambda state: design_plan(state, settings, llm))
    builder.add_node("review", review)
    builder.add_node("codegen", lambda state: codegen(state, settings, llm))
    builder.add_node("execute", lambda state: execute(state, settings, runner))
    builder.add_node("debug", lambda state: debug(state, settings, llm))
    builder.add_node("analyze", lambda state: analyze(state, settings))
    builder.add_node("report", lambda state: report(state, settings))

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "extract_claims")
    builder.add_edge("extract_claims", "triage")
    builder.add_conditional_edges("triage", _route_after_triage)
    builder.add_edge("design_plan", "review")
    builder.add_conditional_edges("review", _route_after_review)
    builder.add_edge("codegen", "execute")
    builder.add_conditional_edges("execute", _route_after_execute)
    builder.add_conditional_edges("debug", _route_after_debug)
    builder.add_edge("analyze", "report")
    builder.add_edge("report", END)

    if checkpointer is not None:
        # Any checkpointer, not just ours, must know our types: LangGraph will
        # refuse unregistered ones in a future version.
        use_project_serializer(checkpointer)

    return builder.compile(checkpointer=checkpointer)
