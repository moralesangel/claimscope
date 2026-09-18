"""Graph construction (PLAN.md section 6).

Phase 1 is the linear head of the pipeline: ingest -> extract_claims -> END.
Later phases insert triage, planning, the review interrupt, execution and
reporting between extraction and the end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from claimscope.config import Settings, get_settings
from claimscope.llm import StructuredLLM
from claimscope.nodes.extract_claims import extract_claims
from claimscope.nodes.ingest import ingest
from claimscope.state import GraphState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_graph(
    settings: Settings | None = None,
    llm: StructuredLLM | None = None,
) -> CompiledStateGraph[GraphState]:
    """Build and compile the phase 1 graph.

    ``settings`` and ``llm`` are injected so tests can run the whole graph
    against a stub model with no network access.
    """
    settings = settings or get_settings()

    builder: StateGraph[GraphState] = StateGraph(GraphState)
    builder.add_node("ingest", lambda state: ingest(state, settings))
    builder.add_node("extract_claims", lambda state: extract_claims(state, settings, llm))

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "extract_claims")
    builder.add_edge("extract_claims", END)

    return builder.compile()
