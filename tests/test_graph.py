"""End-to-end test of the phase 1 graph, with arXiv and the LLM stubbed out."""

from __future__ import annotations

import json

import pymupdf
import pytest

from claimscope.config import Settings
from claimscope.graph import build_graph
from claimscope.schemas import PaperSource
from stubs import StubLLM


@pytest.fixture
def fake_paper(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> PaperSource:
    """Replace the network fetch with a locally generated PDF."""
    paper_dir = settings.runs_dir / "2401.00001"
    paper_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = paper_dir / "paper.pdf"

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Our code is available at https://github.com/lab/repo")
    doc.save(pdf_path)
    doc.close()

    source = PaperSource(
        paper_id="2401.00001",
        title="A Test Paper",
        text="Our code is available at https://github.com/lab/repo",
        pdf_path=str(pdf_path),
        repo_url="https://github.com/lab/repo",
    )
    monkeypatch.setattr("claimscope.nodes.ingest.fetch_paper", lambda paper_id, runs_dir: source)
    return source


def test_graph_runs_ingest_then_extraction(
    settings: Settings, stub_llm: StubLLM, fake_paper: PaperSource
) -> None:
    graph = build_graph(settings, stub_llm)

    result = graph.invoke({"paper_id": "2401.00001"})

    assert result["paper_title"] == "A Test Paper"
    assert result["repo_url"] == "https://github.com/lab/repo"
    assert [c.id for c in result["claims"]] == ["residual_beats_plain", "imagenet_top1"]


def test_graph_writes_claims_json(
    settings: Settings, stub_llm: StubLLM, fake_paper: PaperSource
) -> None:
    build_graph(settings, stub_llm).invoke({"paper_id": "2401.00001"})

    written = settings.runs_dir / "2401.00001" / "claims.json"
    assert written.exists()
    assert len(json.loads(written.read_text(encoding="utf-8"))) == 2


def test_extraction_sees_the_ingested_text(
    settings: Settings, stub_llm: StubLLM, fake_paper: PaperSource
) -> None:
    build_graph(settings, stub_llm).invoke({"paper_id": "2401.00001"})

    assert "https://github.com/lab/repo" in stub_llm.prompts[0]
