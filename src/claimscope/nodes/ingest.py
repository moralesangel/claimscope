"""Ingest node: fetch a paper from arXiv and extract its text (PLAN.md section 6)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import arxiv
import pymupdf
import requests

from claimscope.config import Settings, get_settings
from claimscope.schemas import PaperSource
from claimscope.state import GraphState

logger = logging.getLogger(__name__)

# Ordered by preference: an explicit "code is at" sentence beats a bare link in
# the references, which is often a citation to someone else's repository.
_REPO_PATTERNS = (
    re.compile(
        r"(?:code|implementation|source)[^.]{0,80}?"
        r"(https?://(?:www\.)?github\.com/[\w.-]+/[\w.-]+)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"(https?://(?:www\.)?github\.com/[\w.-]+/[\w.-]+)", re.IGNORECASE),
)

_TRAILING_PUNCTUATION = ".,;:)]}>\"'"

# PDF extraction wraps long URLs mid-token, e.g. "https://github.com/\ntensorflow/
# tensor2tensor". Rejoin those before matching, or the repo is silently missed.
# Only a break right after "/" or "-" is a wrap: anywhere else (especially after
# the sentence's final ".") the next line is ordinary prose, not more URL.
_URL_LINE_BREAK = re.compile(r"(https?://\S*[/-])[ \t]*\n[ \t]*(?=\w)")


def _join_wrapped_urls(text: str) -> str:
    """Undo line wrapping inside URLs introduced by PDF text extraction."""
    previous = None
    while previous != text:
        previous = text
        text = _URL_LINE_BREAK.sub(r"\1", text)
    return text


def normalize_repo_url(raw: str) -> str:
    """Strip trailing punctuation and a .git suffix from a scraped URL."""
    url = raw.strip().rstrip(_TRAILING_PUNCTUATION)
    if url.endswith(".git"):
        url = url[: -len(".git")]
    return url


def find_repo_url(text: str) -> str | None:
    """Return the most likely official code repository mentioned in the paper."""
    joined = _join_wrapped_urls(text)
    for pattern in _REPO_PATTERNS:
        match = pattern.search(joined)
        if match:
            return normalize_repo_url(match.group(1))
    return None


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF, one blank line between pages."""
    # pymupdf ships no type information for Document.
    with pymupdf.open(pdf_path) as doc:  # type: ignore[no-untyped-call]
        return "\n\n".join(str(page.get_text()) for page in doc)


def fetch_paper(paper_id: str, runs_dir: Path) -> PaperSource:
    """Download and parse an arXiv paper, caching everything under runs/<paper_id>/.

    Re-running with the same id reuses the cached PDF and text rather than
    hitting arXiv again.
    """
    cache_dir = runs_dir / paper_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = cache_dir / "paper.pdf"
    text_path = cache_dir / "paper.txt"

    client = arxiv.Client()
    result = next(client.results(arxiv.Search(id_list=[paper_id])))

    if not pdf_path.exists():
        if not result.pdf_url:
            raise ValueError(f"arXiv entry {paper_id} has no PDF link")
        logger.info("downloading %s", result.pdf_url)
        response = requests.get(result.pdf_url, timeout=60)
        response.raise_for_status()
        pdf_path.write_bytes(response.content)

    if text_path.exists():
        text = text_path.read_text(encoding="utf-8")
    else:
        text = extract_pdf_text(pdf_path)
        text_path.write_text(text, encoding="utf-8")

    return PaperSource(
        paper_id=paper_id,
        title=result.title.strip(),
        text=text,
        pdf_path=str(pdf_path),
        repo_url=find_repo_url(text),
    )


def ingest(state: GraphState, settings: Settings | None = None) -> GraphState:
    """Graph node: populate the state with the paper's text and metadata."""
    settings = settings or get_settings()
    source = fetch_paper(state["paper_id"], Path(settings.runs_dir))
    logger.info(
        "ingested %s (%d chars), repo=%s", source.paper_id, len(source.text), source.repo_url
    )
    return {
        "paper_title": source.title,
        "paper_text": source.text,
        "pdf_path": source.pdf_path,
        "repo_url": source.repo_url,
    }
