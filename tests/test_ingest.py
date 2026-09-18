"""Tests for the ingest node. No network: PDF parsing is exercised on a generated file."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from claimscope.nodes.ingest import extract_pdf_text, find_repo_url, normalize_repo_url


class TestFindRepoUrl:
    def test_prefers_an_explicit_code_sentence_over_a_citation(self) -> None:
        text = (
            "We build on prior work https://github.com/someone/unrelated for background.\n"
            "Our code is available at https://github.com/authors/ourproject.\n"
        )
        assert find_repo_url(text) == "https://github.com/authors/ourproject"

    def test_falls_back_to_any_github_link(self) -> None:
        text = "See https://github.com/authors/project for details."
        assert find_repo_url(text) == "https://github.com/authors/project"

    def test_returns_none_when_absent(self) -> None:
        assert find_repo_url("This paper has no code release.") is None

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("https://github.com/a/b.git", "https://github.com/a/b"),
            ("https://github.com/a/b.", "https://github.com/a/b"),
            ("https://github.com/a/b),", "https://github.com/a/b"),
            ("  https://github.com/a/b  ", "https://github.com/a/b"),
        ],
    )
    def test_normalizes_trailing_noise(self, raw: str, expected: str) -> None:
        assert normalize_repo_url(raw) == expected

    def test_finds_url_split_across_a_sentence(self) -> None:
        text = "Implementation details and code: https://github.com/lab/repo (MIT licensed)."
        assert find_repo_url(text) == "https://github.com/lab/repo"

    def test_rejoins_a_url_wrapped_by_pdf_extraction(self) -> None:
        # Verbatim from arXiv 1706.03762; PyMuPDF wraps the URL mid-token.
        text = (
            "The code we used to train and evaluate our models is available at "
            "https://github.com/\ntensorflow/tensor2tensor.\n"
        )
        assert find_repo_url(text) == "https://github.com/tensorflow/tensor2tensor"

    def test_rejoins_a_url_wrapped_more_than_once(self) -> None:
        text = "Code: https://github.com/\nsome-org/\nsome-repo.\n"
        assert find_repo_url(text) == "https://github.com/some-org/some-repo"

    def test_does_not_swallow_the_prose_after_the_url(self) -> None:
        # The sentence ends in "." and the next line starts a new section; an
        # over-eager rejoin produced ".../tensor2tensor.AcknowledgementsWe".
        text = (
            "The code is available at https://github.com/\n"
            "tensorflow/tensor2tensor.\n"
            "Acknowledgements\n"
            "We are grateful to our reviewers.\n"
        )
        assert find_repo_url(text) == "https://github.com/tensorflow/tensor2tensor"

    def test_does_not_join_across_a_blank_line(self) -> None:
        # A paragraph break is not a wrapped URL; the next line is unrelated prose.
        text = "See https://github.com/lab/repo\n\nWe thank our reviewers."
        assert find_repo_url(text) == "https://github.com/lab/repo"


class TestExtractPdfText:
    def test_reads_text_from_every_page(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "sample.pdf"
        doc = pymupdf.open()
        for content in ("First page text", "Second page text"):
            page = doc.new_page()
            page.insert_text((72, 72), content)
        doc.save(pdf_path)
        doc.close()

        text = extract_pdf_text(pdf_path)
        assert "First page text" in text
        assert "Second page text" in text
