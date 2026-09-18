"""Match extracted claims to annotated ones (PLAN.md section 9).

The plan suggests an LLM judge. That is the right tool for borderline pairs, but
it is non-deterministic and costs a call per comparison, so the default is
lexical: a claim is the same claim if it talks about the same arms and metric in
the same direction. The LLM judge is available for the pairs lexical matching
leaves unresolved, behind an explicit flag.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from claimscope.schemas import Claim
from eval.schemas import AnnotatedClaim

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.35

# Words that carry no signal about which claim is which.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "we",
        "our",
        "they",
        "their",
        "than",
        "then",
    ]
)


@dataclass(frozen=True)
class Match:
    """One extracted claim paired with the annotation it corresponds to."""

    predicted: Claim
    annotated: AnnotatedClaim
    score: float


@dataclass
class MatchResult:
    matches: list[Match]
    unmatched_predicted: list[Claim]
    unmatched_annotated: list[AnnotatedClaim]


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if word not in _STOPWORDS and len(word) > 1}


def similarity(predicted: Claim, annotated: AnnotatedClaim) -> float:
    """How likely two claims are the same claim, from 0 to 1.

    Jaccard over content words, with the metric and arm names weighted up: two
    claims about the same paper often share wording, so the distinguishing
    signal is what is being compared and on which measure.
    """
    predicted_text = " ".join([predicted.text, predicted.metric, *predicted.arms])
    left, right = _tokens(predicted_text), _tokens(annotated.text)
    if not left or not right:
        return 0.0

    overlap = left & right
    union = left | right
    base = len(overlap) / len(union)

    # Sharing the metric or an arm name is strong evidence.
    bonus = 0.0
    if _tokens(predicted.metric) & right:
        bonus += 0.15
    if any(_tokens(arm) & right for arm in predicted.arms):
        bonus += 0.15

    return min(1.0, base + bonus)


def match_claims(
    predicted: list[Claim],
    annotated: list[AnnotatedClaim],
    threshold: float = DEFAULT_THRESHOLD,
) -> MatchResult:
    """Pair extracted claims with annotations, greedily and one-to-one.

    Greedy on the best score first: a claim can only match one annotation, so
    taking the strongest pairs first avoids a weak pair stealing an annotation
    that a stronger one needed.
    """
    scored = sorted(
        (
            (similarity(claim, annotation), index, annotation_index)
            for index, claim in enumerate(predicted)
            for annotation_index, annotation in enumerate(annotated)
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )

    used_predicted: set[int] = set()
    used_annotated: set[int] = set()
    matches: list[Match] = []

    for score, index, annotation_index in scored:
        if score < threshold:
            break
        if index in used_predicted or annotation_index in used_annotated:
            continue
        used_predicted.add(index)
        used_annotated.add(annotation_index)
        matches.append(Match(predicted[index], annotated[annotation_index], score))

    return MatchResult(
        matches=matches,
        unmatched_predicted=[c for i, c in enumerate(predicted) if i not in used_predicted],
        unmatched_annotated=[a for i, a in enumerate(annotated) if i not in used_annotated],
    )
