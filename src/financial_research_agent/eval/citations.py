"""Deterministic citation-groundedness check over produced insights."""

from pydantic import BaseModel

from financial_research_agent.agents.schemas import GroundedInsight
from financial_research_agent.logging_config import get_logger

log = get_logger(__name__)


class GroundednessMetrics(BaseModel):
    """How well insights are supported by real evidence."""

    insights: int
    with_evidence: int
    evidence_in_corpus: int
    groundedness_rate: float


def evaluate_groundedness(
    insights: list[GroundedInsight], corpus_texts: list[str]
) -> GroundednessMetrics:
    """Fraction of insights whose evidence excerpts appear in the corpus.

    An insight is grounded only if it has evidence AND every excerpt is a
    real substring of some stored chunk (i.e. the citation is verifiable,
    not hallucinated).
    """
    grounded = 0
    with_evidence = 0
    evidence_in_corpus = 0
    total_refs = 0
    for insight in insights:
        if insight.evidence:
            with_evidence += 1
        insight_ok = bool(insight.evidence)
        for ref in insight.evidence:
            total_refs += 1
            snippet = ref.excerpt.rstrip(".").strip()
            found = any(snippet in text for text in corpus_texts)
            if found:
                evidence_in_corpus += 1
            else:
                insight_ok = False
        if insight_ok:
            grounded += 1
    n = len(insights) or 1
    metrics = GroundednessMetrics(
        insights=len(insights),
        with_evidence=with_evidence,
        evidence_in_corpus=evidence_in_corpus,
        groundedness_rate=grounded / n,
    )
    log.info(
        "groundedness_evaluated",
        insights=metrics.insights,
        groundedness_rate=round(metrics.groundedness_rate, 3),
        refs_verified=f"{evidence_in_corpus}/{total_refs}",
    )
    return metrics