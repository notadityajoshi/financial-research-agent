"""Deterministic retrieval metrics: recall@k and MRR over a golden dataset."""

from typing import Protocol

from pydantic import BaseModel

from financial_research_agent.eval.dataset import RetrievalCase
from financial_research_agent.logging_config import get_logger
from financial_research_agent.retrieval.vector_store import SearchResult

log = get_logger(__name__)


class Retriever(Protocol):
    """Anything that returns ranked SearchResults for a query."""

    async def search(self, query: str, *, limit: int = ...) -> list[SearchResult]: ...


class RetrievalMetrics(BaseModel):
    """Aggregate retrieval quality over a dataset."""

    cases: int
    recall_at_k: float
    mrr: float
    k: int


def _first_relevant_rank(results: list[SearchResult], markers: list[str]) -> int | None:
    """1-based rank of the first relevant result, or None if none relevant."""
    lowered = [m.lower() for m in markers]
    for rank, result in enumerate(results, start=1):
        text = result.text.lower()
        if any(marker in text for marker in lowered):
            return rank
    return None


async def evaluate_retrieval(
    retriever: Retriever, cases: list[RetrievalCase], *, k: int = 5
) -> RetrievalMetrics:
    """Compute recall@k and MRR for the retriever over the cases."""
    hits = 0
    reciprocal_sum = 0.0
    for case in cases:
        results = await retriever.search(case.query, limit=k)
        rank = _first_relevant_rank(results, case.relevant_markers)
        if rank is not None:
            hits += 1
            reciprocal_sum += 1.0 / rank
    n = len(cases) or 1
    metrics = RetrievalMetrics(
        cases=len(cases),
        recall_at_k=hits / n,
        mrr=reciprocal_sum / n,
        k=k,
    )
    log.info(
        "retrieval_evaluated",
        cases=metrics.cases,
        recall_at_k=round(metrics.recall_at_k, 3),
        mrr=round(metrics.mrr, 3),
    )
    return metrics
