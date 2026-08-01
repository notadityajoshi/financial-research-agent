"""Offline evaluation tests: fake retriever, synthetic insights."""

from financial_research_agent.agents.schemas import EvidenceRef, GroundedInsight
from financial_research_agent.eval.citations import evaluate_groundedness
from financial_research_agent.eval.dataset import RetrievalCase
from financial_research_agent.eval.retrieval import evaluate_retrieval
from financial_research_agent.retrieval.vector_store import SearchResult


class ScriptedRetriever:
    """Returns canned results per query."""

    def __init__(self, table: dict[str, list[str]]) -> None:
        self._table = table

    async def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        return [
            SearchResult(text=t, score=1.0, metadata={})
            for t in self._table.get(query, [])[:limit]
        ]


CASES = [
    RetrievalCase(query="risks", relevant_markers=["risk"]),
    RetrievalCase(query="growth", relevant_markers=["revenue"]),
]


async def test_perfect_retrieval() -> None:
    retriever = ScriptedRetriever(
        {"risks": ["a risk factor"], "growth": ["revenue rose"]}
    )
    m = await evaluate_retrieval(retriever, CASES, k=5)
    assert m.recall_at_k == 1.0
    assert m.mrr == 1.0


async def test_relevant_at_rank_two_halves_mrr() -> None:
    retriever = ScriptedRetriever(
        {"risks": ["irrelevant", "a risk factor"], "growth": ["revenue rose"]}
    )
    m = await evaluate_retrieval(retriever, CASES, k=5)
    assert m.recall_at_k == 1.0
    assert m.mrr == (0.5 + 1.0) / 2


async def test_miss_lowers_recall() -> None:
    retriever = ScriptedRetriever({"risks": ["nothing here"], "growth": ["revenue"]})
    m = await evaluate_retrieval(retriever, CASES, k=5)
    assert m.recall_at_k == 0.5


def test_groundedness_all_verifiable() -> None:
    corpus = ["The company faces intense competition in AI chips."]
    insights = [
        GroundedInsight(
            title="Competition",
            detail="Rivals intensifying.",
            severity="high",
            evidence=[
                EvidenceRef(
                    excerpt="The company faces intense competition",
                    form_type="10-K",
                    filing_date="2026-01-01",
                )
            ],
        )
    ]
    m = evaluate_groundedness(insights, corpus)
    assert m.groundedness_rate == 1.0
    assert m.evidence_in_corpus == 1


def test_groundedness_catches_hallucinated_citation() -> None:
    corpus = ["Real filing text about revenue."]
    insights = [
        GroundedInsight(
            title="Fabricated",
            detail="Not in the filing.",
            severity="high",
            evidence=[
                EvidenceRef(
                    excerpt="A claim never present in any chunk",
                    form_type="10-K",
                    filing_date="2026-01-01",
                )
            ],
        )
    ]
    m = evaluate_groundedness(insights, corpus)
    assert m.groundedness_rate == 0.0
