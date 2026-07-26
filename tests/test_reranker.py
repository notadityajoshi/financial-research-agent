"""Offline reranker tests with an injected fake scoring model."""

from financial_research_agent.retrieval.reranker import Reranker
from financial_research_agent.retrieval.vector_store import SearchResult


class FakeScorer:
    """Scores by word overlap between query and text."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [
            float(len(set(q.lower().split()) & set(t.lower().split())))
            for q, t in pairs
        ]


def _result(text: str, score: float = 0.0) -> SearchResult:
    return SearchResult(text=text, score=score, metadata={"ticker": "TEST"})


def test_reranker_reorders_by_relevance() -> None:
    candidates = [
        _result("totally unrelated boilerplate", score=0.9),
        _result("supply chain risk factors ahead", score=0.1),
    ]
    top = Reranker(model=FakeScorer()).rerank("risk factors", candidates, limit=2)
    assert top[0].text.startswith("supply chain")
    assert top[0].score == 2.0  # cross-encoder score replaces RRF score


def test_limit_applied() -> None:
    candidates = [_result(f"risk item {i}") for i in range(10)]
    assert len(Reranker(model=FakeScorer()).rerank("risk", candidates, limit=3)) == 3


def test_empty_candidates() -> None:
    assert Reranker(model=FakeScorer()).rerank("anything", []) == []