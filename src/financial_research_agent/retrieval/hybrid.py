"""Hybrid retrieval: dense + BM25 fused with Reciprocal Rank Fusion."""

from financial_research_agent.logging_config import get_logger
from financial_research_agent.retrieval.bm25_index import BM25Index
from financial_research_agent.retrieval.vector_store import SearchResult, VectorStore

log = get_logger(__name__)

RRF_K = 60  # standard damping constant from the RRF paper


class HybridRetriever:
    """Fuses semantic (dense) and lexical (BM25) rankings via RRF."""

    def __init__(self, vector_store: VectorStore, bm25: BM25Index) -> None:
        self._vector_store = vector_store
        self._bm25 = bm25

    @staticmethod
    def _fuse(
        ranked_lists: list[list[SearchResult]], limit: int
    ) -> list[SearchResult]:
        """RRF: score(doc) = sum over lists of 1 / (K + rank)."""
        scores: dict[str, float] = {}
        by_text: dict[str, SearchResult] = {}
        for results in ranked_lists:
            for rank, result in enumerate(results, start=1):
                scores[result.text] = scores.get(result.text, 0.0) + 1.0 / (
                    RRF_K + rank
                )
                by_text.setdefault(result.text, result)
        top = sorted(scores, key=lambda t: scores[t], reverse=True)[:limit]
        return [
            SearchResult(
                text=text, score=scores[text], metadata=by_text[text].metadata
            )
            for text in top
        ]

    async def search(
        self, query: str, *, limit: int = 5, pool: int = 20
    ) -> list[SearchResult]:
        """Hybrid search: fetch `pool` candidates per method, fuse, return top `limit`."""
        dense = await self._vector_store.search(query, limit=pool)
        lexical = self._bm25.search(query, limit=pool)
        fused = self._fuse([dense, lexical], limit)
        log.info(
            "hybrid_search",
            query=query[:60],
            dense=len(dense),
            lexical=len(lexical),
            fused=len(fused),
        )
        return fused