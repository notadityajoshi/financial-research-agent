"""Hybrid retrieval: dense + BM25 fused with Reciprocal Rank Fusion."""

from financial_research_agent.logging_config import get_logger
from financial_research_agent.retrieval.bm25_index import BM25Index
from financial_research_agent.retrieval.vector_store import SearchResult, VectorStore
from financial_research_agent.retrieval.reranker import Reranker
log = get_logger(__name__)

RRF_K = 60  # standard damping constant from the RRF paper


class HybridRetriever:
    """Fuses semantic (dense) and lexical (BM25) rankings via RRF."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25: BM25Index,
        reranker: Reranker | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._bm25 = bm25
        self._reranker = reranker

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
        """Hybrid search; if a reranker is set, RRF selects candidates and
        the cross-encoder picks the final top `limit`."""
        dense = await self._vector_store.search(query, limit=pool)
        lexical = self._bm25.search(query, limit=pool)
        fuse_limit = pool if self._reranker else limit
        fused = self._fuse([dense, lexical], fuse_limit)
        if self._reranker:
            fused = self._reranker.rerank(query, fused, limit=limit)
        log.info(
            "hybrid_search",
            query=query[:60],
            dense=len(dense),
            lexical=len(lexical),
            fused=len(fused),
            reranked=self._reranker is not None,
        )
        return fused