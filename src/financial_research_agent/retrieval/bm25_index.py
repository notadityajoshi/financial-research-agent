"""In-memory BM25 lexical index over stored chunks."""

import re

from rank_bm25 import BM25Okapi

from financial_research_agent.logging_config import get_logger
from financial_research_agent.retrieval.vector_store import SearchResult, StoredChunk

log = get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens."""
    return _TOKEN.findall(text.lower())


class BM25Index:
    """Exact-term ranking over a fixed set of chunks."""

    def __init__(self, docs: list[StoredChunk]) -> None:
        self._docs = docs
        self._bm25 = BM25Okapi([_tokenize(d.text) for d in docs] or [[""]])

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        """Return the top chunks by BM25 score (zero-score hits dropped)."""
        if not self._docs:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = [
            SearchResult(
                text=self._docs[i].text,
                score=float(scores[i]),
                metadata=self._docs[i].metadata,
            )
            for i in ranked[:limit]
            if scores[i] > 0
        ]
        log.info("bm25_search", query=query[:60], results=len(results))
        return results