"""Cross-encoder reranking: precise relevance scoring of candidate chunks."""

from typing import Protocol

from financial_research_agent.config import get_settings
from financial_research_agent.logging_config import get_logger
from financial_research_agent.retrieval.vector_store import SearchResult

log = get_logger(__name__)


class ScoringModel(Protocol):
    """Anything that scores (query, text) pairs — real model or test fake."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return one relevance score per pair."""
        ...


class Reranker:
    """Rescores retrieval candidates with a cross-encoder."""

    def __init__(self, model: ScoringModel | None = None) -> None:
        self._model = model  # injected in tests; lazy-loaded in production

    def _get_model(self) -> ScoringModel:
        if self._model is None:
            from sentence_transformers import CrossEncoder  # slow import, defer

            model_name = get_settings().rerank_model
            log.info("reranker_loading", model=model_name)
            self._model = CrossEncoder(model_name)
        return self._model

    def rerank(
        self, query: str, candidates: list[SearchResult], *, limit: int = 5
    ) -> list[SearchResult]:
        """Return top `limit` candidates by cross-encoder relevance."""
        if not candidates:
            return []
        scores = self._get_model().predict(
            [(query, c.text) for c in candidates]
        )
        rescored = sorted(
            (
                SearchResult(text=c.text, score=float(s), metadata=c.metadata)
                for c, s in zip(candidates, scores, strict=True)
            ),
            key=lambda r: r.score,
            reverse=True,
        )[:limit]
        log.info("reranked", query=query[:60], candidates=len(candidates), kept=len(rescored))
        return rescored