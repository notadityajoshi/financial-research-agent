"""Qdrant vector store: idempotent chunk indexing and dense search."""

import uuid

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from financial_research_agent.ingestion.chunking import Chunk
from financial_research_agent.llm.embeddings import EmbeddingClient
from financial_research_agent.logging_config import get_logger

log = get_logger(__name__)

_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "financial-research-agent")


def _chunk_id(chunk: Chunk) -> str:
    """Deterministic ID from provenance: re-ingestion overwrites, not duplicates."""
    key = "|".join(
        (
            chunk.metadata.get("ticker", ""),
            chunk.metadata.get("form_type", ""),
            chunk.metadata.get("filing_date", ""),
            str(chunk.index),
        )
    )
    return str(uuid.uuid5(_NAMESPACE, key))


class SearchResult(BaseModel):
    """One retrieved chunk with its similarity score."""

    text: str
    score: float
    metadata: dict[str, str]


class StoredChunk(BaseModel):
    """A chunk as persisted in the store: text plus provenance."""

    text: str
    metadata: dict[str, str]


class VectorStore:
    """Async wrapper over one Qdrant collection of document chunks."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        embedder: EmbeddingClient,
        *,
        collection: str = "filings",
        dim: int = 768,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._collection = collection
        self._dim = dim

    async def ensure_collection(self) -> None:
        """Create the collection if it does not exist."""
        if not await self._client.collection_exists(self._collection):
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self._dim, distance=models.Distance.COSINE
                ),
            )
            log.info("collection_created", collection=self._collection)

    async def index_chunks(self, chunks: list[Chunk], *, batch_size: int = 32) -> int:
        """Embed and upsert chunks in batches; returns count indexed."""
        await self.ensure_collection()
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = await self._embedder.embed([c.text for c in batch])
            await self._client.upsert(
                collection_name=self._collection,
                points=[
                    models.PointStruct(
                        id=_chunk_id(chunk),
                        vector=vector,
                        payload={"text": chunk.text, **chunk.metadata},
                    )
                    for chunk, vector in zip(batch, vectors, strict=True)
                ],
            )
        log.info("chunks_indexed", count=len(chunks), collection=self._collection)
        return len(chunks)

    async def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        """Dense semantic search for the query."""
        vector = (await self._embedder.embed([query]))[0]
        response = await self._client.query_points(
            collection_name=self._collection, query=vector, limit=limit
        )
        results = [
            SearchResult(
                text=str(point.payload.get("text", "")),
                score=point.score,
                metadata={
                    k: str(v) for k, v in point.payload.items() if k != "text"
                },
            )
            for point in response.points
            if point.payload
        ]
        log.info("dense_search", query=query[:60], results=len(results))
        return results

    async def scroll_all(self, *, page_size: int = 256) -> list[StoredChunk]:
        """Load every stored chunk, for building derived indexes (e.g. BM25)."""
        docs: list[StoredChunk] = []
        offset = None
        while True:
            points, offset = await self._client.scroll(
                collection_name=self._collection,
                limit=page_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                if point.payload:
                    docs.append(
                        StoredChunk(
                            text=str(point.payload.get("text", "")),
                            metadata={
                                k: str(v)
                                for k, v in point.payload.items()
                                if k != "text"
                            },
                        )
                    )
            if offset is None:
                break
        log.info("chunks_scrolled", count=len(docs))
        return docs