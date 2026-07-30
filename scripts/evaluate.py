"""Run the evaluation harness against the live Qdrant store (real data)."""

import asyncio

from qdrant_client import AsyncQdrantClient

from financial_research_agent.config import get_settings
from financial_research_agent.eval.dataset import GOLDEN_RETRIEVAL
from financial_research_agent.eval.retrieval import evaluate_retrieval
from financial_research_agent.llm.embeddings import create_embedding_client
from financial_research_agent.logging_config import configure_logging
from financial_research_agent.retrieval.bm25_index import BM25Index
from financial_research_agent.retrieval.hybrid import HybridRetriever
from financial_research_agent.retrieval.reranker import Reranker
from financial_research_agent.retrieval.vector_store import VectorStore


async def main() -> None:
    configure_logging()
    settings = get_settings()
    store = VectorStore(
        AsyncQdrantClient(path=settings.qdrant_path),
        create_embedding_client(),
        dim=settings.embedding_dim,
    )
    docs = await store.scroll_all()
    if not docs:
        print("Store is empty — run a research pass first to index a filing.")
        return

    retriever = HybridRetriever(store, BM25Index(docs), Reranker())
    metrics = await evaluate_retrieval(retriever, GOLDEN_RETRIEVAL, k=5)
    print("\n=== Retrieval Evaluation (hybrid + rerank) ===")
    print(f"cases       : {metrics.cases}")
    print(f"recall@{metrics.k}    : {metrics.recall_at_k:.2%}")
    print(f"MRR         : {metrics.mrr:.3f}")


if __name__ == "__main__":
    asyncio.run(main())