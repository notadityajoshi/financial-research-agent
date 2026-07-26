"""RAG nodes: index the latest 10-K and produce citation-grounded insights."""

from financial_research_agent.agents.nodes import Node, NodeError, fault_isolated
from financial_research_agent.agents.schemas import (
    EvidenceRef,
    FilingAnalysis,
    GroundedInsight,
)
from financial_research_agent.agents.state import ResearchState
from financial_research_agent.config import get_settings
from financial_research_agent.llm.base import ChatMessage, LLMClient, Role
from financial_research_agent.llm.structured import generate_structured
from financial_research_agent.logging_config import get_logger
from financial_research_agent.retrieval.bm25_index import BM25Index
from financial_research_agent.retrieval.hybrid import HybridRetriever
from financial_research_agent.retrieval.indexer import FilingIndexer
from financial_research_agent.retrieval.reranker import Reranker
from financial_research_agent.retrieval.vector_store import (
    SearchResult,
    VectorStore,
)
from financial_research_agent.integrations.sec_edgar import SECEdgarClient

log = get_logger(__name__)

_QUERIES = (
    "principal risk factors",
    "competition and market position",
    "growth strategy and new products",
)

_MAX_EVIDENCE = 8


def make_index_filing(sec: SECEdgarClient, store: VectorStore) -> Node:
    """Node factory: index the most recent 10-K from state.filings."""

    async def index_filing(state: ResearchState) -> dict:
        filing = next((f for f in state.filings if f.form_type == "10-K"), None)
        if filing is None:
            return {
                "errors": [
                    NodeError(node="index_filing", message="no 10-K in filings")
                ]
            }
        await FilingIndexer(sec, store).index_filing(
            state.ticker, filing, max_chunks=get_settings().max_index_chunks
        )
        return {}

    return fault_isolated("index_filing", index_filing)


def _to_ref(result: SearchResult) -> EvidenceRef:
    return EvidenceRef(
        excerpt=result.text[:300],
        form_type=result.metadata.get("form_type", ""),
        filing_date=result.metadata.get("filing_date", ""),
    )


def make_filing_analyst(
    store: VectorStore, llm: LLMClient, reranker: Reranker | None = None
) -> Node:
    """Node factory: retrieval-grounded insights with resolved citations."""

    async def analyze_filing(state: ResearchState) -> dict:
        retriever = HybridRetriever(
            store, BM25Index(await store.scroll_all()), reranker
        )
        seen: dict[str, SearchResult] = {}
        for query in _QUERIES:
            for result in await retriever.search(query, limit=4):
                seen.setdefault(result.text, result)
        evidence = list(seen.values())[:_MAX_EVIDENCE]
        if not evidence:
            return {
                "errors": [
                    NodeError(node="analyze_filing", message="no evidence retrieved")
                ]
            }

        numbered = "\n\n".join(
            f"[{i}] {r.text[:600]}" for i, r in enumerate(evidence, 1)
        )
        messages = [
            ChatMessage(
                role=Role.SYSTEM,
                content=(
                    "You are an equity analyst. Using ONLY the numbered filing "
                    "excerpts, produce insights about the company's risks, "
                    "competitive position and growth strategy. Every item MUST "
                    "list the excerpt numbers that support it in source_ids. "
                    "Make no claim that the excerpts do not support."
                ),
            ),
            ChatMessage(
                role=Role.USER,
                content=f"Company: {state.ticker}\n\nExcerpts:\n\n{numbered}",
            ),
        ]
        draft = await generate_structured(llm, messages, FilingAnalysis)

        insights: list[GroundedInsight] = []
        for item in draft.items:
            refs = [
                _to_ref(evidence[i - 1])
                for i in item.source_ids
                if 1 <= i <= len(evidence)
            ]
            if refs:  # uncited claims are discarded, not trusted
                insights.append(
                    GroundedInsight(
                        title=item.title,
                        detail=item.detail,
                        severity=item.severity,
                        evidence=refs,
                    )
                )
        log.info(
            "filing_insights",
            drafted=len(draft.items),
            grounded=len(insights),
        )
        return {"filing_insights": insights}

    return fault_isolated("analyze_filing", analyze_filing)