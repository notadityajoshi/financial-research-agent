"""Filing indexer: download, clean, chunk and index one SEC filing."""

from typing import Protocol

from financial_research_agent.ingestion.chunking import build_chunks
from financial_research_agent.ingestion.html_cleaner import html_to_text
from financial_research_agent.integrations.sec_edgar import Filing
from financial_research_agent.logging_config import get_logger
from financial_research_agent.retrieval.vector_store import VectorStore

log = get_logger(__name__)


class FilingDownloader(Protocol):
    """Anything that can fetch a filing's primary document."""

    async def download_filing(self, filing: Filing) -> str: ...


class FilingIndexer:
    """Pipeline: SEC download → clean text → chunks → vector store."""

    def __init__(self, sec: FilingDownloader, store: VectorStore) -> None:
        self._sec = sec
        self._store = store

    async def index_filing(
        self, ticker: str, filing: Filing, *, max_chunks: int | None = None
    ) -> int:
        """Index one filing; idempotent (deterministic chunk IDs). Returns count."""
        html = await self._sec.download_filing(filing)
        text = html_to_text(html)
        chunks = build_chunks(
            text,
            {
                "ticker": ticker.upper(),
                "form_type": filing.form_type,
                "filing_date": filing.filing_date,
            },
        )
        if max_chunks is not None:
            chunks = chunks[:max_chunks]
        count = await self._store.index_chunks(chunks)
        log.info(
            "filing_indexed", ticker=ticker.upper(), form=filing.form_type, chunks=count
        )
        return count
