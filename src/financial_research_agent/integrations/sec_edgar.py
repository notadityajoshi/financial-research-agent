"""SEC EDGAR client: ticker-to-CIK lookup and filing retrieval."""

from pydantic import BaseModel

from financial_research_agent.config import get_settings
from financial_research_agent.http_client import create_http_client, get_with_retry
from financial_research_agent.logging_config import get_logger

log = get_logger(__name__)

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"


class Filing(BaseModel):
    """Metadata for one SEC filing."""

    cik: int
    company_name: str
    form_type: str
    filing_date: str
    accession_number: str
    primary_document: str

    @property
    def document_url(self) -> str:
        """Direct URL to the filing's primary document."""
        return ARCHIVES_URL.format(
            cik=self.cik,
            accession=self.accession_number.replace("-", ""),
            document=self.primary_document,
        )


class SECEdgarClient:
    """Async client for the free SEC EDGAR API."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.sec_user_agent:
            msg = "SEC_USER_AGENT must be set (SEC fair-access policy)"
            raise ValueError(msg)
        self._headers = {"User-Agent": settings.sec_user_agent}

    async def get_cik(self, ticker: str) -> int:
        """Resolve a stock ticker to its SEC CIK number."""
        async with create_http_client(headers=self._headers) as client:
            response = await get_with_retry(client, TICKER_MAP_URL)
        data: dict = response.json()
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry["ticker"] == ticker_upper:
                cik = int(entry["cik_str"])
                log.info("cik_resolved", ticker=ticker_upper, cik=cik)
                return cik
        msg = f"Ticker not found on SEC EDGAR: {ticker}"
        raise ValueError(msg)

    async def get_recent_filings(
        self,
        ticker: str,
        form_types: tuple[str, ...] = ("10-K", "10-Q"),
        limit: int = 5,
    ) -> list[Filing]:
        """Return the most recent filings matching the given form types."""
        cik = await self.get_cik(ticker)
        async with create_http_client(headers=self._headers) as client:
            response = await get_with_retry(client, SUBMISSIONS_URL.format(cik=cik))
        data = response.json()
        company_name: str = data["name"]
        recent = data["filings"]["recent"]

        filings: list[Filing] = []
        for form, date, accession, document in zip(
            recent["form"],
            recent["filingDate"],
            recent["accessionNumber"],
            recent["primaryDocument"],
            strict=True,
        ):
            if form in form_types:
                filings.append(
                    Filing(
                        cik=cik,
                        company_name=company_name,
                        form_type=form,
                        filing_date=date,
                        accession_number=accession,
                        primary_document=document,
                    )
                )
            if len(filings) >= limit:
                break

        log.info("filings_listed", ticker=ticker.upper(), count=len(filings))
        return filings

    async def download_filing(self, filing: Filing) -> str:
        """Download and return the filing's primary document (HTML)."""
        async with create_http_client(headers=self._headers) as client:
            response = await get_with_retry(client, filing.document_url)
        log.info(
            "filing_downloaded",
            form=filing.form_type,
            date=filing.filing_date,
            bytes=len(response.content),
        )
        return response.text
