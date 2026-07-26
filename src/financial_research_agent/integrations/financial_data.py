"""SEC XBRL company-facts client: official fundamental financial data."""

from datetime import date

from pydantic import BaseModel

from financial_research_agent.config import get_settings
from financial_research_agent.http_client import create_http_client, get_with_retry
from financial_research_agent.integrations.sec_edgar import SECEdgarClient
from financial_research_agent.logging_config import get_logger

log = get_logger(__name__)

COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# Same economic concept, different tags across companies — first match wins.
CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss",),
    "total_assets": ("Assets",),
    "total_liabilities": ("Liabilities",),
    "shareholders_equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
}


class FinancialFact(BaseModel):
    """One reported financial figure for one fiscal year."""

    metric: str
    concept: str
    value: float
    unit: str
    fiscal_year: int
    end_date: str
    form: str


def _is_full_year(entry: dict) -> bool:
    """True if a duration entry spans roughly one fiscal year.

    Balance-sheet facts are point-in-time (no 'start') and always pass.
    """
    if "start" not in entry:
        return True
    days = (date.fromisoformat(entry["end"]) - date.fromisoformat(entry["start"])).days
    return days > 300


class FinancialDataClient:
    """Fetches multi-year fundamentals from SEC XBRL company facts."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.sec_user_agent:
            msg = "SEC_USER_AGENT must be set (SEC fair-access policy)"
            raise ValueError(msg)
        self._headers = {"User-Agent": settings.sec_user_agent}
        self._edgar = SECEdgarClient()

    async def get_annual_facts(
        self, ticker: str, years: int = 4
    ) -> dict[str, list[FinancialFact]]:
        """Return annual facts per metric, oldest to newest."""
        cik = await self._edgar.get_cik(ticker)
        async with create_http_client(headers=self._headers) as client:
            response = await get_with_retry(client, COMPANY_FACTS_URL.format(cik=cik))
        gaap: dict = response.json().get("facts", {}).get("us-gaap", {})

        results: dict[str, list[FinancialFact]] = {}
        for metric, aliases in CONCEPT_ALIASES.items():
            concept = next((a for a in aliases if a in gaap), None)
            if concept is None:
                log.warning("concept_missing", metric=metric, ticker=ticker.upper())
                continue

            entries: list[dict] = gaap[concept].get("units", {}).get("USD", [])
            by_end: dict[str, dict] = {
                e["end"]: e
                for e in entries
                if e.get("form") == "10-K" and e.get("fp") == "FY" and _is_full_year(e)
            }  # later filings overwrite earlier: restatements win

            facts = [
                FinancialFact(
                    metric=metric,
                    concept=concept,
                    value=float(e["val"]),
                    unit="USD",
                    fiscal_year=int(e["end"][:4]),
                    end_date=e["end"],
                    form=e["form"],
                )
                for e in sorted(by_end.values(), key=lambda e: e["end"])
            ]
            results[metric] = facts[-years:]

        log.info("annual_facts_loaded", ticker=ticker.upper(), metrics=len(results))
        return results