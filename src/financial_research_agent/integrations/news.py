"""Company news client backed by Google News RSS (keyless, free)."""

import time
from datetime import UTC, datetime

import feedparser
from pydantic import BaseModel

from financial_research_agent.http_client import create_http_client, get_with_retry
from financial_research_agent.logging_config import get_logger

log = get_logger(__name__)

RSS_URL = "https://news.google.com/rss/search"


class NewsArticle(BaseModel):
    """One news article reference with provenance."""

    title: str
    source: str
    url: str
    published: str  # ISO 8601 UTC, empty if unavailable


def _to_iso(parsed: time.struct_time | None) -> str:
    """Convert feedparser's UTC struct_time to an ISO 8601 string."""
    if parsed is None:
        return ""
    return datetime(*parsed[:6], tzinfo=UTC).isoformat()


class NewsClient:
    """Fetches recent company news via Google News RSS."""

    async def get_company_news(
        self, company: str, *, days: int = 30, limit: int = 10
    ) -> list[NewsArticle]:
        """Return up to `limit` deduplicated articles from the last `days` days."""
        params = {
            "q": f'"{company}" when:{days}d',
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en",
        }
        async with create_http_client() as client:
            response = await get_with_retry(client, RSS_URL, params=params)

        feed = feedparser.parse(response.text)

        articles: list[NewsArticle] = []
        seen_titles: set[str] = set()
        for entry in feed.entries:
            title: str = entry.get("title", "").strip()
            key = title.lower()
            if not title or key in seen_titles:
                continue  # skip empties and syndicated duplicates
            seen_titles.add(key)
            articles.append(
                NewsArticle(
                    title=title,
                    source=entry.get("source", {}).get("title", "Unknown"),
                    url=entry.get("link", ""),
                    published=_to_iso(entry.get("published_parsed")),
                )
            )
            if len(articles) >= limit:
                break

        log.info("news_fetched", company=company, count=len(articles))
        return articles
