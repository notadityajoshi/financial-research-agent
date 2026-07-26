"""Shared state flowing through the research graph."""

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from financial_research_agent.core.metrics import MetricsSummary
from financial_research_agent.integrations.financial_data import FinancialFact
from financial_research_agent.integrations.news import NewsArticle
from financial_research_agent.integrations.sec_edgar import Filing


class NodeError(BaseModel):
    """A failure in one node, recorded instead of crashing the graph."""

    node: str
    message: str


class ResearchState(BaseModel):
    """Everything the graph knows about one research run."""

    ticker: str
    filings: list[Filing] = Field(default_factory=list)
    facts: dict[str, list[FinancialFact]] = Field(default_factory=dict)
    metrics: MetricsSummary | None = None
    news: list[NewsArticle] = Field(default_factory=list)
    # Annotated reducer: parallel branches append concurrently without conflict.
    errors: Annotated[list[NodeError], operator.add] = Field(default_factory=list)
