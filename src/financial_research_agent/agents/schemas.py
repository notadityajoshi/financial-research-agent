"""Schemas for LLM analyst outputs."""

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["low", "medium", "high"]


class AnalysisItem(BaseModel):
    """One identified risk or opportunity."""

    title: str = Field(description="Short headline, under 12 words")
    detail: str = Field(description="Two to three sentence explanation")
    severity: Severity = Field(description="Materiality to the investment case")


class RiskAnalysis(BaseModel):
    """Risk analyst output."""

    items: list[AnalysisItem] = Field(description="Three to five distinct risks")


class OpportunityAnalysis(BaseModel):
    """Opportunity analyst output."""

    items: list[AnalysisItem] = Field(
        description="Three to five distinct opportunities"
    )