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


class DraftGroundedItem(BaseModel):
    """LLM draft insight citing numbered evidence excerpts."""

    title: str = Field(description="Short headline, under 12 words")
    detail: str = Field(description="Two to three sentence explanation")
    severity: Severity = Field(description="Materiality to the investment case")
    source_ids: list[int] = Field(
        description="Numbers of the excerpts that support this item"
    )


class FilingAnalysis(BaseModel):
    """Filing analyst draft output."""

    items: list[DraftGroundedItem] = Field(
        description="Three to five insights, each citing excerpts"
    )


class EvidenceRef(BaseModel):
    """Resolved citation: the actual excerpt and its provenance."""

    excerpt: str
    form_type: str
    filing_date: str


class GroundedInsight(BaseModel):
    """Final insight with verifiable evidence attached."""

    title: str
    detail: str
    severity: Severity
    evidence: list[EvidenceRef]


class InvestmentSummary(BaseModel):
    """Executive synthesis of all analysis streams."""

    headline: str = Field(description="One-sentence overall assessment")
    thesis: str = Field(description="Three to five sentence investment thesis")
    strengths: list[str] = Field(description="Two to four key strengths")
    concerns: list[str] = Field(description="Two to four key concerns")
