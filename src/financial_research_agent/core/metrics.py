"""Deterministic financial metric calculations.

Pure functions only: no I/O, no LLM calls, no side effects.
Every number in a report traces back to arithmetic in this module.
"""

from pydantic import BaseModel

from financial_research_agent.integrations.financial_data import FinancialFact


class AnnualMetrics(BaseModel):
    """Computed ratios for one fiscal year. None = inputs unavailable."""

    fiscal_year: int
    revenue: float | None = None
    gross_margin_pct: float | None = None
    operating_margin_pct: float | None = None
    net_margin_pct: float | None = None
    revenue_growth_pct: float | None = None
    net_income_growth_pct: float | None = None
    roe_pct: float | None = None
    roa_pct: float | None = None
    liabilities_to_equity: float | None = None


class MetricsSummary(BaseModel):
    """Multi-year metrics plus cross-period aggregates."""

    annual: list[AnnualMetrics]
    revenue_cagr_pct: float | None = None


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    """Divide, returning None on missing inputs or zero denominator."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _pct(value: float | None) -> float | None:
    """Fraction to percentage, preserving None."""
    return None if value is None else value * 100


def _growth_pct(current: float | None, previous: float | None) -> float | None:
    """Year-over-year growth percentage; None if undefined."""
    if current is None or previous is None or previous == 0:
        return None
    return (current - previous) / abs(previous) * 100


def _by_year(facts: dict[str, list[FinancialFact]]) -> dict[int, dict[str, float]]:
    """Pivot metric→facts into fiscal_year→{metric: value}."""
    table: dict[int, dict[str, float]] = {}
    for metric, series in facts.items():
        for fact in series:
            table.setdefault(fact.fiscal_year, {})[metric] = fact.value
    return table


def compute_revenue_cagr(annual: list[AnnualMetrics]) -> float | None:
    """Compound annual growth rate of revenue across the series."""
    revenues = [(m.fiscal_year, m.revenue) for m in annual if m.revenue]
    if len(revenues) < 2:
        return None
    (first_year, first), (last_year, last) = revenues[0], revenues[-1]
    years = last_year - first_year
    if years <= 0 or first <= 0 or last <= 0:
        return None
    return ((last / first) ** (1 / years) - 1) * 100


def compute_metrics(facts: dict[str, list[FinancialFact]]) -> MetricsSummary:
    """Compute all annual ratios and aggregates from raw facts."""
    table = _by_year(facts)
    annual: list[AnnualMetrics] = []
    previous: dict[str, float] = {}

    for year in sorted(table):
        row = table[year]
        revenue = row.get("revenue")
        annual.append(
            AnnualMetrics(
                fiscal_year=year,
                revenue=revenue,
                gross_margin_pct=_pct(_safe_div(row.get("gross_profit"), revenue)),
                operating_margin_pct=_pct(
                    _safe_div(row.get("operating_income"), revenue)
                ),
                net_margin_pct=_pct(_safe_div(row.get("net_income"), revenue)),
                revenue_growth_pct=_growth_pct(revenue, previous.get("revenue")),
                net_income_growth_pct=_growth_pct(
                    row.get("net_income"), previous.get("net_income")
                ),
                roe_pct=_pct(
                    _safe_div(row.get("net_income"), row.get("shareholders_equity"))
                ),
                roa_pct=_pct(_safe_div(row.get("net_income"), row.get("total_assets"))),
                liabilities_to_equity=_safe_div(
                    row.get("total_liabilities"), row.get("shareholders_equity")
                ),
            )
        )
        previous = row

    return MetricsSummary(
        annual=annual, revenue_cagr_pct=compute_revenue_cagr(annual)
    )