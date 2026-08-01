"""Unit tests for deterministic metric calculations (offline, synthetic data)."""

import pytest

from financial_research_agent.core.metrics import compute_metrics
from financial_research_agent.integrations.financial_data import FinancialFact


def _fact(metric: str, year: int, value: float) -> FinancialFact:
    return FinancialFact(
        metric=metric,
        concept="Test",
        value=value,
        unit="USD",
        fiscal_year=year,
        end_date=f"{year}-12-31",
        form="10-K",
    )


@pytest.fixture
def facts() -> dict[str, list[FinancialFact]]:
    return {
        "revenue": [_fact("revenue", 2023, 100.0), _fact("revenue", 2024, 150.0)],
        "gross_profit": [_fact("gross_profit", 2024, 90.0)],
        "net_income": [
            _fact("net_income", 2023, 20.0),
            _fact("net_income", 2024, 30.0),
        ],
        "shareholders_equity": [_fact("shareholders_equity", 2024, 120.0)],
        "total_assets": [_fact("total_assets", 2024, 300.0)],
        "total_liabilities": [_fact("total_liabilities", 2024, 180.0)],
    }


def test_margins_and_ratios(facts: dict[str, list[FinancialFact]]) -> None:
    latest = compute_metrics(facts).annual[-1]
    assert latest.fiscal_year == 2024
    assert latest.gross_margin_pct == pytest.approx(60.0)
    assert latest.net_margin_pct == pytest.approx(20.0)
    assert latest.roe_pct == pytest.approx(25.0)
    assert latest.roa_pct == pytest.approx(10.0)
    assert latest.liabilities_to_equity == pytest.approx(1.5)


def test_growth(facts: dict[str, list[FinancialFact]]) -> None:
    latest = compute_metrics(facts).annual[-1]
    assert latest.revenue_growth_pct == pytest.approx(50.0)
    assert latest.net_income_growth_pct == pytest.approx(50.0)


def test_first_year_growth_is_none(facts: dict[str, list[FinancialFact]]) -> None:
    first = compute_metrics(facts).annual[0]
    assert first.revenue_growth_pct is None


def test_cagr(facts: dict[str, list[FinancialFact]]) -> None:
    assert compute_metrics(facts).revenue_cagr_pct == pytest.approx(50.0)


def test_zero_denominator_returns_none() -> None:
    facts = {
        "revenue": [_fact("revenue", 2024, 0.0)],
        "gross_profit": [_fact("gross_profit", 2024, 10.0)],
    }
    assert compute_metrics(facts).annual[0].gross_margin_pct is None


def test_missing_metric_returns_none(facts: dict[str, list[FinancialFact]]) -> None:
    assert compute_metrics(facts).annual[-1].operating_margin_pct is None
