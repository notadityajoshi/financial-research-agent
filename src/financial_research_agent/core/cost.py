"""Deterministic LLM cost calculation from token usage."""

from pydantic import BaseModel

# USD per 1M tokens (input, output). Ollama is local and free.
# Extend as providers are added; unknown models cost 0 (with a warning upstream).
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "llama3.2:3b": (0.0, 0.0),
    "nomic-embed-text": (0.0, 0.0),
}


class CostBreakdown(BaseModel):
    """Token usage and USD cost for one or many LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0

    def __add__(self, other: "CostBreakdown") -> "CostBreakdown":
        return CostBreakdown(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            usd=round(self.usd + other.usd, 6),
        )


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> CostBreakdown:
    """Cost of one call. Unknown models are treated as free (0.0)."""
    in_price, out_price = PRICE_TABLE.get(model, (0.0, 0.0))
    usd = input_tokens / 1e6 * in_price + output_tokens / 1e6 * out_price
    return CostBreakdown(
        input_tokens=input_tokens, output_tokens=output_tokens, usd=round(usd, 6)
    )
