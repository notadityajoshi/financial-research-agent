"""Golden evaluation dataset: queries with known-relevant markers."""

from pydantic import BaseModel


class RetrievalCase(BaseModel):
    """One retrieval test: a query and substrings that mark relevant chunks."""

    query: str
    relevant_markers: list[str]


# Concept-level markers: a retrieved chunk is 'relevant' if it contains any.
# Generic on purpose — works across large-cap 10-Ks, not just one company.
GOLDEN_RETRIEVAL: list[RetrievalCase] = [
    RetrievalCase(
        query="principal risk factors",
        relevant_markers=["risk", "adversely", "could harm", "uncertain"],
    ),
    RetrievalCase(
        query="competition and market position",
        relevant_markers=["compet", "market", "rivals", "pricing"],
    ),
    RetrievalCase(
        query="revenue and results of operations",
        relevant_markers=["revenue", "net income", "operating", "gross margin"],
    ),
    RetrievalCase(
        query="liquidity and capital resources",
        relevant_markers=["liquidity", "cash", "capital", "credit"],
    ),
]
