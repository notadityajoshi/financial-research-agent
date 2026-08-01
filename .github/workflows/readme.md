![CI](https://github.com/notadityajoshi/financial-research-agent/actions/workflows/ci.yml/badge.svg)

# Autonomous Financial Research Agent

Multi-agent system that researches a public company end-to-end from a single ticker: SEC filings, official XBRL fundamentals, news, hybrid RAG over the 10-K with **verifiable citations**, deterministic financial metrics, and a professional PDF report.

**Input:** `"NVDA"` → **Output:** cited investment research PDF in ~2 minutes.

## Architecture

    Streamlit → FastAPI → Redis (arq) → Worker
                                          └─ LangGraph
                                              ├─ parallel: SEC filings / XBRL facts / news
                                              ├─ deterministic metrics (pure functions)
                                              ├─ RAG branch: index 10-K → hybrid retrieval → cited insights
                                              ├─ parallel LLM analysts: risks / opportunities
                                              └─ summary → PDF

    PostgreSQL (run state) · Qdrant (vectors) · Ollama (local LLM)
    Langfuse + OpenTelemetry (observability)

## Key engineering decisions

| Decision | Why |
|---|---|
| LLM interprets, never calculates | Every number is arithmetic from SEC XBRL; the LLM only reasons over pre-computed context |
| Citations resolved by the system | LLM cites excerpt IDs; the system resolves them to stored chunk text; uncited claims are discarded |
| Hybrid retrieval + reranking | BM25 + dense embeddings fused with RRF, cross-encoder rerank on top |
| Fault-isolated graph nodes | One dead data source degrades the report (visible "Data Gaps" section) instead of killing the run |
| Provider abstraction | Ollama locally (£0), OpenAI/Bedrock in prod — config change only, zero code change |
| Offline test suite | 88 tests, no network, no paid APIs: fakes, in-memory Qdrant, aiosqlite. CI runs the lot free |

## Quickstart

```bash
git clone https://github.com/notadityajoshi/financial-research-agent.git
cd financial-research-agent
cp .env.example .env.docker   # set SEC_USER_AGENT to "AppName your@email.com"
docker compose up --build
```

Then:

```bash
curl -X POST localhost:8000/runs \
  -H 'Content-Type: application/json' -H 'X-API-Key: dev-local-key-change-me' \
  -d '{"ticker":"NVDA"}'
# poll /runs/{id}; download /runs/{id}/report when completed
```

API docs at `localhost:8000/docs`. Streamlit UI: `uv run streamlit run src/financial_research_agent/frontend/app.py`.

## Evaluation

Deterministic harness (`scripts/evaluate.py`): recall@5 and MRR over a golden query set against the live index, plus a citation-groundedness check verifying every cited excerpt is a real substring of the stored corpus.

## Cost

£0 baseline. Local LLM (Ollama), free data sources (SEC EDGAR, XBRL, Google News RSS), free-tier observability (Langfuse). Per-run token/cost accounting persisted to Postgres for when a paid provider is switched on.

## Stack

Python 3.12 · uv · FastAPI · LangGraph · Pydantic · SQLAlchemy async + Alembic · PostgreSQL · Redis + arq · Qdrant · sentence-transformers · fpdf2 · structlog · OpenTelemetry · Langfuse · Docker Compose · GitHub Actions · pytest (88 offline tests)