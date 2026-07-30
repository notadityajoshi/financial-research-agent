"""Streamlit UI: thin client over the research API. No business logic here."""

import streamlit as st

from financial_research_agent.config import get_settings
from financial_research_agent.frontend.api_client import ResearchAPIClient

st.set_page_config(page_title="Financial Research Agent", page_icon="📊")


@st.cache_resource
def get_client() -> ResearchAPIClient:
    settings = get_settings()
    first_key = next(iter(settings.api_keys.split(",")), "").strip()
    return ResearchAPIClient(settings.api_base_url, api_key=first_key)


client = get_client()

st.title("Autonomous Financial Research Agent")
st.caption("SEC filings · deterministic metrics · cited RAG insights · PDF report")

if not client.health():
    st.error("API is unreachable. Start it: `uv run uvicorn --factory "
             "financial_research_agent.api.main:create_app --port 8000`")
    st.stop()

ticker = st.text_input("Ticker", value="NVDA", max_chars=10)

if st.button("Analyze", type="primary", disabled=not ticker.strip()):
    try:
        run = client.create_run(ticker.strip())
        st.session_state["run_id"] = run.id
    except Exception as exc:  # noqa: BLE001 — surface API errors to the user
        st.error(f"Could not start run: {exc}")

run_id = st.session_state.get("run_id")
if run_id:
    run = client.get_run(run_id)
    st.divider()
    st.write(f"**Run** `{run.id}` — **{run.ticker}**")

    if run.status == "completed":
        st.success("Completed")
        st.download_button(
            "Download PDF report",
            data=client.get_report(run.id),
            file_name=f"{run.ticker}_report.pdf",
            mime="application/pdf",
        )
    elif run.status == "failed":
        st.error(f"Failed: {run.error}")
    else:
        st.info(f"Status: {run.status} — analysis takes a few minutes locally")
        st.button("Refresh status")