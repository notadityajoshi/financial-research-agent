"""Offline unit tests for token-aware chunking."""

from financial_research_agent.ingestion.chunking import (
    build_chunks,
    chunk_text,
    count_tokens,
)

SENTENCES = " ".join(f"Sentence number {i} ends right here." for i in range(60))


def test_short_text_single_chunk() -> None:
    chunks = chunk_text("One small paragraph.")
    assert chunks == ["One small paragraph."]


def test_max_tokens_respected() -> None:
    for chunk in chunk_text(SENTENCES, max_tokens=48, overlap_tokens=12):
        assert count_tokens(chunk) <= 48


def test_overlap_between_chunks() -> None:
    chunks = chunk_text(SENTENCES, max_tokens=48, overlap_tokens=12)
    assert len(chunks) > 1
    last_word = chunks[0].split()[-1]
    assert last_word in chunks[1]


def test_long_unbroken_text_hard_split() -> None:
    giant = "word " * 500  # no punctuation: forces raw token split
    for chunk in chunk_text(giant, max_tokens=32, overlap_tokens=4):
        assert count_tokens(chunk) <= 32


def test_metadata_and_indices() -> None:
    chunks = build_chunks(SENTENCES, {"ticker": "NVDA"}, max_tokens=48)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["ticker"] == "NVDA" for c in chunks)
    assert all(c.token_count == count_tokens(c.text) for c in chunks)
