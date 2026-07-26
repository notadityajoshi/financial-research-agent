"""Token-aware chunking of clean filing text for retrieval."""

import re

import tiktoken
from pydantic import BaseModel

from financial_research_agent.logging_config import get_logger

log = get_logger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class Chunk(BaseModel):
    """One retrievable piece of a source document, with provenance."""

    text: str
    index: int
    token_count: int
    metadata: dict[str, str]


def count_tokens(text: str) -> int:
    """Token count under the cl100k_base encoding."""
    return len(_ENCODING.encode(text))


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """Split by raw tokens when no natural boundary fits."""
    tokens = _ENCODING.encode(text)
    return [
        _ENCODING.decode(tokens[i : i + max_tokens])
        for i in range(0, len(tokens), max_tokens)
    ]


def _split_oversized(paragraph: str, max_tokens: int) -> list[str]:
    """Split an oversized paragraph by sentences, then raw tokens."""
    pieces: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(paragraph):
        if count_tokens(sentence) <= max_tokens:
            pieces.append(sentence)
        else:
            pieces.extend(_hard_split(sentence, max_tokens))
    return pieces


def _overlap_tail(text: str, overlap_tokens: int) -> str:
    """Last `overlap_tokens` tokens of text, decoded."""
    tokens = _ENCODING.encode(text)
    return _ENCODING.decode(tokens[-overlap_tokens:]) if tokens else ""


def chunk_text(
    text: str, *, max_tokens: int = 512, overlap_tokens: int = 64
) -> list[str]:
    """Pack text into chunks of at most max_tokens, with token overlap.

    Invariant: every returned chunk is <= max_tokens.
    """
    pieces: list[str] = []
    for paragraph in filter(None, (p.strip() for p in text.split("\n"))):
        if count_tokens(paragraph) <= max_tokens:
            pieces.append(paragraph)
        else:
            pieces.extend(_split_oversized(paragraph, max_tokens))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n{piece}" if current else piece
        if count_tokens(candidate) <= max_tokens:
            current = candidate
            continue
        if current:
            chunks.append(current)
        seeded = f"{_overlap_tail(current, overlap_tokens)}\n{piece}"
        current = seeded if count_tokens(seeded) <= max_tokens else piece
    if current:
        chunks.append(current)
    return chunks


def build_chunks(
    text: str,
    metadata: dict[str, str],
    *,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    """Chunk text and attach provenance metadata to every chunk."""
    chunks = [
        Chunk(text=t, index=i, token_count=count_tokens(t), metadata=dict(metadata))
        for i, t in enumerate(
            chunk_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        )
    ]
    log.info("chunks_built", count=len(chunks))
    return chunks