"""Convert SEC filing HTML into clean plain text for chunking and RAG."""

import re

from bs4 import BeautifulSoup

from financial_research_agent.logging_config import get_logger

log = get_logger(__name__)

# Tags whose content is never useful prose.
_NOISE_TAGS = ("script", "style", "ix:header")

_MULTISPACE = re.compile(r"[ \t]+")
_MULTINEWLINE = re.compile(r"\n{3,}")


def html_to_text(html: str) -> str:
    """Return clean plain text from filing HTML.

    Removes scripts, styles and hidden XBRL metadata; preserves
    paragraph breaks; collapses runs of whitespace.
    """
    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    text = soup.get_text(separator="\n")

    text = _MULTISPACE.sub(" ", text)
    lines = (line.strip() for line in text.splitlines())
    text = "\n".join(line for line in lines if line)
    text = _MULTINEWLINE.sub("\n\n", text)

    log.info("html_cleaned", input_chars=len(html), output_chars=len(text))
    return text