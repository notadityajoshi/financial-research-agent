"""Offline unit tests for filing HTML cleaning."""

from financial_research_agent.ingestion.html_cleaner import html_to_text

SAMPLE = """
<html><head><style>p { color: red; }</style>
<script>alert('x');</script></head>
<body>
<ix:header><ix:hidden>meta-noise</ix:hidden></ix:header>
<p>Item 1A.   Risk\tFactors</p>


<p>Demand for our products may&nbsp;decline.</p>
</body></html>
"""


def test_noise_tags_removed() -> None:
    text = html_to_text(SAMPLE)
    assert "alert" not in text
    assert "color" not in text
    assert "meta-noise" not in text


def test_prose_preserved() -> None:
    text = html_to_text(SAMPLE)
    assert "Risk Factors" in text
    assert "Demand for our products" in text


def test_whitespace_collapsed() -> None:
    text = html_to_text(SAMPLE)
    assert "  " not in text
    assert "\n\n\n" not in text


def test_entities_decoded() -> None:
    assert "may decline" in html_to_text(SAMPLE)


def test_empty_input() -> None:
    assert html_to_text("") == ""
