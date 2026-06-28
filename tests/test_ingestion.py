"""Tests for ingestion/medline.py — no network required."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinical_rag.ingestion.medline import _strip_html, parse_topics

# ── _strip_html ──────────────────────────────────────────────────────────────

def test_strip_html_removes_tags():
    assert _strip_html("<p>Hello <b>world</b></p>") == "Hello world"


def test_strip_html_unescapes_entities():
    # Entities are unescaped first (&amp; → &), then any resulting tags are stripped
    assert _strip_html("&lt;p&gt;Type 2 &amp; Type 1&lt;/p&gt;") == "Type 2 & Type 1"


def test_strip_html_collapses_whitespace():
    assert _strip_html("<p>  too   many   spaces  </p>") == "too many spaces"


def test_strip_html_empty():
    assert _strip_html("") == ""


def test_strip_html_no_tags():
    assert _strip_html("plain text") == "plain text"


# ── parse_topics ─────────────────────────────────────────────────────────────

_FIXTURE_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<health-topics total="3">
  <health-topic title="A1C" url="https://medlineplus.gov/a1c.html" id="1001" language="English">
    <also-called>Hemoglobin A1C</also-called>
    <full-summary>&lt;p&gt;The A1C test measures average blood sugar.&lt;/p&gt;</full-summary>
  </health-topic>
  <health-topic title="Diabetes (Spanish)" url="https://medlineplus.gov/sp/diabetes.html" id="1002" language="Spanish">
    <full-summary>&lt;p&gt;Contenido en español.&lt;/p&gt;</full-summary>
  </health-topic>
  <health-topic title="No Summary Topic" url="https://medlineplus.gov/nosummary.html" id="1003" language="English">
  </health-topic>
</health-topics>
"""


@pytest.fixture()
def xml_path(tmp_path: Path) -> Path:
    p = tmp_path / "test_topics.xml"
    p.write_text(_FIXTURE_XML, encoding="utf-8")
    return p


def test_parse_topics_english_only(xml_path):
    docs = parse_topics(xml_path)
    titles = [d.title for d in docs]
    assert "Diabetes (Spanish)" not in titles, "Spanish topic should be filtered out"


def test_parse_topics_skips_empty_summary(xml_path):
    docs = parse_topics(xml_path)
    titles = [d.title for d in docs]
    assert "No Summary Topic" not in titles, "Topic with no summary should be skipped"


def test_parse_topics_count(xml_path):
    docs = parse_topics(xml_path)
    assert len(docs) == 1


def test_parse_topics_fields(xml_path):
    doc = parse_topics(xml_path)[0]
    assert doc.id == "1001"
    assert doc.title == "A1C"
    assert doc.url == "https://medlineplus.gov/a1c.html"
    assert "average blood sugar" in doc.text
    assert "<p>" not in doc.text, "HTML tags should be stripped"
    assert doc.also_called == ["Hemoglobin A1C"]


def test_parse_topics_html_stripped(xml_path):
    doc = parse_topics(xml_path)[0]
    assert "&lt;" not in doc.text, "HTML entities should be unescaped"
    assert "<p>" not in doc.text
