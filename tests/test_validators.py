"""Tests for the grounding validator — no live model or network calls."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic_ai import ModelRetry

from clinical_rag.agent.models import ClinicalAnswer
from clinical_rag.agent.validators import check_grounding

# ── helpers ───────────────────────────────────────────────────────────────────


def _answer(**kwargs) -> ClinicalAnswer:
    defaults = dict(answer="", grounded=False, confidence=0.0, citations=[])
    defaults.update(kwargs)
    return ClinicalAnswer(**defaults)


def _retrieved_pair() -> dict[int, dict]:
    return {
        1: {
            "title": "A1C Test",
            "url": "https://medlineplus.gov/a1c.html",
            "distance": 0.1,
        },
        2: {
            "title": "Blood Sugar",
            "url": "https://medlineplus.gov/bloodsugar.html",
            "distance": 0.2,
        },
    }


# ── tests ─────────────────────────────────────────────────────────────────────


def test_fabricated_citation_raises():
    output = _answer(answer="Some claim [9].", grounded=False, confidence=0.0)
    with pytest.raises(ModelRetry):
        check_grounding(output, _retrieved_pair())


def test_no_citation_sets_grounded_false():
    output = _answer(answer="No citations here.", grounded=True, confidence=0.9)
    result = check_grounding(output, _retrieved_pair())
    assert result.grounded is False
    assert result.citations == []


def test_valid_citation_regrounds():
    retrieved = {
        1: {"title": "A1C", "url": "https://medlineplus.gov/a1c.html", "distance": 0.1}
    }
    output = _answer(answer="A1C measures glucose [1].", grounded=False, confidence=0.0)
    result = check_grounding(output, retrieved)
    assert result.grounded is True
    assert result.citations[0].index == 1
    assert result.citations[0].title == "A1C"
    assert result.confidence == pytest.approx(0.9)


def test_uncited_strong_evidence_nudges():
    retrieved = {1: {"title": "A1C", "url": "u", "distance": 0.1}}
    output = _answer(answer="A1C measures glucose.")
    with pytest.raises(ModelRetry):
        check_grounding(output, retrieved, weak_threshold=0.45, allow_nudge=True)


def test_uncited_weak_evidence_no_nudge():
    retrieved = {1: {"title": "A1C", "url": "u", "distance": 0.6}}
    output = _answer(answer="A1C measures glucose.")
    result = check_grounding(output, retrieved, weak_threshold=0.45, allow_nudge=True)
    assert result.grounded is False


def test_uncited_nudge_disabled_on_last_retry():
    retrieved = {1: {"title": "A1C", "url": "u", "distance": 0.1}}
    output = _answer(answer="A1C measures glucose.")
    result = check_grounding(output, retrieved, weak_threshold=0.45, allow_nudge=False)
    assert result.grounded is False
