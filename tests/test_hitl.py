"""Tests for the HITL gate — no live model or network calls."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinical_rag.agent.hitl import Decision, apply_decision, needs_approval
from clinical_rag.agent.models import ClinicalAnswer

# ── helpers ───────────────────────────────────────────────────────────────────


def _answer(**kwargs) -> ClinicalAnswer:
    defaults = dict(answer="some answer", grounded=True, confidence=0.8, citations=[])
    defaults.update(kwargs)
    return ClinicalAnswer(**defaults)


# ── tests ─────────────────────────────────────────────────────────────────────


def test_needs_approval_when_ungrounded():
    answer = _answer(grounded=False, confidence=0.9)
    assert needs_approval(answer, confidence_threshold=0.6) is True


def test_needs_approval_when_low_confidence():
    answer = _answer(grounded=True, confidence=0.4)
    assert needs_approval(answer, confidence_threshold=0.6) is True


def test_no_approval_when_grounded_and_confident():
    answer = _answer(grounded=True, confidence=0.8)
    assert needs_approval(answer, confidence_threshold=0.6) is False


def test_apply_decision_approve_returns_original():
    answer = _answer()
    result = apply_decision(answer, Decision("approve"))
    assert result is answer


def test_apply_decision_reject_returns_none():
    answer = _answer()
    result = apply_decision(answer, Decision("reject"))
    assert result is None


def test_apply_decision_edit_replaces_text():
    answer = _answer(answer="original text", grounded=True, confidence=0.7)
    result = apply_decision(answer, Decision("edit", "new text"))
    assert result is not None
    assert result.answer == "new text"
    assert result.grounded == answer.grounded
    assert result.confidence == answer.confidence
    assert result.citations == answer.citations


def test_apply_decision_edit_without_text_raises():
    answer = _answer()
    with pytest.raises(ValueError):
        apply_decision(answer, Decision("edit", None))
