"""Tests for the broaden-search fallback logic — no live model or network calls."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinical_rag.agent.clinical_agent import _is_weak, broaden_decision

# ── tests ─────────────────────────────────────────────────────────────────────


def test_weak_first_pass_allows_broaden():
    retrieved = {1: {"distance": 0.6}}
    assert broaden_decision(retrieved, False, 0.45) is None


def test_strong_first_pass_refuses():
    retrieved = {1: {"distance": 0.1}}
    result = broaden_decision(retrieved, False, 0.45)
    assert result is not None


def test_broaden_never_twice():
    retrieved = {1: {"distance": 0.6}}
    result = broaden_decision(retrieved, True, 0.45)
    assert result is not None
    assert "only be used once" in result


def test_empty_results_is_weak():
    assert _is_weak({}, 0.45) is True
    assert broaden_decision({}, False, 0.45) is None
