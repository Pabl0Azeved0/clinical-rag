"""Tests for optional Logfire tracing — no real logfire.configure calls."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import logfire

import clinical_rag.observability as observability
from clinical_rag.observability import setup_tracing

# ── tests ─────────────────────────────────────────────────────────────────────


def test_disabled_is_noop(monkeypatch):
    calls = []
    monkeypatch.setattr(logfire, "configure", lambda **kw: calls.append(kw))

    settings = SimpleNamespace(enable_tracing=False, logfire_token=None)
    result = setup_tracing(settings)

    assert result is False
    assert calls == []


def test_enabled_configures_and_instruments(monkeypatch):
    # Reset the idempotency guard so setup_tracing runs fresh.
    observability._CONFIGURED["done"] = False

    configure_calls = []
    instrument_pydantic_calls = []
    instrument_openai_calls = []

    monkeypatch.setattr(logfire, "configure", lambda **kw: configure_calls.append(kw))
    monkeypatch.setattr(
        logfire, "instrument_pydantic_ai", lambda: instrument_pydantic_calls.append(1)
    )
    monkeypatch.setattr(
        logfire, "instrument_openai", lambda: instrument_openai_calls.append(1)
    )

    settings = SimpleNamespace(enable_tracing=True, logfire_token=None)
    result = setup_tracing(settings)

    assert result is True
    assert len(configure_calls) == 1
    assert configure_calls[0]["send_to_logfire"] is False
    assert len(instrument_pydantic_calls) == 1
    assert len(instrument_openai_calls) == 1

    # Cleanup so other tests are not affected by the guard being set.
    observability._CONFIGURED["done"] = False
