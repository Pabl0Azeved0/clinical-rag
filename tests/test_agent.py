"""Tests for the PydanticAI agent layer — no live model or network calls."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinical_rag.agent.clinical_agent import AgentDeps, _format_evidence, build_agent
from clinical_rag.agent.models import ClinicalAnswer
from clinical_rag.retrieval.retriever import Retriever

# ── helpers ───────────────────────────────────────────────────────────────────


def _fake_results() -> list[dict]:
    return [
        {
            "text": "Passage about A1C.",
            "title": "A1C Test",
            "url": "https://medlineplus.gov/a1c.html",
            "doc_id": "d1",
            "distance": 0.1,
        },
        {
            "text": "Passage about blood sugar.",
            "title": "Blood Sugar",
            "url": "https://medlineplus.gov/bloodsugar.html",
            "doc_id": "d2",
            "distance": 0.2,
        },
    ]


def _test_settings():
    return SimpleNamespace(
        llm_provider="ollama",
        llm_model="llama3.2:3b",
        llm_base_url="http://localhost:11434",
        llm_api_key=None,
        weak_distance_threshold=0.45,
        retrieval_transport="in_process",
        mcp_server_url=None,
    )


# ── test_retrieve_k_override ──────────────────────────────────────────────────


class _RecordingStore:
    """Fake store that records the k passed to query()."""

    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def query(self, text: str, k: int) -> list[dict]:
        self.calls.append((text, k))
        return []


def test_retrieve_k_override():
    store = _RecordingStore()
    retriever = Retriever(store, top_k=5)

    retriever.retrieve("q")
    assert store.calls[-1] == ("q", 5)

    retriever.retrieve("q", 8)
    assert store.calls[-1] == ("q", 8)


# ── test_format_evidence_numbers_and_maps ─────────────────────────────────────


def test_format_evidence_numbers_and_maps():
    results = _fake_results()
    context, numbered = _format_evidence(results)

    assert "[1]" in context
    assert "[2]" in context
    assert "A1C Test" in context
    assert "https://medlineplus.gov/a1c.html" in context
    assert "Blood Sugar" in context
    assert "https://medlineplus.gov/bloodsugar.html" in context

    assert numbered == {1: results[0], 2: results[1]}


# ── test_agent_runs_and_populates_numbered_map ────────────────────────────────


class _FakeRetriever:
    """Duck-typed retriever that returns two fixed results."""

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        return _fake_results()


def test_agent_runs_and_populates_numbered_map():
    import json

    from pydantic_ai.models.test import TestModel

    # The agent uses PromptedOutput, so the model's final output is parsed from text.
    # TestModel calls the tool (call_tools defaults to all) then returns this JSON.
    canned = json.dumps(
        {
            "answer": "A1C measures average blood glucose [1].",
            "citations": [
                {
                    "index": 1,
                    "title": "A1C Test",
                    "url": "https://medlineplus.gov/a1c.html",
                }
            ],
            "grounded": True,
            "confidence": 0.9,
        }
    )

    settings = _test_settings()
    deps = AgentDeps(retriever=_FakeRetriever(), settings=settings)
    agent = build_agent(settings)

    with agent.override(model=TestModel(custom_output_text=canned)):
        result = agent.run_sync("what is the A1C test", deps=deps)

    assert isinstance(result.output, ClinicalAnswer)
    assert result.output.citations[0].index == 1
    assert deps.retrieved  # R6: map must be non-empty after tool ran


# ── build_model provider wiring ───────────────────────────────────────────────


def test_build_model_groq_uses_openai_chat_model():
    import pytest
    from pydantic_ai.models.openai import OpenAIChatModel

    from clinical_rag.agent.clinical_agent import build_model

    model = build_model(
        SimpleNamespace(
            llm_provider="groq", llm_model="llama-3.3-70b-versatile", llm_api_key="k"
        )
    )
    assert isinstance(model, OpenAIChatModel)

    with pytest.raises(ValueError):
        build_model(
            SimpleNamespace(llm_provider="bogus", llm_model="x", llm_api_key=None)
        )
