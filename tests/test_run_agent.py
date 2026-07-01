"""Tests for run_agent — trajectory extraction + graceful retry-exhaustion refusal."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel

from clinical_rag.agent.clinical_agent import (
    UNGROUNDED_FALLBACK,
    AgentDeps,
    build_agent,
    run_agent,
)
from clinical_rag.agent.models import ClinicalAnswer


class _FakeRetriever:
    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        return [
            {
                "text": "A1C measures average blood glucose.",
                "title": "A1C",
                "url": "https://medlineplus.gov/a1c.html",
                "doc_id": "d1",
                "distance": 0.1,
            }
        ]


def _settings():
    return SimpleNamespace(
        llm_provider="ollama",
        llm_model="llama3.2:3b",
        llm_base_url="http://localhost:11434",
        llm_api_key=None,
        weak_distance_threshold=0.45,
        retrieval_transport="in_process",
        mcp_server_url=None,
    )


def test_run_agent_returns_output_and_trajectory():
    canned = json.dumps(
        {
            "answer": "A1C measures average blood glucose [1].",
            "citations": [
                {"index": 1, "title": "A1C", "url": "https://medlineplus.gov/a1c.html"}
            ],
            "grounded": True,
            "confidence": 0.9,
        }
    )
    settings = _settings()
    deps = AgentDeps(retriever=_FakeRetriever(), settings=settings)
    agent = build_agent(settings)
    with agent.override(model=TestModel(custom_output_text=canned)):
        output, tool_calls = run_agent(agent, "what is the A1C test", deps)

    assert isinstance(output, ClinicalAnswer)
    assert "search_evidence" in [tc["tool"] for tc in tool_calls]


def test_run_agent_degrades_on_retry_exhaustion():
    class _BoomAgent:
        def run_sync(self, question, deps=None):
            raise UnexpectedModelBehavior("Exceeded maximum output retries (3)")

    output, tool_calls = run_agent(_BoomAgent(), "anything", deps=None)

    assert output.grounded is False
    assert output.citations == []
    assert output.answer == UNGROUNDED_FALLBACK
    assert tool_calls == []
