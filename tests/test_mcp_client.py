"""The agent fetches retrieval over MCP when retrieval_transport='mcp'."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic_ai.models.test import TestModel

import clinical_rag.mcp_server.server as server
from clinical_rag.agent.clinical_agent import AgentDeps, build_agent
from clinical_rag.agent.models import ClinicalAnswer


class _FakeStore:
    def query(self, text, k):
        return [
            {
                "text": "A1C measures average blood glucose.",
                "title": "A1C",
                "url": "https://medlineplus.gov/a1c.html",
                "doc_id": "d1",
                "distance": 0.1,
            }
        ]


class _NoRetriever:
    def retrieve(self, query, k=None):
        raise AssertionError(
            "in-process retriever must not be used under mcp transport"
        )


def _settings():
    return SimpleNamespace(
        llm_provider="ollama",
        llm_model="llama3.2:3b",
        llm_base_url="http://localhost:11434",
        llm_api_key=None,
        weak_distance_threshold=0.45,
        retrieval_transport="mcp",
        mcp_server_url=None,
    )


def test_agent_retrieves_over_mcp(monkeypatch):
    monkeypatch.setattr(server, "_store", lambda: _FakeStore())
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
    deps = AgentDeps(retriever=_NoRetriever(), settings=settings)
    agent = build_agent(settings)
    with agent.override(model=TestModel(custom_output_text=canned)):
        result = agent.run_sync("what is the A1C test", deps=deps)

    assert isinstance(result.output, ClinicalAnswer)
    assert deps.retrieved  # filled via the MCP round-trip, not the local retriever
