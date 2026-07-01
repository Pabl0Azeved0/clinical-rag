"""Tests for the MCP retrieval server — mocked store, no model or network."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastmcp import Client

import clinical_rag.mcp_server.server as server


class _FakeStore:
    """Minimal store: records the k it was queried with, returns fixed passages."""

    def __init__(self):
        self.last_k = None

    def query(self, text: str, k: int) -> list[dict]:
        self.last_k = k
        return [
            {
                "text": "A1C measures average blood glucose.",
                "title": "A1C",
                "url": "https://medlineplus.gov/a1c.html",
                "doc_id": "d1",
                "distance": 0.1,
            }
        ]


def _call(query: str, k: int):
    async def run():
        async with Client(server.mcp) as client:
            res = await client.call_tool(
                "search_clinical_evidence", {"query": query, "k": k}
            )
            return res.data

    return asyncio.run(run())


def test_tool_is_registered():
    async def run():
        async with Client(server.mcp) as client:
            return [t.name for t in await client.list_tools()]

    assert asyncio.run(run()) == ["search_clinical_evidence"]


def test_search_returns_passages(monkeypatch):
    fake = _FakeStore()
    monkeypatch.setattr(server, "_store", lambda: fake)

    data = _call("what is the A1C test", 3)

    assert data[0]["title"] == "A1C"
    assert data[0]["url"] == "https://medlineplus.gov/a1c.html"
    assert set(data[0]) >= {"text", "title", "url", "distance"}
    assert fake.last_k == 3  # per-call k is forwarded through the retriever
