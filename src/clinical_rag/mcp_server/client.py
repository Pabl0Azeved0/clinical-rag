"""Client helper: fetch retrieval passages from the MCP server.

Uses an in-memory (same-process) transport by default so grounding/numbering can
stay in the agent (R6); set MCP_SERVER_URL to hit a real running server instead.
"""

from __future__ import annotations

from fastmcp import Client

from clinical_rag.config import Settings


def _client(settings: Settings) -> Client:
    if settings.mcp_server_url:
        return Client(settings.mcp_server_url)
    from clinical_rag.mcp_server.server import mcp

    return Client(mcp)


async def search_evidence_over_mcp(
    settings: Settings, query: str, k: int
) -> list[dict]:
    async with _client(settings) as client:
        res = await client.call_tool(
            "search_clinical_evidence", {"query": query, "k": k}
        )
        return list(res.data)
