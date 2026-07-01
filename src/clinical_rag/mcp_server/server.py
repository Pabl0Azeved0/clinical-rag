"""Standalone MCP server exposing clinical-evidence retrieval as one typed tool.

Retrieval only — no generation, no diagnosis, no PHI. The scope is deliberately
narrow so the tool can be shared with other agents without exposing the LLM or
any patient data. Numbering, citation grounding, and answer generation stay in
the agent process (see the grounding validator); this server just returns the
raw retrieved passages.

Run standalone (stdio transport):
    .venv/bin/python -m clinical_rag.mcp_server.server
"""

from __future__ import annotations

from fastmcp import FastMCP

from clinical_rag.config import get_settings
from clinical_rag.embeddings.embedder import Embedder
from clinical_rag.retrieval.retriever import Retriever
from clinical_rag.vectorstore.chroma_store import ChromaStore

mcp = FastMCP("clinical-evidence")

# The embedder + store are heavy to build, so load them lazily once per process.
_STORE = {}


def _store() -> ChromaStore:
    if "store" not in _STORE:
        cfg = get_settings()
        _STORE["store"] = ChromaStore(
            cfg.resolve(cfg.chroma_dir),
            cfg.collection_name,
            Embedder(cfg.embedding_model_name),
        )
    return _STORE["store"]


@mcp.tool
def search_clinical_evidence(query: str, k: int = 5) -> list[dict]:
    """Search indexed MedlinePlus health topics for passages relevant to a query.

    Returns up to k passages, each a dict with keys: text, title, url, doc_id,
    distance (cosine; lower is closer). Retrieval only — this tool does not
    diagnose, give medical advice, or accept/return any personal health data.
    """
    if k <= 0:  # defensive: a public tool shouldn't crash on a bad k
        k = 5
    return Retriever(_store(), k).retrieve(query, k)


if __name__ == "__main__":
    mcp.run()
