from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present (no-op if missing)
load_dotenv()

# __file__ is src/clinical_rag/config.py → parents[0]=clinical_rag, parents[1]=src, parents[2]=repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    # Directories (resolved relative to repo root)
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    chroma_dir: str = "data/chroma"

    # Embedding
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # ChromaDB
    collection_name: str = "medline_topics"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Retrieval
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    # Broaden-fallback trigger: first search is "weak" if best distance exceeds this.
    weak_distance_threshold: float = field(
        default_factory=lambda: float(os.getenv("WEAK_DISTANCE_THRESHOLD", "0.45"))
    )

    # LLM (pluggable — implemented in Phase 2)
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "llama3.2:3b")
    )
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434")
    )
    llm_api_key: str | None = field(default_factory=lambda: os.getenv("LLM_API_KEY"))

    # Timeouts (seconds): per LLM call, and a wall-clock cap on a whole agent run so a
    # slow/looping local model can't hang the UI indefinitely.
    request_timeout: float = field(
        default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT", "60"))
    )
    agent_timeout: float = field(
        default_factory=lambda: float(os.getenv("AGENT_TIMEOUT", "180"))
    )

    # Retrieval transport for the agent: "in_process" (local Retriever) or "mcp"
    # (fetch passages from the standalone MCP server via a fastmcp client).
    retrieval_transport: str = field(
        default_factory=lambda: os.getenv("RETRIEVAL_TRANSPORT", "in_process")
    )
    # Optional MCP server URL; when unset the "mcp" transport uses an in-memory
    # (same-process) client against the imported server for offline/testing.
    mcp_server_url: str | None = field(
        default_factory=lambda: os.getenv("MCP_SERVER_URL")
    )

    # HITL gate: answers with grounded=False OR confidence below this pause for human
    # approval before being surfaced.
    approval_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("APPROVAL_CONFIDENCE_THRESHOLD", "0.6"))
    )

    # Observability (Logfire/OpenTelemetry). Off by default; console-only unless a token is set.
    enable_tracing: bool = field(
        default_factory=lambda: os.getenv("ENABLE_TRACING", "false").lower()
        in ("1", "true", "yes")
    )
    logfire_token: str | None = field(
        default_factory=lambda: os.getenv("LOGFIRE_TOKEN")
    )

    def resolve(self, path: str) -> Path:
        """Return an absolute Path, resolving relative paths against the repo root."""
        p = Path(path)
        return p if p.is_absolute() else _REPO_ROOT / p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
