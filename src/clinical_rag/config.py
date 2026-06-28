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
    top_k: int = 5

    # LLM (pluggable — implemented in Phase 2)
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama3.2:3b"))
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434")
    )
    llm_api_key: str | None = field(default_factory=lambda: os.getenv("LLM_API_KEY"))

    def resolve(self, path: str) -> Path:
        """Return an absolute Path, resolving relative paths against the repo root."""
        p = Path(path)
        return p if p.is_absolute() else _REPO_ROOT / p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
