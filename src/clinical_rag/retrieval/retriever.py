from __future__ import annotations

from clinical_rag.vectorstore.chroma_store import ChromaStore


class Retriever:
    def __init__(self, store: ChromaStore, top_k: int) -> None:
        self._store = store
        self._top_k = top_k

    def retrieve(self, query: str) -> list[dict]:
        return self._store.query(query, self._top_k)
