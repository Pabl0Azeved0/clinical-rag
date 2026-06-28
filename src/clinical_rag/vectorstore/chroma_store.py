from __future__ import annotations

from pathlib import Path

import chromadb

from clinical_rag.domain.models import Chunk
from clinical_rag.embeddings.embedder import Embedder


class ChromaStore:
    def __init__(self, persist_dir: Path, collection_name: str, embedder: Embedder) -> None:
        self._embedder = embedder
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        # get_or_create so repeated runs accumulate / upsert safely
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            # No embedding_function — we supply embeddings explicitly
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 256) -> None:
        """Upsert chunks in batches, computing embeddings via the embedder."""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self._embedder.embed_texts(texts)
            self._collection.upsert(
                ids=[c.id for c in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[
                    {
                        "title": c.title,
                        "url": c.url,
                        "doc_id": c.doc_id,
                        "chunk_index": c.chunk_index,
                    }
                    for c in batch
                ],
            )

    def query(self, text: str, k: int) -> list[dict]:
        """Return top-k results as dicts with keys: text, title, url, doc_id, distance."""
        embedding = self._embedder.embed_query(text)
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append(
                {
                    "text": doc,
                    "title": meta["title"],
                    "url": meta["url"],
                    "doc_id": meta["doc_id"],
                    "distance": dist,
                }
            )
        return output
