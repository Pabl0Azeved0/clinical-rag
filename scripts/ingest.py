"""Ingest MedlinePlus health topics into ChromaDB.

Usage:
    python scripts/ingest.py [--limit N] [--rebuild]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on the path when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinical_rag.chunking.chunker import chunk_documents
from clinical_rag.config import get_settings
from clinical_rag.embeddings.embedder import Embedder
from clinical_rag.ingestion.medline import download_topics, parse_topics
from clinical_rag.vectorstore.chroma_store import ChromaStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest MedlinePlus topics into ChromaDB.")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of topics (for quick test)")
    parser.add_argument("--rebuild", action="store_true", help="Delete and recreate the collection")
    args = parser.parse_args()

    cfg = get_settings()

    raw_dir = cfg.resolve(cfg.raw_dir)
    chroma_dir = cfg.resolve(cfg.chroma_dir)

    # Download + parse
    xml_path = download_topics(raw_dir)
    print(f"Parsing topics from {xml_path} ...")
    docs = parse_topics(xml_path)
    print(f"Parsed {len(docs)} English topics with summaries.")

    if args.limit:
        docs = docs[: args.limit]
        print(f"Limited to {len(docs)} topics.")

    # Chunk
    chunks = chunk_documents(docs, cfg.chunk_size, cfg.chunk_overlap)
    print(f"Created {len(chunks)} chunks (chunk_size={cfg.chunk_size}, overlap={cfg.chunk_overlap}).")

    # Embed + store
    embedder = Embedder(cfg.embedding_model_name)
    print(f"Loading embedding model '{cfg.embedding_model_name}' ...")

    if args.rebuild:
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_dir))
        try:
            client.delete_collection(cfg.collection_name)
            print(f"Deleted existing collection '{cfg.collection_name}'.")
        except Exception:
            pass

    store = ChromaStore(chroma_dir, cfg.collection_name, embedder)
    print(f"Upserting {len(chunks)} chunks into ChromaDB collection '{cfg.collection_name}' ...")
    store.add_chunks(chunks)

    print(f"\nDone. Ingested {len(docs)} topics → {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
