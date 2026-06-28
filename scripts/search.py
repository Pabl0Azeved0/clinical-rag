"""Search the ChromaDB collection with a natural-language query.

Usage:
    python scripts/search.py "what is the A1C test" [--k 5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinical_rag.config import get_settings
from clinical_rag.embeddings.embedder import Embedder
from clinical_rag.retrieval.retriever import Retriever
from clinical_rag.vectorstore.chroma_store import ChromaStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Search MedlinePlus topics in ChromaDB.")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--k", type=int, default=None, help="Number of results (default: top_k from config)")
    args = parser.parse_args()

    cfg = get_settings()
    k = args.k if args.k is not None else cfg.top_k

    embedder = Embedder(cfg.embedding_model_name)
    store = ChromaStore(cfg.resolve(cfg.chroma_dir), cfg.collection_name, embedder)
    retriever = Retriever(store, k)

    results = retriever.retrieve(args.query)

    if not results:
        print("No results found. Have you run `make ingest` yet?")
        return

    print(f"\nTop {len(results)} results for: \"{args.query}\"\n")
    for i, r in enumerate(results, 1):
        snippet = r["text"][:200].replace("\n", " ")
        print(f"[{r['distance']:.4f}] {r['title']} — {r['url']}")
        print(f"  {snippet}...")
        print()


if __name__ == "__main__":
    main()
