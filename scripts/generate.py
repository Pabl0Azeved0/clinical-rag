"""Generate a cited answer from MedlinePlus indexed content.

Usage:
    python scripts/generate.py "what are the symptoms of diabetes" [--k 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinical_rag.config import get_settings
from clinical_rag.embeddings.embedder import Embedder
from clinical_rag.generation.generator import Generator
from clinical_rag.llm.provider import get_provider
from clinical_rag.retrieval.retriever import Retriever
from clinical_rag.vectorstore.chroma_store import ChromaStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Answer a clinical question using RAG."
    )
    parser.add_argument("query", help="Question to answer")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of chunks to retrieve (default: top_k from config)",
    )
    args = parser.parse_args()

    cfg = get_settings()
    k = args.k if args.k is not None else cfg.top_k

    embedder = Embedder(cfg.embedding_model_name)
    store = ChromaStore(cfg.resolve(cfg.chroma_dir), cfg.collection_name, embedder)
    retriever = Retriever(store, k)
    chunks = retriever.retrieve(args.query)

    gen = Generator(get_provider(cfg))
    answer = gen.generate_answer(args.query, chunks)

    print(answer.text)
    print()
    print("Sources:")
    for s in answer.sources:
        print(f"  [{s['index']}] {s['title']} — {s['url']}")


if __name__ == "__main__":
    main()
