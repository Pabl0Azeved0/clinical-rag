"""Answer a clinical question using the PydanticAI agent layer over RAG.

Usage:
    python scripts/agent_ask.py "what is the A1C test" [--k 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinical_rag.agent.clinical_agent import AgentDeps, build_agent, run_agent
from clinical_rag.config import get_settings
from clinical_rag.embeddings.embedder import Embedder
from clinical_rag.generation.generator import DISCLAIMER
from clinical_rag.retrieval.retriever import Retriever
from clinical_rag.vectorstore.chroma_store import ChromaStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Answer a clinical question using the PydanticAI agent."
    )
    parser.add_argument("question", help="Question to answer")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of chunks to retrieve (default: top_k from config)",
    )
    args = parser.parse_args()

    cfg = get_settings()

    from clinical_rag.observability import setup_tracing

    setup_tracing(cfg)

    embedder = Embedder(cfg.embedding_model_name)
    store = ChromaStore(cfg.resolve(cfg.chroma_dir), cfg.collection_name, embedder)
    retriever = Retriever(store, cfg.top_k if args.k is None else args.k)

    deps = AgentDeps(retriever=retriever, settings=cfg)
    agent = build_agent(cfg)

    answer, _ = run_agent(agent, args.question, deps)

    print(answer.answer)
    print()
    if answer.citations:
        print("Citations:")
        for c in answer.citations:
            print(f"  [{c.index}] {c.title} — {c.url}")
        print()
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
