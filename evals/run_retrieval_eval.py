"""Deterministic retrieval eval: recall@k and MRR against the golden set.

No LLM judge — this measures whether retrieval surfaces the expected MedlinePlus
topic for each question, which is fast, reproducible, and the honest end-state
signal for the retrieval layer. (LLM-judge generation metrics live in
run_ragas.py, which needs a capable judge model.)

Run:  make eval   (or  .venv/bin/python evals/run_retrieval_eval.py [--show-all])
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


def unique_urls(results: list[dict]) -> list[str]:
    """Topic-level ranking: unique source URLs in retrieval order."""
    seen: list[str] = []
    for r in results:
        if r["url"] not in seen:
            seen.append(r["url"])
    return seen


def rank_of(expected_url: str, results: list[dict]) -> int | None:
    """1-based rank of the expected topic in the retrieved results, or None."""
    urls = unique_urls(results)
    return urls.index(expected_url) + 1 if expected_url in urls else None


def evaluate(
    golden: list[dict], retrieve, fetch: int = 20, ks=(1, 3, 5, 10)
) -> tuple[list[dict], dict]:
    """`retrieve(query, fetch) -> list[dict]`. Returns (per-question rows, metrics)."""
    rows = []
    for item in golden:
        results = retrieve(item["question"], fetch)
        rows.append(
            {
                "question": item["question"],
                "expected": item["expected_source_url"],
                "rank": rank_of(item["expected_source_url"], results),
            }
        )
    n = len(rows) or 1
    metrics = {
        f"recall@{k}": sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n
        for k in ks
    }
    metrics["mrr"] = sum(1.0 / r["rank"] for r in rows if r["rank"]) / n
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval eval against the golden set."
    )
    parser.add_argument(
        "--show-all", action="store_true", help="Print every question's rank"
    )
    parser.add_argument(
        "--fetch", type=int, default=20, help="Chunks to fetch per query"
    )
    args = parser.parse_args()

    from clinical_rag.config import get_settings
    from clinical_rag.embeddings.embedder import Embedder
    from clinical_rag.retrieval.retriever import Retriever
    from clinical_rag.vectorstore.chroma_store import ChromaStore

    cfg = get_settings()
    golden = yaml.safe_load((_ROOT / "evals" / "golden_set.yaml").read_text())
    store = ChromaStore(
        cfg.resolve(cfg.chroma_dir),
        cfg.collection_name,
        Embedder(cfg.embedding_model_name),
    )
    retriever = Retriever(store, args.fetch)

    rows, metrics = evaluate(golden, retriever.retrieve, fetch=args.fetch)

    print(f"\nRetrieval eval over {len(rows)} questions (fetch={args.fetch}):\n")
    for name, val in metrics.items():
        print(f"  {name:<10} {val:.3f}")

    misses = [r for r in rows if not r["rank"] or r["rank"] > 3]
    if misses:
        print(f"\nQuestions where the expected topic ranked > 3 ({len(misses)}):")
        for r in misses:
            rank = r["rank"] if r["rank"] else "—"
            print(f"  rank {rank}: {r['question']}")

    if args.show_all:
        print("\nAll ranks:")
        for r in rows:
            print(f"  rank {r['rank'] if r['rank'] else '—':>3}  {r['question']}")


if __name__ == "__main__":
    main()
