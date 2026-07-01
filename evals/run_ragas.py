"""LLM-judge end-state eval (RAGAS) — faithfulness, answer relevancy, context recall/precision.

Deliberately kept opt-in and separate from the deterministic retrieval eval:

  * RAGAS metrics need a *capable* judge LLM. With llama3.2:3b on CPU each metric makes
    several judge calls per question, so a full run is slow and the small model is a weak
    judge. Point the judge at a stronger local model or an API provider for real numbers.
  * It also pulls extra deps (see evals/requirements-ragas.txt), so it is not part of the
    base install and is NOT run in the default CPU environment.

Setup:
    .venv/bin/pip install -r evals/requirements-ragas.txt
    # judge model comes from LLM_MODEL / LLM_BASE_URL (Ollama) unless overridden below.

Run (start small — sample defaults to 5):
    .venv/bin/python evals/run_ragas.py --sample 5

The dataset (question, generated answer, retrieved contexts, ground-truth URL) is built
with THIS project's agent + retriever, so that half is real; only the ragas.evaluate call
depends on the optional deps and the chosen judge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


def build_samples(golden: list[dict], k: int) -> list[dict]:
    """Run the agent over the golden questions to build RAGAS input rows."""
    from clinical_rag.agent.clinical_agent import AgentDeps, build_agent, run_agent
    from clinical_rag.config import get_settings
    from clinical_rag.embeddings.embedder import Embedder
    from clinical_rag.retrieval.retriever import Retriever
    from clinical_rag.vectorstore.chroma_store import ChromaStore

    cfg = get_settings()
    store = ChromaStore(
        cfg.resolve(cfg.chroma_dir),
        cfg.collection_name,
        Embedder(cfg.embedding_model_name),
    )
    agent = build_agent(cfg)

    samples = []
    for item in golden:
        deps = AgentDeps(retriever=Retriever(store, k), settings=cfg)
        answer, _ = run_agent(agent, item["question"], deps)
        samples.append(
            {
                "user_input": item["question"],
                "response": answer.answer,
                "retrieved_contexts": [c["text"] for c in deps.retrieved.values()],
                "reference": item["expected_source_url"],
            }
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS LLM-judge eval (opt-in).")
    parser.add_argument("--sample", type=int, default=5, help="Number of golden questions to judge")
    parser.add_argument("--k", type=int, default=5, help="Passages to retrieve per question")
    args = parser.parse_args()

    try:
        from langchain_ollama import ChatOllama
        from ragas import EvaluationDataset, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            ResponseRelevancy,
        )
    except ImportError as exc:  # pragma: no cover - optional deps
        sys.exit(
            f"RAGAS deps missing ({exc}). Install them first:\n"
            "  .venv/bin/pip install -r evals/requirements-ragas.txt"
        )

    from clinical_rag.config import get_settings

    cfg = get_settings()
    golden = yaml.safe_load((_ROOT / "evals" / "golden_set.yaml").read_text())[: args.sample]

    print(f"Building {len(golden)} samples with the agent (this is the slow part)…")
    samples = build_samples(golden, args.k)

    judge = LangchainLLMWrapper(ChatOllama(model=cfg.llm_model, base_url=cfg.llm_base_url))
    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference()],
        llm=judge,
    )
    print("\nRAGAS results:")
    print(result)


if __name__ == "__main__":
    main()
