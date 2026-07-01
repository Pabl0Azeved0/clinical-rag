"""Tests for the retrieval-eval metric functions — no model, no network."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from run_retrieval_eval import evaluate, rank_of, unique_urls


def _r(url: str) -> dict:
    return {"url": url, "title": url, "text": "", "distance": 0.1}


def test_unique_urls_dedupes_in_order():
    results = [_r("a"), _r("a"), _r("b"), _r("a"), _r("c")]
    assert unique_urls(results) == ["a", "b", "c"]


def test_rank_of_found_and_missing():
    results = [_r("a"), _r("a"), _r("b"), _r("c")]
    assert rank_of("a", results) == 1
    assert rank_of("b", results) == 2  # topic-level rank, dedup ignores repeats
    assert rank_of("z", results) is None


def test_evaluate_metrics():
    golden = [
        {"question": "q1", "expected_source_url": "a"},  # rank 1
        {"question": "q2", "expected_source_url": "b"},  # rank 2
        {"question": "q3", "expected_source_url": "z"},  # miss
    ]
    fake = {
        "q1": [_r("a"), _r("b")],
        "q2": [_r("a"), _r("b")],
        "q3": [_r("a"), _r("b")],
    }

    def retrieve(query, fetch):
        return fake[query]

    rows, metrics = evaluate(golden, retrieve, fetch=5)

    assert metrics["recall@1"] == 1 / 3  # only q1's expected is at rank 1
    assert metrics["recall@3"] == 2 / 3  # q1 + q2 within top 3
    assert metrics["mrr"] == (1.0 + 0.5 + 0.0) / 3
