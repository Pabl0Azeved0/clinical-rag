# Evals

## Eval-driven development plan

Every change to chunking strategy, embedding model, or retrieval parameters must be
measured against `golden_set.yaml` before merging. This keeps improvements objective.

## Golden set schema

```yaml
- question:            "Natural-language clinical question"
  expected_source_url: "https://medlineplus.gov/<topic>.html"
  notes:               "Optional rationale"
```

The golden set lives in `golden_set.yaml`. Target: 30–50 questions covering diverse
clinical topics (diabetes, cardiovascular, mental health, infectious disease, etc.).

## Metrics (Phase 3 — RAGAS)

| Metric | What it measures |
|---|---|
| Context Recall | Did retrieval surface the expected source document? |
| Context Precision | Were retrieved chunks relevant, or noisy? |
| Faithfulness | Is the generated answer grounded in retrieved context? |
| Answer Relevance | Does the answer address the question? |

RAGAS will be wired up in Phase 3 once LLM generation (Phase 2) is in place.
Until then, retrieval is evaluated manually against `expected_source_url`.
