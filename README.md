# clinical-rag

A RAG-based clinical Q&A system that retrieves cited evidence from public health sources.

> **Not medical advice.** This tool retrieves and surfaces cited passages from public health
> documents (MedlinePlus). It does not diagnose, prescribe, or provide clinical guidance.
> Always consult a qualified healthcare professional.

**Status: Phase 1 — scaffold + ingestion + retrieval**

---

## The Problem

Clinicians and researchers face an overwhelming volume of clinical documentation. Finding
reliable, cited evidence quickly — without hallucination — is a genuine bottleneck.
`clinical-rag` addresses this by building a retrieval pipeline anchored exclusively in
authoritative public health sources, with every answer citing its origin.

---

## Architecture

```
MedlinePlus XML
      │
      ▼
 [Ingestion]  parse + clean HTML
      │
      ▼
 [Chunking]   paragraph-aware, overlap
      │
      ▼
 [Embeddings] all-MiniLM-L6-v2 (local CPU)
      │
      ▼
 [ChromaDB]   persistent local vector store
      │
      ▼
 [Retrieval]  cosine similarity top-k
      │
      ▼
 [LLM Gen]    pluggable (see below) — Phase 2
      │
      ▼
 [Streamlit]  cited Q&A UI — Phase 2
```

### Pluggable LLM Generation (Phase 2)

The generation layer is intentionally provider-agnostic — a key engineering decision for
portability, cost control, and health-data privacy:

| Mode | Provider | When to use |
|---|---|---|
| Local | Ollama + Llama 3.2:3b | Default dev; no data leaves the machine |
| API key | Google Gemini / Groq (free tier) | Better quality with user's own key |
| Deploy | API key via Streamlit secrets | Streamlit Community Cloud (no Ollama) |

Selected via `LLM_PROVIDER` env var. No hardcoding.

---

## Tech Stack

- **Embeddings:** `sentence-transformers` — `all-MiniLM-L6-v2` (local, CPU, no API cost)
- **Vector store:** ChromaDB (local `PersistentClient`, no server required)
- **Data source:** MedlinePlus Health Topics bulk XML (public domain)
- **Chunking:** custom paragraph-aware character chunking (no LangChain/LlamaIndex)
- **Evals:** golden set YAML + RAGAS (Phase 3)
- **UI:** Streamlit (Phase 2)

---

## Quickstart

```bash
# 1. Install
make install

# 2. Ingest MedlinePlus (downloads ~10 MB XML, builds vector store)
make ingest

# 3. Search
python scripts/search.py "what is the A1C test" --k 3

# Quick smoke test (30 topics only)
.venv/bin/python scripts/ingest.py --limit 30
```

### Environment variables (optional)

Copy `.env.example` to `.env`. All LLM vars are optional — generation is Phase 2.

---

## Engineering Decisions & Trade-offs

**Why ChromaDB over FAISS?** ChromaDB offers a persistent client with metadata
filtering out of the box, which simplifies citation tracking. FAISS is faster at scale
but requires manual metadata management. For a portfolio project on a single machine,
ChromaDB's simplicity wins.

**Why all-MiniLM-L6-v2?** 22M params, 384-dim embeddings, fast on CPU. Quality is
sufficient for topic-level retrieval from clean MedlinePlus prose. Can swap for
`all-mpnet-base-v2` for higher quality at ~3× cost.

**Why no LangChain/LlamaIndex?** This is a portfolio project. Custom pipeline code
shows architectural understanding; framework code hides it. Every layer is transparent
and testable.

**Chunking strategy:** Paragraph-aware with 150-char overlap. MedlinePlus summaries are
already well-structured prose — paragraph splitting preserves semantic units better than
naive fixed-size splitting.

---

## Limitations

- Only covers MedlinePlus Health Topics (~1,000 English topics). No journals, no PDFs.
- Retrieval quality depends on query–topic vocabulary overlap; no query expansion yet.
- LLM generation (Phase 2) can still hallucinate — all answers must be grounded in
  retrieved context (enforced via prompt constraints).
- Not for clinical decision support. Not HIPAA-assessed.
