# clinical-rag

An **agentic** clinical-evidence assistant. It answers health questions using only
retrieved, **cited** passages from public health sources (MedlinePlus) — and refuses,
rather than guesses, when the evidence isn't there.

> **Not medical advice.** This tool retrieves and surfaces cited passages from public
> health documents. It does not diagnose, prescribe, or provide clinical guidance.
> Always consult a qualified healthcare professional.

Built on a deterministic RAG pipeline wrapped in a **PydanticAI** agent, with three
layers of evaluation (unit, trajectory, retrieval) and a human-in-the-loop gate.

---

## The headline decision: what is deterministic vs. what is agentic

The central design choice is *where the LLM is allowed to make decisions* — because in a
clinical context, **a fabricated citation is worse than "no evidence found."**

**Stays deterministic (never behind the LLM):**
- query embedding & vector search
- citation extraction and `[n]` numbering (one source of truth per run)
- the "no chunks ⇒ no-evidence answer" guard
- the not-medical-advice disclaimer (appended *outside* the agent — it can never depend
  on the model remembering it)
- `grounded` / `confidence` scoring (from citation coverage and retrieval distance —
  not the model's self-report)

**Becomes agentic (the LLM decides):**
- whether to call `search_evidence`, and with what query
- whether to broaden the search once when results are weak
- how to phrase a grounded answer from the retrieved passages

The guardrails around the agent are what make it trustworthy, not the model's goodwill.

---

## Architecture

```
                    MedlinePlus XML  (1,015 English topics)
                          │  ingest → chunk → embed (all-MiniLM-L6-v2, local CPU)
                          ▼
                    ChromaDB  (local, cosine)  ◄───────────────┐
                          ▲                                     │
       retrieval-only     │  search_clinical_evidence           │
       (no gen, no PHI)   │  exposed as an MCP server ──────────┘
                          │
   user ─►  PydanticAI Agent
              │  tools:  search_evidence  ·  broaden_search (≤ 1×, only if weak)
              │  output: ClinicalAnswer  (PromptedOutput: works with small local models)
              ▼
        grounding validator  (deterministic, R7)
          • every cited [n] must exist in the retrieved set → else ModelRetry / refuse
          • grounded = citation coverage ;  confidence = f(retrieval distance)
          • one gentle "cite what you retrieved" nudge, then accept
              ▼
        HITL gate  ──  not grounded OR confidence < 0.6  ──►  human approve / edit / reject
              ▼
        answer + visible citations + mandatory disclaimer  (Streamlit UI)
```

Retrieval is available in-process **or** over an **MCP** server (`retrieval_transport`
flag) — the same numbering/grounding stays in the agent either way. Runs are traceable
via **Logfire** (`ENABLE_TRACING=1`): tool-call spans, LLM calls, tokens, latency.

---

## Evaluation (the point of the project)

Three independent layers, so a change that helps one thing can't silently break another:

| Layer | What it checks | Result |
|---|---|---|
| **Unit** | pipeline + guardrails (validator, broaden cap, HITL, MCP, degradation) | **53 passing** |
| **Trajectory** | the agent's *tool-call trace* per scenario (mocked model) | **5/5** scenarios |
| **Retrieval** | does the expected topic get retrieved, over a 36-question golden set | **recall@1 0.81 · recall@3 1.00 · MRR 0.90** |

**Trajectory scenarios** (assert on the tool-call sequence + guardrails, via PydanticAI's
`FunctionModel`): happy path · weak→broaden-once · **refusal with no fabricated citation**
· prompt-injection can't smuggle an ungrounded claim · famous-fact-not-in-corpus says "no
evidence" instead of answering from memory.

**Retrieval numbers** (`make eval`): the expected MedlinePlus topic lands in the top 3 for
**every** question, and is the #1 result 81% of the time.

**LLM-judge metrics (RAGAS)** are wired in `evals/run_ragas.py` (faithfulness, response
relevancy, context precision) using the *pluggable* provider as the judge — but they need a
capable judge to be meaningful, so they're opt-in (`evals/requirements-ragas.txt`) and not
run on the default CPU/3B setup. See *Honest limitations* below.

---

## Pluggable LLM

Provider-agnostic by design (portability, cost control, health-data privacy), selected via
`LLM_PROVIDER` / `LLM_MODEL` — no hardcoded model strings, no code change to swap. Ollama and
Groq both speak the OpenAI-compatible API, so the same model class is reused with a different
base URL + key.

| Mode | Provider | Notes |
|---|---|---|
| **API (recommended)** | Groq · `llama-3.3-70b-versatile` | fast, grounds reliably; free tier is rate-limited |
| Local / offline | Ollama · `llama3.2:3b` | no data leaves the machine, but grounding is poor (below) |
| Deploy | Groq key via Streamlit secrets | Cloud can't run Ollama |

### Why a capable model matters — real A/B (5 golden questions)

The agent asks a lot of the model: call a tool, then return a *validated, typed* answer with
inline citations. Small local models can't keep up:

| Model | Grounded | Latency | |
|---|---|---|---|
| **Groq `llama-3.3-70b`** | **80% (4/5)** | **~13s** (2–4s typical) | ✅ default |
| Groq `llama-3.1-8b-instant` | — | — | too weak: emits malformed tool calls |
| local `llama3.2:3b` | 0% (0/5) | ~372s (≈6 min) | slow; rarely cites |
| local `qwen2.5:3b` | 0% (0/3) | ~35s | fast, but never calls the search tool |

The abstraction is the payoff: an `.env` change from Ollama to Groq — no code — moved grounding
**0% → 80%** and latency **minutes → seconds**.

### Output mode is capability-based

- **Small local models** → **`PromptedOutput`** (prompt for JSON text, then parse): they skip
  the tool under native JSON-schema mode and emit prose under tool-output mode.
- **Capable API models** → default **tool output**. (Groq actively *rejects* PromptedOutput —
  *"json mode cannot be combined with tool/function calling"*.)

A model that exhausts its retry budget (or a transient provider error) degrades to an honest
`grounded=False` refusal — it never crashes and never fabricates a citation.

---

## Tech stack

- **Agent:** PydanticAI (`pydantic-ai-slim`) · typed `ClinicalAnswer` output · output validator
- **Retrieval-as-a-service:** FastMCP server exposing one retrieval-only tool
- **Embeddings:** `sentence-transformers` `all-MiniLM-L6-v2` (local, CPU)
- **Vector store:** ChromaDB (local `PersistentClient`, cosine)
- **Data:** MedlinePlus Health Topics bulk XML (public domain, ~1,015 English topics)
- **UI:** Streamlit (citations, trajectory panel, HITL approval)
- **Observability:** Logfire / OpenTelemetry (opt-in)
- **Tests/evals:** pytest · golden-set retrieval eval · RAGAS (opt-in)

No LangChain/LlamaIndex in the core pipeline — every layer is transparent and testable.

---

## Quickstart

```bash
make install                     # venv + deps
make ingest                      # download MedlinePlus, build the vector store (~1,015 topics)
ollama pull llama3.2:3b          # local model for generation

make ui                          # Streamlit demo at http://localhost:8501
make eval                        # retrieval eval (recall@k, MRR) over the golden set
make test                        # full test suite

# one-shot CLI:
.venv/bin/python scripts/agent_ask.py "what is the A1C test"
ENABLE_TRACING=1 .venv/bin/python scripts/agent_ask.py "..."   # with Logfire trace
```

Key env vars (all optional): `LLM_PROVIDER`, `LLM_MODEL`, `RETRIEVAL_TRANSPORT`
(`in_process`|`mcp`), `APPROVAL_CONFIDENCE_THRESHOLD`, `ENABLE_TRACING`, `LOGFIRE_TOKEN`.

---

## Engineering decisions & trade-offs

- **Deterministic disclaimer & grounding.** Safety-critical text and the grounded/confidence
  signals are computed in code, never left to the model — a model that "forgets" the
  disclaimer is a compliance failure, not a quality blip.
- **`PromptedOutput` over tool/native output.** `llama3.2:3b` skips tools under native
  JSON-schema mode and emits prose under tool-output mode; prompted JSON keeps tool-calling
  intact while still yielding a parseable typed answer.
- **Extend the retriever, don't bypass it, for MCP.** The agent's tools call the retriever
  (in-process or via MCP) so numbering/grounding stay in one place — reranking and grounding
  keep a single home.
- **Graceful degradation.** If a weak model exhausts its retry budget, the run degrades to an
  honest `grounded=False` refusal instead of crashing — and never emits a fabricated citation.
- **ChromaDB over FAISS:** persistent client + metadata filtering out of the box simplifies
  citation tracking on a single machine.

---

## Honest limitations

- **Grounding is model-bound.** On Groq's 70B the grounded rate is ~80%; on a local 3B it is
  ~0% (the small model rarely emits the inline `[n]`, and the validator then correctly reports
  `grounded=False`). The system never crashes and never fabricates regardless — but for real
  grounded answers use a capable provider; local Ollama is an *offline/degraded* fallback. This
  model dependency is also why trustworthy RAGAS (LLM-judge) numbers need a capable judge.
- **Groq free tier is rate-limited.** Most queries return in a few seconds, but the free tier
  occasionally throttles (a ~20–50s spike) or rejects a malformed tool call (auto-retried). A
  paid key smooths this out.
- **Coverage:** MedlinePlus Health Topics only (~1,015 English topics). No journals or PDFs.
- **Retrieval** depends on query–topic vocabulary overlap; no query expansion beyond the
  single bounded `broaden_search`.
- Not for clinical decision support. Not HIPAA-assessed.
