# clinical-rag

An **agentic clinical-evidence assistant** that answers health questions using only
retrieved, **cited** passages from public health sources — and **refuses, rather than
guesses, when the evidence isn't there.**

Built on a deterministic RAG pipeline wrapped in a **PydanticAI** agent, with a
human-in-the-loop gate and three independent layers of evaluation.

> **Not medical advice.** This tool surfaces cited passages from public health documents.
> It does not diagnose, prescribe, or provide clinical guidance. Always consult a qualified
> healthcare professional.

---

## Demo

![clinical-rag demo](docs/clinical_rag.gif)

*Ask a clinical question → the agent retrieves evidence, answers with inline `[n]`
citations, and shows its tool-call trajectory. Low-confidence answers are held for human
review before they're shown.*

Try it → https://clinical-rag-dtazjxhcmsgkbmxhhafzku.streamlit.app/

---

## Why this project is different

Most RAG demos are "chat with a PDF." This one is built around the constraint that matters
in a clinical context: **a fabricated citation is worse than "no evidence found."**

Everything safety-critical is **deterministic** — computed in code, never left to the
model's goodwill:

| Deterministic (never behind the LLM) | Agentic (the LLM decides) |
|---|---|
| query embedding & vector search | whether to call `search_evidence`, and with what query |
| citation extraction & `[n]` numbering | whether to broaden the search once when results are weak |
| the "no chunks ⇒ no-evidence" guard | how to phrase a grounded answer from retrieved passages |
| the not-medical-advice disclaimer (appended *outside* the agent) | |
| `grounded` / `confidence` scoring (from citation coverage & retrieval distance) | |

The guardrails around the agent are what make it trustworthy — not the model.

---

## Evaluation (the point of the project)

Three independent layers, so a change that helps one thing can't silently break another:

| Layer | What it checks | Result |
|---|---|---|
| **Unit** | pipeline + guardrails (validator, broaden cap, HITL, MCP, degradation) | **55 passing** |
| **Trajectory** | the agent's *tool-call trace* per scenario (mocked model) | **5 / 5 scenarios** |
| **Retrieval** | does the expected topic get retrieved, over a 36-question golden set | **recall@1 0.81 · recall@3 1.00 · MRR 0.90** |

**Trajectory scenarios** assert on the tool-call sequence and guardrails (via PydanticAI's
`FunctionModel`): happy path · weak → broaden-once · refusal with no fabricated citation ·
prompt-injection can't smuggle an ungrounded claim · a famous fact *not in the corpus*
returns "no evidence" instead of answering from memory.

**Retrieval** (`make eval`): the expected MedlinePlus topic lands in the top 3 for *every*
question, and is the #1 result 81% of the time.

---

## The pluggable-LLM payoff (real A/B, 5 golden questions)

The agent asks a lot of the model: call a tool, then return a *validated, typed* answer with
inline citations. Provider is swappable via `.env` — no code change. That abstraction is the
payoff:

| Model | Grounded | Latency | |
|---|---|---|---|
| **Groq `llama-3.3-70b-versatile`** | **80% (4/5)** | **~13s** (2–4s typical) | ✅ default |
| Groq `llama-3.1-8b-instant` | — | — | too weak: malformed tool calls |
| local `llama3.2:3b` (Ollama) | 0% (0/5) | ~372s (≈6 min) | offline fallback; rarely cites |
| local `qwen2.5:3b` (Ollama) | 0% (0/3) | ~35s | fast, but never calls the search tool |

An `.env` change from local Ollama to Groq — no code — moved grounding **0% → 80%** and
latency **minutes → seconds**. The system never crashes and never fabricates a citation on
*any* of these; a weak model simply degrades to an honest `grounded=False` refusal.

---

## Architecture

```
                    MedlinePlus XML  (1,015 English topics)
                          │  ingest → chunk → embed (all-MiniLM-L6-v2, local CPU)
                          ▼
                    ChromaDB  (local, cosine)  ◄───────────────┐
                          ▲                                     │
       retrieval-only     │  search_clinical_evidence          │
       (no gen, no PHI)   │  exposed as an MCP server ──────────┘
                          │
   user ─►  PydanticAI Agent
              │  tools:  search_evidence · broaden_search (≤ 1×, only if weak)
              │  output: ClinicalAnswer  (typed; PromptedOutput for small local models)
              ▼
        grounding validator  (deterministic)
          • every cited [n] must exist in the retrieved set → else ModelRetry / refuse
          • grounded = citation coverage ;  confidence = f(retrieval distance)
          ▼
        HITL gate  ──  not grounded OR confidence < 0.6  ──►  human approve / edit / reject
          ▼
        answer + visible citations + mandatory disclaimer  (Streamlit UI)
```

Retrieval runs **in-process or over an MCP server** (`retrieval_transport` flag) — the same
numbering/grounding stays in the agent either way. Runs are traceable via **Logfire**
(`ENABLE_TRACING=1`): tool-call spans, LLM calls, tokens, latency.

---

## Quickstart

```bash
make install                     # venv + deps
make ingest                      # download MedlinePlus, build the vector store (~1,015 topics)
ollama pull llama3.2:3b          # optional: local model for offline generation

make ui                          # Streamlit demo at http://localhost:8501
make eval                        # retrieval eval (recall@k, MRR) over the golden set
make test                        # full test suite

# one-shot CLI:
.venv/bin/python scripts/agent_ask.py "what is the A1C test"
ENABLE_TRACING=1 .venv/bin/python scripts/agent_ask.py "..."   # with Logfire trace
```

**Recommended:** set `LLM_PROVIDER=groq` + a Groq key (free tier) for reliable grounding.
Local Ollama works offline but is a degraded fallback (see the A/B table above).

Key env vars (all optional): `LLM_PROVIDER`, `LLM_MODEL`, `RETRIEVAL_TRANSPORT`
(`in_process`|`mcp`), `APPROVAL_CONFIDENCE_THRESHOLD`, `ENABLE_TRACING`, `LOGFIRE_TOKEN`.

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

## Engineering decisions & trade-offs

- **Deterministic disclaimer & grounding.** Safety-critical text and the grounded/confidence
  signals are computed in code, never left to the model — a model that "forgets" the
  disclaimer is a compliance failure, not a quality blip.
- **`PromptedOutput` over native/tool output for small models.** `llama3.2:3b` skips tools
  under native JSON-schema mode and emits prose under tool-output mode; prompted JSON keeps
  tool-calling intact while still yielding a parseable typed answer. (Capable API models like
  Groq use default tool output — Groq actively *rejects* PromptedOutput combined with tools.)
- **Extend the retriever, don't bypass it, for MCP.** The agent's tools call the retriever
  (in-process or via MCP) so numbering and grounding keep a single home.
- **Graceful degradation.** If a weak model exhausts its retry budget, the run degrades to an
  honest `grounded=False` refusal instead of crashing — and never emits a fabricated citation.
- **ChromaDB over FAISS.** A persistent client with metadata filtering out of the box
  simplifies citation tracking on a single machine.

---

## Honest limitations

- **Grounding is model-bound.** On Groq's 70B the grounded rate is ~80%; on a local 3B it's
  ~0% (the small model rarely emits the inline `[n]`, and the validator then correctly reports
  `grounded=False`). The system never crashes and never fabricates regardless — but for real
  grounded answers, use a capable provider; local Ollama is an *offline/degraded* fallback.
  This is also why trustworthy RAGAS (LLM-judge) numbers need a capable judge, so those are
  opt-in rather than run on the default setup.
- **Groq free tier is rate-limited.** Most queries return in seconds, but the free tier
  occasionally throttles (a ~20–50s spike) or rejects a malformed tool call (auto-retried).
- **Coverage:** MedlinePlus Health Topics only (~1,015 English topics). No journals or PDFs.
- **Retrieval** depends on query–topic vocabulary overlap; no query expansion beyond the
  single bounded `broaden_search`.
- **Not for clinical decision support. Not HIPAA-assessed.**
