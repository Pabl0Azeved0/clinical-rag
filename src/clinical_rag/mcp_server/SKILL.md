# Skill: clinical-evidence retrieval (MCP)

A narrow MCP server that searches indexed **MedlinePlus** health topics and
returns cited source passages. It is a retrieval microservice, not a doctor.

## Tool

`search_clinical_evidence(query: str, k: int = 5) -> list[dict]`

Returns up to `k` passages, each: `{text, title, url, doc_id, distance}`
(cosine distance, lower = closer).

## When to use

- To fetch **evidence passages with citable sources** (title + URL) for a
  clinical/health question, so an answer can be grounded in and cite them.
- As a shared retrieval backend for any agent that must not invent medical facts.

## When NOT to use

- **Not for diagnosis or treatment advice** — it only returns passages; it never
  interprets them for a specific person.
- **Not for PHI / patient data** — do not pass personal health information in the
  query; the service neither expects nor stores it.
- Not a general web search — it only covers the ingested MedlinePlus corpus.

## Notes

Answer generation, `[n]` numbering, and citation-grounding are the *caller's*
responsibility (in clinical-rag they live in the agent + grounding validator).
This server intentionally stays deterministic and side-effect free.
