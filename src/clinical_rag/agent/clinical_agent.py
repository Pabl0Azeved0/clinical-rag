from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai import Agent, PromptedOutput, RunContext
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from clinical_rag.config import Settings
from clinical_rag.retrieval.retriever import Retriever
from clinical_rag.agent.models import ClinicalAnswer
from clinical_rag.agent.validators import grounding_validator


@dataclass
class AgentDeps:
    retriever: Retriever
    settings: Settings
    # R6: run-scoped {index -> retrieved dict} map, filled by search_evidence and
    # later read by the Task 2 grounding validator. One numbering, one source of truth.
    retrieved: dict[int, dict] = field(default_factory=dict)
    broaden_used: bool = False


INSTRUCTIONS = (
    "You are a clinical information assistant. To answer, you MUST call the "
    "search_evidence tool and use ONLY the numbered passages it returns. Cite supporting "
    "passages inline using bracketed numbers like [1] or [2], and list them in citations "
    "with their matching index. If the evidence does not contain enough information to "
    "answer, set grounded=false and say you don't have enough indexed evidence; do NOT "
    "answer from prior knowledge and do NOT fabricate a citation. Never give personal "
    "medical advice or a diagnosis. "
    "If the first search_evidence results look weak or empty, you may call broaden_search "
    "ONCE to retry with a wider net; never call it more than once."
)


def build_model(settings: Settings) -> OpenAIChatModel:
    """Map the existing pluggable provider config to a PydanticAI model.

    Both providers speak the OpenAI-compatible API, so the same model class is used
    with a different base URL + key.
    """
    if settings.llm_provider == "ollama":
        return OpenAIChatModel(
            settings.llm_model,
            provider=OpenAIProvider(
                base_url=f"{settings.llm_base_url}/v1", api_key="ollama"
            ),
        )
    if settings.llm_provider == "groq":
        return OpenAIChatModel(
            settings.llm_model,
            provider=OpenAIProvider(
                base_url="https://api.groq.com/openai/v1",
                api_key=settings.llm_api_key,
            ),
        )
    raise ValueError(f"agent model not wired for provider: {settings.llm_provider}")


def _format_evidence(
    results: list[dict], start: int = 1
) -> tuple[str, dict[int, dict]]:
    """Number retrieved dicts once; return (context string, {index: result})."""
    numbered: dict[int, dict] = {}
    blocks: list[str] = []
    for i, r in enumerate(results, start):
        numbered[i] = r
        blocks.append(f"[{i}] {r['title']} ({r['url']})\n{r['text']}")
    return "\n\n".join(blocks), numbered


def _next_index(retrieved: dict[int, dict]) -> int:
    return (max(retrieved) if retrieved else 0) + 1


def _is_weak(retrieved: dict[int, dict], threshold: float) -> bool:
    if not retrieved:
        return True
    return min(r["distance"] for r in retrieved.values()) > threshold


def broaden_decision(
    retrieved: dict[int, dict], broaden_used: bool, threshold: float
) -> str | None:
    """Return a refusal message if broaden must NOT run, else None (ok to run)."""
    if broaden_used:
        return "broaden_search may only be used once per question."
    if not _is_weak(retrieved, threshold):
        return "Existing results are sufficient; broaden_search is not needed."
    return None


async def _retrieve(deps: AgentDeps, query: str, k: int | None) -> list[dict]:
    """Fetch passages via the configured transport (local Retriever or MCP)."""
    # The model can emit a nonsensical k (e.g. 0); fall back to the configured top_k
    # so both transports behave identically (the in-process Retriever masks this via
    # `k or top_k`, but the MCP server would pass 0 straight to Chroma and error).
    if not k or k <= 0:
        k = getattr(deps.settings, "top_k", 5)
    if deps.settings.retrieval_transport == "mcp":
        from clinical_rag.mcp_server.client import search_evidence_over_mcp

        return await search_evidence_over_mcp(deps.settings, query, k)
    return deps.retriever.retrieve(query, k)


def build_agent(settings: Settings) -> Agent[AgentDeps, ClinicalAnswer]:
    # Output mode depends on model capability:
    #  - Ollama / small local models (llama3.2:3b) can't do tool-based final output
    #    reliably, so we use PromptedOutput (prompt for JSON text, parse it).
    #  - Capable API providers (e.g. Groq) use the default tool output. They also
    #    REJECT PromptedOutput here: it sets JSON mode, and Groq forbids json mode +
    #    tool calling in the same request.
    prompted = settings.llm_provider == "ollama"
    agent = Agent(
        build_model(settings),
        deps_type=AgentDeps,
        output_type=PromptedOutput(ClinicalAnswer) if prompted else ClinicalAnswer,
        instructions=INSTRUCTIONS,
        retries=3,
        # Cap each LLM call so a stalled request can't hang the whole run.
        model_settings=ModelSettings(
            timeout=getattr(settings, "request_timeout", 60.0)
        ),
    )

    @agent.tool
    async def search_evidence(
        ctx: RunContext[AgentDeps], query: str, k: int | None = None
    ) -> str:
        """Search the indexed clinical evidence and return numbered, citation-tagged context.

        k defaults to the configured top_k when the model omits it.
        """
        results = await _retrieve(ctx.deps, query, k)
        start = _next_index(ctx.deps.retrieved)
        context, numbered = _format_evidence(results, start)
        ctx.deps.retrieved.update(numbered)  # R6: store the numbered map on run context
        return context if context else "No evidence found in the index."

    @agent.tool
    async def broaden_search(
        ctx: RunContext[AgentDeps], query: str, k: int = 10
    ) -> str:
        """Retry retrieval with a wider net. Allowed once, only when the first search was weak."""
        refusal = broaden_decision(
            ctx.deps.retrieved,
            ctx.deps.broaden_used,
            ctx.deps.settings.weak_distance_threshold,
        )
        if refusal:
            return refusal
        ctx.deps.broaden_used = True
        results = await _retrieve(ctx.deps, query, k)
        start = _next_index(ctx.deps.retrieved)
        context, numbered = _format_evidence(results, start)
        ctx.deps.retrieved.update(numbered)
        return context if context else "No additional evidence found in the index."

    agent.output_validator(grounding_validator)

    return agent


# Surfaced when the model exhausts its retry budget (e.g. a weak local model that
# keeps failing to cite under the grounding nudge). Degrading to an honest refusal
# keeps the guardrail intact — we never ship a fabricated/ungrounded answer — while
# not crashing the caller.
UNGROUNDED_FALLBACK = (
    "I couldn't ground an answer in the indexed evidence for that question. "
    "Please rephrase or ask something more specific."
)


def run_agent(
    agent: Agent[AgentDeps, ClinicalAnswer], question: str, deps: AgentDeps
) -> tuple[ClinicalAnswer, list[dict]]:
    """Run one agent turn; return (answer, tool-call trace).

    A retry-exhaustion (`UnexpectedModelBehavior`) degrades to a grounded=False
    refusal instead of raising, so scripts/UI never crash on a weak model. A transient
    provider error (`ModelHTTPError`, e.g. Groq intermittently rejecting a malformed
    tool call as tool_use_failed) is retried a few times with clean run state; if it
    keeps failing the last error propagates for the caller to handle.
    """

    def _refusal():
        return (
            ClinicalAnswer(
                answer=UNGROUNDED_FALLBACK, citations=[], grounded=False, confidence=0.0
            ),
            [],
        )

    last_error: ModelHTTPError | None = None
    for _ in range(3):
        try:
            result = agent.run_sync(question, deps=deps)
        except UnexpectedModelBehavior:
            return _refusal()
        except ModelHTTPError as exc:
            last_error = exc
            deps.retrieved.clear()
            deps.broaden_used = False
            continue
        tool_calls = [
            {"tool": p.tool_name, "args": p.args}
            for m in result.all_messages()
            for p in getattr(m, "parts", [])
            if isinstance(p, ToolCallPart)
        ]
        return result.output, tool_calls
    raise last_error
