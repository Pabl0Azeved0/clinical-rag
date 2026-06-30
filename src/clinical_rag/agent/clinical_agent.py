from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai import Agent, PromptedOutput, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

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
    """Map the existing pluggable provider config to a PydanticAI model."""
    if settings.llm_provider == "ollama":
        return OpenAIChatModel(
            settings.llm_model,
            provider=OpenAIProvider(
                base_url=f"{settings.llm_base_url}/v1", api_key="ollama"
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


def build_agent(settings: Settings) -> Agent[AgentDeps, ClinicalAnswer]:
    # PromptedOutput (not native/tool output): small local models like llama3.2:3b
    # skip the tool entirely under Ollama's native JSON-schema mode, and emit prose
    # instead of a final-result tool call under tool-output mode. Prompted JSON keeps
    # tool-calling intact while still yielding a parseable ClinicalAnswer; retries let
    # the model self-correct malformed JSON.
    agent = Agent(
        build_model(settings),
        deps_type=AgentDeps,
        output_type=PromptedOutput(ClinicalAnswer),
        instructions=INSTRUCTIONS,
        retries=3,
    )

    @agent.tool
    async def search_evidence(
        ctx: RunContext[AgentDeps], query: str, k: int = 5
    ) -> str:
        """Search the indexed clinical evidence and return numbered, citation-tagged context."""
        results = ctx.deps.retriever.retrieve(query, k)
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
        results = ctx.deps.retriever.retrieve(query, k)
        start = _next_index(ctx.deps.retrieved)
        context, numbered = _format_evidence(results, start)
        ctx.deps.retrieved.update(numbered)
        return context if context else "No additional evidence found in the index."

    agent.output_validator(grounding_validator)

    return agent
