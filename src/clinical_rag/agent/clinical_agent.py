from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai import Agent, PromptedOutput, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from clinical_rag.config import Settings
from clinical_rag.retrieval.retriever import Retriever
from clinical_rag.agent.models import ClinicalAnswer


@dataclass
class AgentDeps:
    retriever: Retriever
    settings: Settings
    # R6: run-scoped {index -> retrieved dict} map, filled by search_evidence and
    # later read by the Task 2 grounding validator. One numbering, one source of truth.
    retrieved: dict[int, dict] = field(default_factory=dict)


INSTRUCTIONS = (
    "You are a clinical information assistant. To answer, you MUST call the "
    "search_evidence tool and use ONLY the numbered passages it returns. Cite supporting "
    "passages inline using bracketed numbers like [1] or [2], and list them in citations "
    "with their matching index. If the evidence does not contain enough information to "
    "answer, set grounded=false and say you don't have enough indexed evidence; do NOT "
    "answer from prior knowledge and do NOT fabricate a citation. Never give personal "
    "medical advice or a diagnosis."
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


def _format_evidence(results: list[dict]) -> tuple[str, dict[int, dict]]:
    """Number retrieved dicts once; return (context string, {index: result})."""
    numbered: dict[int, dict] = {}
    blocks: list[str] = []
    for i, r in enumerate(results, 1):
        numbered[i] = r
        blocks.append(f"[{i}] {r['title']} ({r['url']})\n{r['text']}")
    return "\n\n".join(blocks), numbered


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
        context, numbered = _format_evidence(results)
        ctx.deps.retrieved.update(numbered)  # R6: store the numbered map on run context
        return context if context else "No evidence found in the index."

    return agent
