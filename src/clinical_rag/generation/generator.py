from __future__ import annotations

from dataclasses import dataclass, field

from clinical_rag.llm.provider import LLMProvider

DISCLAIMER = (
    "This information is for educational purposes only and is not medical advice. "
    "Consult a qualified healthcare professional."
)

_PROMPT_TEMPLATE = """\
You are a clinical information assistant. Answer the question below using ONLY the \
numbered context passages provided. Cite supporting passages inline using bracketed \
numbers like [1] or [2]. If the context does not contain enough information to answer \
the question, say "I don't have enough information in the indexed content to answer \
that." Do not provide personal medical advice or diagnosis.

Context:
{context}

Question: {query}

Answer:"""


def _build_prompt(query: str, context: str) -> str:
    return _PROMPT_TEMPLATE.format(context=context, query=query)


@dataclass
class Answer:
    query: str
    text: str
    sources: list[dict] = field(default_factory=list)


class Generator:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def generate_answer(self, query: str, chunks: list[dict]) -> Answer:
        if not chunks:
            return Answer(
                query=query,
                text=f"No indexed evidence available to answer this question.\n\n{DISCLAIMER}",
                sources=[],
            )

        sources = [
            {"index": i, "title": c["title"], "url": c["url"]}
            for i, c in enumerate(chunks, 1)
        ]
        context_blocks = "\n\n".join(
            f"[{i}] {c['title']} ({c['url']})\n{c['text']}"
            for i, c in enumerate(chunks, 1)
        )
        prompt = _build_prompt(query, context_blocks)
        raw = self._provider.generate(prompt).strip()
        return Answer(query=query, text=f"{raw}\n\n{DISCLAIMER}", sources=sources)
