from __future__ import annotations

import re

from pydantic_ai import ModelRetry, RunContext

from clinical_rag.agent.models import Citation, ClinicalAnswer

_CITATION_RE = re.compile(r"\[(\d+)\]")


def check_grounding(
    output: ClinicalAnswer,
    retrieved: dict[int, dict],
    weak_threshold: float = 0.0,
    allow_nudge: bool = False,
) -> ClinicalAnswer:
    """Deterministically reground the answer against the retrieved numbered map.

    - Raises ModelRetry if the answer cites an [n] that was never retrieved (fabrication).
    - grounded = True only when at least one valid [n] backs the answer (R7: not LLM self-rated).
    - confidence is derived from retrieval distance, not the model's self-report (R7).
    - citations are rebuilt from the retrieved map so title/url always match real sources.
    - When allow_nudge=True and strong evidence was retrieved but no [n] was cited, raises
      ModelRetry asking the model to cite. The nudge is deterministic and never fabricates
      citations — it only asks the model to cite passages it actually retrieved.
    """
    cited = {int(n) for n in _CITATION_RE.findall(output.answer)}
    fabricated = cited - set(retrieved)
    if fabricated:
        raise ModelRetry(
            f"The answer cites {sorted(fabricated)}, which do not match any retrieved "
            "passage. Only cite bracketed numbers shown in the search results, or remove them."
        )

    if not cited and retrieved and allow_nudge:
        best = min(r["distance"] for r in retrieved.values())
        if best < weak_threshold:
            raise ModelRetry(
                "You retrieved relevant passages but cited none. Cite the specific "
                "bracketed number(s) [n] that support each claim, or explicitly state "
                "there is no indexed evidence to answer."
            )

    grounded = bool(cited)
    citations = [
        Citation(index=n, title=retrieved[n]["title"], url=retrieved[n]["url"])
        for n in sorted(cited)
    ]

    if cited:
        best = min(retrieved[n]["distance"] for n in cited)
    elif retrieved:
        best = min(r["distance"] for r in retrieved.values())
    else:
        best = 1.0
    confidence = max(0.0, min(1.0, 1.0 - best))

    return output.model_copy(
        update={"grounded": grounded, "confidence": confidence, "citations": citations}
    )


def grounding_validator(ctx: RunContext, output: ClinicalAnswer) -> ClinicalAnswer:
    # Nudge only on the first attempt: give the model one chance to cite evidence it
    # clearly used, then accept grounded=False. Nudging on every retry lets a weak
    # model loop (nudge -> re-search -> nudge ...) and burn its whole retry budget.
    allow_nudge = ctx.retry == 0
    return check_grounding(
        output,
        ctx.deps.retrieved,
        weak_threshold=ctx.deps.settings.weak_distance_threshold,
        allow_nudge=allow_nudge,
    )
