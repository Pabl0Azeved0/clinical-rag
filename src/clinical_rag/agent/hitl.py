"""Human-in-the-loop gate for low-grounding / low-confidence answers.

PydanticAI's native tool-approval (requires_approval / ApprovalRequired /
DeferredToolRequests) gates TOOL CALLS. Our finalization is a PromptedOutput text
answer (chosen in Task 1 because llama3.2:3b can't do reliable tool-output), so the
gate lives at the finalization boundary instead, keyed on the deterministic
grounded / confidence signals the validator computes (R7). The human still gets an
approve / reject / edit decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from clinical_rag.agent.models import ClinicalAnswer


def needs_approval(
    answer: ClinicalAnswer, confidence_threshold: float, has_evidence: bool = True
) -> bool:
    """True when an answer should pause for human review before being surfaced.

    A *pure no-evidence refusal* (nothing was retrieved) has nothing for a human to
    approve — it is surfaced directly as a clean refusal. Review is reserved for the
    cases that actually warrant it: an ungrounded answer produced *despite* evidence
    being available, or a grounded answer whose confidence is below the threshold.
    """
    if not answer.grounded:
        return has_evidence
    return answer.confidence < confidence_threshold


@dataclass
class Decision:
    action: str  # "approve" | "reject" | "edit"
    edited_answer: str | None = None


def apply_decision(answer: ClinicalAnswer, decision: Decision) -> ClinicalAnswer | None:
    """Resolve a reviewer decision to the answer to surface.

    approve -> the original answer; reject -> None (nothing surfaced);
    edit -> a copy with the reviewer's edited text. Raises on an unknown action or
    an edit decision without edited_answer.
    """
    if decision.action == "approve":
        return answer
    if decision.action == "reject":
        return None
    if decision.action == "edit":
        if not decision.edited_answer:
            raise ValueError("edit decision requires edited_answer")
        return answer.model_copy(update={"answer": decision.edited_answer})
    raise ValueError(f"unknown decision action: {decision.action}")
