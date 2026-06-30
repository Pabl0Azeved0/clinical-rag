from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A single evidence citation returned by the agent.

    index matches the bracketed number used inline in the answer text.
    """

    index: int
    title: str
    url: str


class ClinicalAnswer(BaseModel):
    """Structured answer produced by the clinical agent.

    grounded: True when the answer is supported by retrieved evidence (coverage flag).
    confidence: Model-reported confidence in [0..1].
    """

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool
    confidence: float

    # TODO(Task 2): grounded/confidence are model-populated for now; Task 2 computes them
    # deterministically (validator from citation coverage / retrieval distance).
