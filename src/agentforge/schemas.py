"""Structured outputs.

Agents never hand each other free-form prose. Every hand-off is one of these
models, so the graph can route on a field instead of parsing English.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    PASS = "pass"
    RETRY = "retry"
    ESCALATE = "escalate"


class Risk(str, Enum):
    """Whether an action needs a human before it runs."""

    READ_ONLY = "read_only"
    REVERSIBLE = "reversible"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"


class Step(BaseModel):
    id: str
    description: str
    agent: Literal["researcher", "executor"]
    depends_on: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    goal: str
    steps: list[Step]

    def next_step(self, done: set[str]) -> Step | None:
        """First step whose dependencies are all satisfied."""
        for step in self.steps:
            if step.id not in done and set(step.depends_on) <= done:
                return step
        return None


class Evidence(BaseModel):
    claim: str
    source: str
    confidence: float = Field(ge=0.0, le=1.0)


class ResearchResult(BaseModel):
    step_id: str
    evidence: list[Evidence] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class ToolCall(BaseModel):
    name: str
    args: dict = Field(default_factory=dict)
    risk: Risk = Risk.READ_ONLY


class ExecutionResult(BaseModel):
    step_id: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    output: str = ""
    ok: bool = True
    error: str | None = None


class Critique(BaseModel):
    """The critic's verdict. `next_agent` is what the graph routes on."""

    verdict: Verdict
    reason: str
    next_agent: Literal["planner", "researcher", "executor", "done"] = "done"
    unsupported_claims: list[str] = Field(default_factory=list)
