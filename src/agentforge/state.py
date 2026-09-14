"""The graph's state.

Every node reads and writes this one object. It is what makes a ten-step task
resumable: the model call is stateless, the workflow is not.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from .schemas import Critique, ExecutionResult, Plan, ResearchResult


def _merge(a: list | None, b: list | None) -> list:
    return (a or []) + (b or [])


class GraphState(TypedDict, total=False):
    goal: str
    plan: Plan | None
    completed: list[str]                       # step ids already done
    research: Annotated[list[ResearchResult], _merge]
    executions: Annotated[list[ExecutionResult], _merge]
    critique: Critique | None
    memory: list[str]                          # retrieved long-term context
    attempts: int                              # critic-driven retries so far
    max_attempts: int
    min_evidence: int                          # sufficiency bar the critic enforces
    awaiting_approval: dict[str, Any] | None   # set when a risky action pauses
    approved: bool
    final: str | None
    trace: Annotated[list[str], _merge]        # human-readable step log
