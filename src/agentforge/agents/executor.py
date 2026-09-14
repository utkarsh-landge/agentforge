"""Executor — performs actions and calls tools.

Anything with an external side effect stops here and waits for a human.
"""

from __future__ import annotations

from ..llm import ModelClient
from ..schemas import ExecutionResult, Risk, ToolCall
from ..state import GraphState
from ..tools import call, risk_of


def _offline_execute(payload: dict) -> dict:
    step_id = payload["step_id"]
    evidence: list[str] = payload.get("evidence", [])
    if not evidence:
        return ExecutionResult(
            step_id=step_id, ok=False, error="no evidence to act on"
        ).model_dump()

    body = call("write_report", title=payload["goal"], sections=evidence[:5])
    return ExecutionResult(
        step_id=step_id,
        tool_calls=[ToolCall(name="write_report", args={"title": payload["goal"]},
                             risk=risk_of("write_report"))],
        output=body,
        ok=True,
    ).model_dump()


def make_executor(client: ModelClient):
    client.register_offline("execute", _offline_execute)

    def executor_node(state: GraphState) -> GraphState:
        plan = state["plan"]
        done = set(state.get("completed", []))
        step = plan.next_step(done) if plan else None
        if step is None:
            return {"trace": ["executor: nothing to do"]}

        evidence = [e.claim for r in state.get("research", []) for e in r.evidence]
        result = client.complete(
            kind="execute",
            payload={"step_id": step.id, "goal": state["goal"], "evidence": evidence},
            schema=ExecutionResult,
            difficulty=0.5,
        )

        # Gate: pause the graph before anything with an external side effect.
        risky = [tc for tc in result.tool_calls if tc.risk is Risk.EXTERNAL_SIDE_EFFECT]
        if risky and not state.get("approved"):
            return {
                "awaiting_approval": {
                    "step_id": step.id,
                    "tools": [tc.name for tc in risky],
                },
                "trace": [f"executor: {step.id} paused for approval ({risky[0].name})"],
            }

        return {
            "executions": [result],
            "completed": sorted(done | {step.id}),
            "trace": [f"executor: {step.id} ok={result.ok}"],
        }

    return executor_node


executor_node = make_executor
