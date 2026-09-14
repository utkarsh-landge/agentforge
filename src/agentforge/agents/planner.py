"""Planner — decomposes the goal. It does not do the work."""

from __future__ import annotations

from ..llm import ModelClient
from ..schemas import Plan, Step
from ..state import GraphState


def _offline_plan(payload: dict) -> dict:
    """Deterministic decomposition: gather, compare, then report."""
    goal = payload["goal"]
    return Plan(
        goal=goal,
        steps=[
            Step(id="s1", description=f"Gather candidate options for: {goal}", agent="researcher"),
            Step(id="s2", description="Compare candidates on the stated criteria",
                 agent="researcher", depends_on=["s1"]),
            Step(id="s3", description="Produce the final recommendation",
                 agent="executor", depends_on=["s2"]),
        ],
    ).model_dump()


def make_planner(client: ModelClient):
    client.register_offline("plan", _offline_plan)

    def planner_node(state: GraphState) -> GraphState:
        plan = client.complete(
            kind="plan",
            payload={"goal": state["goal"], "memory": state.get("memory", [])},
            schema=Plan,
            difficulty=0.7,
        )
        return {
            "plan": plan,
            "completed": state.get("completed", []),
            "trace": [f"planner: {len(plan.steps)} steps"],
        }

    return planner_node


planner_node = make_planner
