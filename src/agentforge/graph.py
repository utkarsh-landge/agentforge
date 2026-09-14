"""The graph.

The reason this is a state machine and not a prompt chain: after execution the
critic can approve, send work back to the researcher, send it back to the
executor, or escalate. Those are four different edges out of one node, and a
linear chain cannot express them.

    planner ─► dispatch ─┬─► researcher ─┐
                         └─► executor ───┴─► critic ─┬─ pass ──────► END
                                                     ├─ retry ─────► dispatch
                                                     └─ escalate ──► END

The executor has a fifth exit: when a step carries an external side effect it
writes `awaiting_approval` and the graph interrupts so a human can decide.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .agents.critic import make_critic
from .agents.executor import make_executor
from .agents.planner import make_planner
from .agents.researcher import make_researcher
from .llm import ModelClient
from .memory import MemoryStore, build_checkpointer
from .schemas import Verdict
from .state import GraphState


def _dispatch(state: GraphState) -> str:
    """Send the next ready step to whichever agent owns it."""
    plan = state.get("plan")
    if plan is None:
        return "critic"
    step = plan.next_step(set(state.get("completed", [])))
    if step is None:
        return "critic"
    return step.agent


def _after_critic(state: GraphState) -> str:
    critique = state.get("critique")
    if critique is None or critique.verdict is Verdict.PASS:
        return END
    if critique.verdict is Verdict.ESCALATE:
        return END
    if state.get("attempts", 0) >= state.get("max_attempts", 2):
        return END
    return critique.next_agent if critique.next_agent in {"researcher", "executor"} else END


def _after_executor(state: GraphState) -> str:
    # A pending approval halts the run until a human resumes it.
    if state.get("awaiting_approval"):
        return END
    return "critic"


def build_graph(client: ModelClient | None = None, memory: MemoryStore | None = None):
    """Compile the workflow. Returns (app, backend_name)."""
    client = client or ModelClient()
    memory = memory or MemoryStore()

    g = StateGraph(GraphState)
    g.add_node("planner", make_planner(client))
    g.add_node("researcher", make_researcher(client))
    g.add_node("executor", make_executor(client))
    g.add_node("critic", make_critic(client))

    g.set_entry_point("planner")
    for node in ("planner", "researcher"):
        g.add_conditional_edges(
            node, _dispatch,
            {"researcher": "researcher", "executor": "executor", "critic": "critic"},
        )
    g.add_conditional_edges("executor", _after_executor, {"critic": "critic", END: END})
    g.add_conditional_edges(
        "critic", _after_critic,
        {"researcher": "researcher", "executor": "executor", END: END},
    )

    checkpointer, backend = build_checkpointer()
    return g.compile(checkpointer=checkpointer), backend


def run(
    goal: str,
    *,
    corpus: list[str] | None = None,
    max_attempts: int = 2,
    approved: bool = False,
    thread_id: str = "default",
    client: ModelClient | None = None,
) -> GraphState:
    """Run one goal to completion (or to a human-approval pause)."""
    client = client or ModelClient()
    memory = MemoryStore()
    for doc in corpus or []:
        memory.add(doc)

    app, _ = build_graph(client, memory)
    initial: GraphState = {
        "goal": goal,
        "completed": [],
        "memory": memory.retrieve(goal, k=8),
        "attempts": 0,
        "max_attempts": max_attempts,
        "approved": approved,
        "trace": [],
    }
    return app.invoke(initial, config={"configurable": {"thread_id": thread_id}})
