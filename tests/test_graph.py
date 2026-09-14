"""Tests for the orchestration mechanics, not the language model."""
from agentforge.graph import build_graph, run
from agentforge.llm import ModelClient, route
from agentforge.memory import MemoryStore
from agentforge.schemas import Plan, Step, Verdict

CORPUS = [
    "pgvector runs inside PostgreSQL (vector database)",
    "Pinecone is billed per pod hour (vector database)",
    "Weaviate supports hybrid search (vector database)",
]


def test_plan_respects_dependencies():
    plan = Plan(goal="g", steps=[
        Step(id="a", description="", agent="researcher"),
        Step(id="b", description="", agent="executor", depends_on=["a"]),
    ])
    assert plan.next_step(set()).id == "a"
    assert plan.next_step({"a"}).id == "b"
    assert plan.next_step({"a", "b"}) is None


def test_routing_sends_cheap_work_to_small_models():
    assert route("classify", 0.1) == "small"
    assert route("critique", 0.1) == "large"
    assert route("research", 0.9) == "large"
    assert route("research", 0.4) == "medium"


def test_cache_prevents_a_second_identical_call():
    c = ModelClient(offline=True)
    c.register_offline("plan", lambda p: Plan(goal=p["goal"], steps=[]).model_dump())
    for _ in range(3):
        c.complete(kind="plan", payload={"goal": "same"}, schema=Plan)
    assert c.usage.calls == 1
    assert c.usage.cached == 2


def test_memory_retrieves_only_relevant_notes():
    m = MemoryStore(embed=False)
    m.add("user prefers AWS and PostgreSQL")
    m.add("the capital of France is Paris")
    hits = m.retrieve("which database should we use on AWS", k=1)
    assert hits and "AWS" in hits[0]


def test_happy_path_completes():
    final = run("Compare vector database options", corpus=CORPUS, thread_id="t-happy")
    assert final["critique"].verdict is Verdict.PASS
    assert final["final"]


def test_risky_action_pauses_for_approval():
    c = ModelClient(offline=True)
    m = MemoryStore(embed=False)
    for d in CORPUS:
        m.add(d)
    app, _ = build_graph(c, m)
    # Registered after build_graph, which installs the default handlers.
    # Executor proposes an external side effect; the graph must halt.
    c.register_offline("execute", lambda p: {
        "step_id": p["step_id"], "ok": True, "output": "drafted",
        "tool_calls": [{"name": "send_email", "args": {}, "risk": "external_side_effect"}],
    })
    state = app.invoke(
        {"goal": "email the team", "completed": [], "memory": CORPUS,
         "attempts": 0, "max_attempts": 2, "approved": False, "trace": []},
        config={"configurable": {"thread_id": "t-approve"}},
    )
    assert state.get("awaiting_approval")
    assert state.get("final") is None


def test_critic_rejects_insufficient_evidence():
    c = ModelClient(offline=True)
    m = MemoryStore(embed=False)
    app, _ = build_graph(c, m)
    state = app.invoke(
        {"goal": "compare things", "completed": [], "memory": [],
         "attempts": 0, "max_attempts": 1, "min_evidence": 3, "approved": True, "trace": []},
        config={"configurable": {"thread_id": "t-insufficient"}},
    )
    assert state["critique"].verdict is not Verdict.PASS
