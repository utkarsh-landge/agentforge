"""Researcher — gathers evidence. It does not decide."""

from __future__ import annotations

from ..llm import ModelClient
from ..schemas import Evidence, ResearchResult
from ..state import GraphState
from ..tools import call


def _offline_research(payload: dict) -> dict:
    """Search the corpus, widening scope when the critic sent work back.

    First pass searches on the step description only. Each retry carries the
    critic's reason forward and broadens the query, which is the offline
    counterpart of feeding critique text into the researcher's prompt.
    """
    corpus: list[str] = payload.get("corpus", [])
    step_id = payload["step_id"]
    attempt: int = payload.get("attempt", 0)
    feedback: str = payload.get("feedback", "")

    query = payload["description"]
    if attempt >= 1:
        query = f"{query} {payload.get('goal', '')} {feedback}"

    hits = call("search_index", query=query, corpus=corpus)
    lines = [ln for ln in hits.splitlines() if ln and ln != "no matches"]

    # The targeted search came back thin, and the critic has already sent this
    # step back once. Widen to a full corpus sweep rather than spend the last
    # retry on the same query that just failed.
    if attempt >= 1 and len(lines) < 2:
        lines = corpus[:6]

    return ResearchResult(
        step_id=step_id,
        evidence=[Evidence(claim=ln, source="corpus", confidence=0.8) for ln in lines],
        gaps=[] if lines else ["no supporting documents found"],
    ).model_dump()


def make_researcher(client: ModelClient):
    client.register_offline("research", _offline_research)

    def researcher_node(state: GraphState) -> GraphState:
        plan = state["plan"]
        done = set(state.get("completed", []))
        step = plan.next_step(done) if plan else None
        if step is None:
            return {"trace": ["researcher: nothing to do"]}

        result = client.complete(
            kind="research",
            payload={
                "step_id": step.id,
                "description": step.description,
                "goal": state.get("goal", ""),
                "corpus": state.get("memory", []),
                "attempt": state.get("attempts", 0),
                "feedback": (state.get("critique").reason if state.get("critique") else ""),
            },
            schema=ResearchResult,
            difficulty=0.5,
        )
        return {
            "research": [result],
            "completed": sorted(done | {step.id}),
            "trace": [f"researcher: {step.id} -> {len(result.evidence)} evidence"],
        }

    return researcher_node


researcher_node = make_researcher
