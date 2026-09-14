"""Critic — the quality gate.

Checks that the output answers the goal and that every claim has evidence
behind it. Returns a structured verdict the graph routes on, so a failure sends
work back to the agent that can fix it instead of failing the whole run.
"""

from __future__ import annotations

from ..llm import ModelClient
from ..schemas import Critique, Verdict
from ..state import GraphState

# Absolutes that need evidence to survive.
HEDGE_WORDS = ("always", "never", "cheapest", "best", "guaranteed")


def _offline_critique(payload: dict) -> dict:
    output: str = payload.get("output", "")
    evidence: list[str] = payload.get("evidence", [])
    ok: bool = payload.get("ok", True)
    attempts: int = payload.get("attempts", 0)
    max_attempts: int = payload.get("max_attempts", 2)

    if not ok:
        # Route by cause: missing evidence is the researcher's problem; a tool
        # or formatting failure is the executor's. Re-running the executor on
        # the same empty evidence would just fail again.
        needs_evidence = not evidence
        return Critique(
            verdict=Verdict.RETRY,
            reason="execution failed: no evidence to act on" if needs_evidence
                   else "execution reported failure",
            next_agent="researcher" if needs_evidence else "executor",
        ).model_dump()

    # Sufficiency, not mere presence. A gate that accepts one citation when the
    # goal needs three is not a gate.
    min_evidence: int = payload.get("min_evidence", 1)
    if len(evidence) < min_evidence:
        agent = "researcher" if attempts < max_attempts else "done"
        verdict = Verdict.RETRY if attempts < max_attempts else Verdict.ESCALATE
        return Critique(
            verdict=verdict,
            reason=f"insufficient evidence: {len(evidence)} of {min_evidence} required",
            next_agent=agent,
        ).model_dump()

    unsupported = [
        w for w in HEDGE_WORDS
        if w in output.lower() and not any(w in e.lower() for e in evidence)
    ]
    if unsupported and attempts < max_attempts:
        return Critique(
            verdict=Verdict.RETRY,
            reason="absolute claim without supporting evidence",
            next_agent="researcher",
            unsupported_claims=unsupported,
        ).model_dump()

    if not output.strip():
        return Critique(
            verdict=Verdict.RETRY, reason="empty output", next_agent="executor",
        ).model_dump()

    return Critique(verdict=Verdict.PASS, reason="meets the goal", next_agent="done").model_dump()


def make_critic(client: ModelClient):
    client.register_offline("critique", _offline_critique)

    def critic_node(state: GraphState) -> GraphState:
        execs = state.get("executions", [])
        last = execs[-1] if execs else None
        evidence = [e.claim for r in state.get("research", []) for e in r.evidence]

        critique = client.complete(
            kind="critique",
            payload={
                "goal": state["goal"],
                "output": last.output if last else "",
                "ok": last.ok if last else False,
                "evidence": evidence,
                "attempts": state.get("attempts", 0),
                "max_attempts": state.get("max_attempts", 2),
                "min_evidence": state.get("min_evidence", 1),
            },
            schema=Critique,
            difficulty=0.8,
        )

        patch: GraphState = {
            "critique": critique,
            "trace": [f"critic: {critique.verdict.value} ({critique.reason})"],
        }
        if critique.verdict is Verdict.RETRY:
            patch["attempts"] = state.get("attempts", 0) + 1
            # Re-open the work the retry actually needs to redo. Re-opening only
            # the executor step sends the researcher back to a plan whose next
            # ready step still belongs to the executor, so it finds nothing to
            # do and the retry accomplishes nothing.
            done = list(state.get("completed", []))
            plan = state.get("plan")
            if critique.next_agent == "researcher" and plan is not None:
                reopen = {s.id for s in plan.steps if s.agent == "researcher"}
                if last:
                    reopen.add(last.step_id)
                patch["completed"] = [c for c in done if c not in reopen]
            elif last:
                patch["completed"] = [c for c in done if c != last.step_id]
        elif critique.verdict is Verdict.PASS and last:
            patch["final"] = last.output
        return patch

    return critic_node


critic_node = make_critic
