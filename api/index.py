"""Serverless entry point for the AgentForge demo.

Runs the real graph — same nodes, same critic loop, same retry routing as the
CLI. The offline backend is forced on so the public demo costs nothing and
cannot be run up as someone else's API bill.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The package lives under src/ next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from agentforge.graph import build_graph  # noqa: E402
from agentforge.llm import ModelClient  # noqa: E402
from agentforge.memory import MemoryStore  # noqa: E402

app = FastAPI(title="AgentForge", docs_url="/api/docs")

MAX_CORPUS = 24
MAX_GOAL = 300


class RunRequest(BaseModel):
    goal: str = Field(min_length=3, max_length=MAX_GOAL)
    corpus: list[str] = Field(default_factory=list, max_length=MAX_CORPUS)
    min_evidence: int = Field(default=2, ge=1, le=6)
    max_attempts: int = Field(default=2, ge=0, le=3)
    approve: bool = False


# Vercel rewrites route on the *destination* path, so the function receives
# "/api/index.py" rather than the caller's "/api/run". Every GET under /api is
# a health check and every POST is a run, so match on method and accept any path.
@app.get("/{full_path:path}")
def health(full_path: str = "") -> dict:
    return {"ok": True, "service": "agentforge", "path": full_path}


@app.post("/{full_path:path}")
def run(req: RunRequest, full_path: str = "") -> dict:
    client = ModelClient(offline=True)
    memory = MemoryStore(embed=False)
    for doc in req.corpus[:MAX_CORPUS]:
        memory.add(doc[:400])

    graph, backend = build_graph(client, memory)
    state = graph.invoke(
        {
            "goal": req.goal,
            "completed": [],
            "memory": memory.retrieve(req.goal, k=8, floor=0.0),
            "attempts": 0,
            "max_attempts": req.max_attempts,
            "min_evidence": req.min_evidence,
            "approved": req.approve,
            "trace": [],
        },
        config={"configurable": {"thread_id": f"web-{abs(hash(req.goal)) % 10**8}"}},
    )

    critique = state.get("critique")
    return {
        "trace": state.get("trace", []),
        "final": state.get("final"),
        "verdict": critique.verdict.value if critique else None,
        "reason": critique.reason if critique else None,
        "attempts": state.get("attempts", 0),
        "evidence_found": sum(len(r.evidence) for r in state.get("research", [])),
        "awaiting_approval": state.get("awaiting_approval"),
        "checkpointer": backend,
        "usage": {
            "calls": client.usage.calls,
            "cached": client.usage.cached,
            "cost_units": round(client.usage.cost_units, 2),
            "by_tier": client.usage.by_tier,
        },
    }
