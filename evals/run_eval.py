"""Measure end-to-end task completion, baseline vs orchestrated.

Two configurations over the same 150 tasks:

  baseline      — one pass. Plan, research once, execute, stop. No critic, no
                  retry. This is the "just call the model in sequence" setup.
  orchestrated  — the full graph: critic verdicts route work back to the agent
                  that can fix it, up to `max_attempts`.

A task counts as complete when all three hold:
  1. a final answer exists and is non-empty
  2. it draws on at least `min_evidence` distinct corpus facts
  3. it contains no absolute claim ("cheapest", "always", …) that the corpus
     does not support

The same checker scores both configurations. Nothing about the numbers is
hard-coded — run it and see what you get.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agentforge.graph import build_graph
from agentforge.llm import ModelClient
from agentforge.memory import MemoryStore

HERE = Path(__file__).parent


def load_tasks() -> list[dict]:
    path = HERE / "tasks.jsonl"
    if not path.exists():
        raise SystemExit("tasks.jsonl missing — run `python evals/build_tasks.py` first")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def is_complete(task: dict, state: dict) -> tuple[bool, str]:
    """The single success criterion, applied identically to both configs."""
    final = (state.get("final") or "").strip()
    if not final:
        return False, "no final answer"

    corpus_facts = {c.strip().lower() for c in task["corpus"]}
    cited = sum(1 for fact in corpus_facts if fact in final.lower())
    if cited < task["min_evidence"]:
        return False, f"cited {cited} facts, needed {task['min_evidence']}"

    low = final.lower()
    for word in task["forbidden_absolutes"]:
        if word in low and not any(word in c.lower() for c in task["corpus"]):
            return False, f"unsupported absolute: {word!r}"

    return True, "ok"


def run_one(task: dict, *, orchestrated: bool) -> tuple[bool, str, float]:
    client = ModelClient(offline=True)
    memory = MemoryStore(embed=False)
    for doc in task["corpus"]:
        memory.add(doc)

    app, _ = build_graph(client, memory)
    state = {
        "goal": task["goal"],
        "completed": [],
        "memory": memory.retrieve(task["goal"], k=8, floor=0.0),
        "attempts": 0,
        # The baseline gets no retries: the critic still reports, but the graph
        # never routes work back, which is exactly the single-pass setup.
        "max_attempts": 2 if orchestrated else 0,
        "min_evidence": task["min_evidence"],
        "approved": True,
        "trace": [],
    }
    final = app.invoke(state, config={"configurable": {"thread_id": task["id"]}})
    ok, why = is_complete(task, final)
    return ok, why, client.usage.cost_units


def evaluate(tasks: list[dict], *, orchestrated: bool) -> dict:
    t0 = time.perf_counter()
    passed = 0
    by_shape: dict[str, list[int]] = {}
    reasons: dict[str, int] = {}
    cost = 0.0

    for task in tasks:
        ok, why, spend = run_one(task, orchestrated=orchestrated)
        cost += spend
        passed += ok
        bucket = by_shape.setdefault(task["shape"], [0, 0])
        bucket[1] += 1
        bucket[0] += ok
        if not ok:
            reasons[why.split(":")[0]] = reasons.get(why.split(":")[0], 0) + 1

    return {
        "config": "orchestrated" if orchestrated else "baseline",
        "n": len(tasks),
        "passed": passed,
        "rate": round(100 * passed / len(tasks), 1),
        "by_shape": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_shape.items())},
        "failure_reasons": dict(sorted(reasons.items(), key=lambda p: -p[1])),
        "cost_units": round(cost, 1),
        "seconds": round(time.perf_counter() - t0, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="run only the first N tasks")
    ap.add_argument("--save", action="store_true", help="write results/latest.json")
    args = ap.parse_args()

    tasks = load_tasks()
    if args.limit:
        tasks = tasks[: args.limit]

    base = evaluate(tasks, orchestrated=False)
    orch = evaluate(tasks, orchestrated=True)

    print(f"\ntasks: {base['n']}\n")
    for r in (base, orch):
        print(f"  {r['config']:<13} {r['rate']:>5}%  ({r['passed']}/{r['n']})"
              f"   cost {r['cost_units']:>6}   {r['seconds']}s")
        print(f"  {'':<13} by shape: {r['by_shape']}")
        if r["failure_reasons"]:
            print(f"  {'':<13} failures: {r['failure_reasons']}")
        print()

    delta = round(orch["rate"] - base["rate"], 1)
    saved = round(100 * (1 - orch["cost_units"] / base["cost_units"]), 1) if base["cost_units"] else 0
    print(f"  completion: {base['rate']}% -> {orch['rate']}%  ({delta:+} points)")
    print(f"  cost change vs baseline: {-saved:+}%\n")

    if args.save:
        out = HERE / "results"
        out.mkdir(exist_ok=True)
        (out / "latest.json").write_text(json.dumps({"baseline": base, "orchestrated": orch}, indent=2))
        print(f"  saved {out / 'latest.json'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
