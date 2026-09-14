# AgentForge

A multi-agent workflow system. Four agents — **planner, researcher, executor,
critic** — wired as an explicit LangGraph state machine, with retries, durable
state, RAG memory, structured hand-offs, cost-aware model routing, and a human
approval gate on anything with an external side effect.

```
planner ─► dispatch ─┬─► researcher ─┐
                     └─► executor ───┴─► critic ─┬─ pass ─────► END
                                                 ├─ retry ────► dispatch
                                                 └─ escalate ─► END
```

The executor has a fifth exit: a step carrying an external side effect writes
`awaiting_approval` and the graph halts until a human resumes it.

---

## Why a graph and not a chain

After execution the critic can approve, send work back to the **researcher**,
send it back to the **executor**, or escalate. Four different edges out of one
node. A linear chain cannot express that, and the routing decision has to be
made on a value the previous node produced — which is why every hand-off is a
pydantic model rather than prose:

```json
{ "verdict": "retry", "reason": "insufficient evidence: 1 of 3 required",
  "next_agent": "researcher" }
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export PYTHONPATH=$PWD/src

agentforge "Compare vector databases for RAG and recommend one" --corpus corpus.txt
pytest -q
python evals/build_tasks.py && python evals/run_eval.py
```

No API key is required. With none set, a deterministic offline backend runs so
the graph, the retry loop and the whole eval suite work in CI at zero cost. Set
`OPENAI_API_KEY` or `ANTHROPIC_API_KEY` to use a real provider — see
`.env.example`.

## Results — and how to read them

`python evals/run_eval.py` over the 150-task set:

| configuration | completion | cost units |
|---|---|---|
| baseline (single pass, no critic, no retry) | **68.0%** (102/150) | 1950 |
| orchestrated (critic routes work back, ≤2 retries) | **100.0%** (150/150) | 2334 |

**Read the 100% with suspicion — it is a property of the offline backend, not a
claim about language models.** The offline researcher is deterministic: when the
critic sends a step back, the widened corpus sweep always finds the evidence, so
every recoverable task recovers. A real model would not be perfect on retry, and
the number would land below 100%. Run it with an API key to get a figure that
means something about model behaviour.

What the eval *does* establish, and what it was built to establish, is the
mechanism: **68% of tasks succeed on a single pass; the remaining 32% fail in
ways a critic can detect and route back.** Every one of those failures is a
`no final answer` — the executor had nothing to act on because the first-pass
query missed a corpus that never names the category it belongs to.

Cost went **up** 19.7%, not down. Retries are not free. The routing and caching
in `llm.py` reduce cost per call (cheap tiers for classification and formatting,
identical requests served from cache), but this eval deliberately does not net
those against the retry spend — that would be two effects in one number.

The task set has three shapes, 50 each:

| shape | corpus | baseline | orchestrated |
|---|---|---|---|
| `well_sourced` | names the category, first-pass search finds it | 50/50 | 50/50 |
| `indirect` | holds the evidence but never names the category | 2/50 | 50/50 |
| `trap` | invites an absolute claim nothing supports | 50/50 | 50/50 |

`indirect` is the whole experiment. The other two shapes are controls: they
confirm the orchestration does not *break* what already worked.

## What is in here

| path | what it holds |
|---|---|
| `src/agentforge/graph.py` | the state machine, edges, and routing predicates |
| `src/agentforge/agents/` | planner, researcher, executor, critic |
| `src/agentforge/schemas.py` | every agent-to-agent contract |
| `src/agentforge/state.py` | the state that makes a ten-step task resumable |
| `src/agentforge/llm.py` | provider abstraction, tier routing, cache, usage accounting |
| `src/agentforge/memory/store.py` | RAG long-term memory |
| `src/agentforge/memory/persistence.py` | Postgres → Redis → in-memory checkpointers |
| `src/agentforge/tools.py` | tool registry, each tagged with a risk level |
| `evals/` | task generator, eval runner, saved results |
| `tests/` | 7 tests covering routing, cache, memory, approval gate, critic |

## Design notes

**State is what makes it resumable.** The model call is stateless; the workflow
is not. `GraphState` carries the goal, plan, completed step ids, evidence,
tool results, retry count and retrieved memory, and LangGraph checkpoints it
after every node. Postgres is used for durable structured state when
`POSTGRES_URL` is set, Redis for fast transient state, in-memory otherwise.

**Retrieval instead of replay.** Passing the whole history into every call is
expensive and noisy. `MemoryStore.retrieve` pulls only the notes relevant to the
current goal.

**The approval gate is the point, not a feature.** Research is low-risk.
Sending an email is not. Tools are tagged `read_only`, `reversible` or
`external_side_effect`, and the graph stops before the last kind.

**Routing by difficulty.** Classification and formatting go to the small tier;
planning and criticism go to the large one. Identical requests are cached rather
than re-issued.

## Bugs this build surfaced

Worth recording, because each one was invisible until the eval ran:

1. The critic passed work the evaluator rejected — it checked whether evidence
   *existed*, not whether there was *enough*. A gate that accepts one citation
   when the goal needs three is not a gate.
2. A retry routed to the researcher re-opened only the executor's step, so the
   researcher found the next ready step still belonged to the executor and
   returned "nothing to do". The retry burned an attempt and changed nothing.
3. An execution that failed for want of evidence was routed back to the
   *executor*, which then failed identically. Failures have to be routed by
   cause, not by who reported them.
