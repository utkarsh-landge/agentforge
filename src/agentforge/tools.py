"""Tools the executor can call.

Each carries a risk level. Anything with an external side effect stops the
graph for human approval before it runs.
"""

from __future__ import annotations

from typing import Callable

from .schemas import Risk

Tool = Callable[..., str]
REGISTRY: dict[str, tuple[Tool, Risk]] = {}


def tool(name: str, risk: Risk = Risk.READ_ONLY) -> Callable[[Tool], Tool]:
    def deco(fn: Tool) -> Tool:
        REGISTRY[name] = (fn, risk)
        return fn

    return deco


@tool("search_index", Risk.READ_ONLY)
def search_index(query: str, corpus: list[str] | None = None) -> str:
    """Keyword search over a supplied corpus. No network, no key."""
    corpus = corpus or []
    stop = {"the", "and", "for", "with", "その", "one", "options", "compare", "recommend"}
    terms = {w.strip(".,:;()").lower() for w in query.split()}
    terms = {w for w in terms if len(w) > 3 and w not in stop}
    if not terms:
        return "no matches"

    scored = []
    for doc in corpus:
        words = {w.strip(".,:;()").lower() for w in doc.split()}
        overlap = len(terms & words)
        if overlap:
            scored.append((overlap, doc))
    scored.sort(key=lambda p: -p[0])
    return "\n".join(d for _, d in scored[:6]) or "no matches"


@tool("compare_table", Risk.READ_ONLY)
def compare_table(rows: list[dict], keys: list[str]) -> str:
    header = " | ".join(keys)
    body = "\n".join(" | ".join(str(r.get(k, "—")) for k in keys) for r in rows)
    return f"{header}\n{body}"


@tool("write_report", Risk.REVERSIBLE)
def write_report(title: str, sections: list[str]) -> str:
    parts = "\n\n".join(f"## {s}" for s in sections)
    return f"# {title}\n\n{parts}"


@tool("send_email", Risk.EXTERNAL_SIDE_EFFECT)
def send_email(to: str, subject: str, body: str) -> str:
    """Deliberately gated: the graph pauses for approval before this runs."""
    return f"sent to {to}: {subject}"


def risk_of(name: str) -> Risk:
    entry = REGISTRY.get(name)
    return entry[1] if entry else Risk.READ_ONLY


def call(name: str, **kwargs) -> str:
    entry = REGISTRY.get(name)
    if entry is None:
        raise KeyError(f"unknown tool: {name}")
    return entry[0](**kwargs)
