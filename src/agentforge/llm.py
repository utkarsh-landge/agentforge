"""Model access, routing and caching.

Two things live here:

1. A provider-agnostic `complete()` that returns a parsed pydantic model.
2. Dynamic routing — a step is sent to the cheapest tier that can handle it,
   and identical requests are served from cache instead of a second call.

With no API key configured, the offline backend runs instead. It is
deterministic, costs nothing, and lets the graph, the retry loop and the whole
eval suite run in CI. It is not a language model and does not pretend to be —
it is a scripted stand-in so the *orchestration* can be exercised and measured.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# Relative cost per 1k tokens, used to pick a tier and to report spend.
TIERS: dict[str, float] = {"small": 0.15, "medium": 1.00, "large": 5.00}


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    cost_units: float = 0.0
    by_tier: dict[str, int] = field(default_factory=dict)

    def record(self, tier: str, cached: bool) -> None:
        if cached:
            self.cached += 1
            return
        self.calls += 1
        self.cost_units += TIERS[tier]
        self.by_tier[tier] = self.by_tier.get(tier, 0) + 1


def route(step_kind: str, difficulty: float) -> str:
    """Pick the smallest tier that can carry the step.

    Not every step needs the strongest model. Classification and formatting go
    to `small`; open-ended reasoning and criticism go up a tier.
    """
    if step_kind in {"classify", "format", "extract"}:
        return "small"
    if step_kind in {"critique", "plan"} or difficulty >= 0.66:
        return "large"
    return "medium"


class ModelClient:
    """Wraps whichever provider is configured, plus cache and usage accounting."""

    def __init__(self, offline: bool | None = None) -> None:
        self._cache: dict[str, Any] = {}
        self.usage = Usage()
        if offline is None:
            offline = not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
        self.offline = offline
        self._offline_handlers: dict[str, Callable[[dict], dict]] = {}

    # -- offline backend ---------------------------------------------------

    def register_offline(self, kind: str, handler: Callable[[dict], dict]) -> None:
        """Register a deterministic handler used when no provider is configured."""
        self._offline_handlers[kind] = handler

    # -- main entry point --------------------------------------------------

    def complete(
        self,
        *,
        kind: str,
        payload: dict,
        schema: type[T],
        difficulty: float = 0.5,
    ) -> T:
        tier = route(kind, difficulty)
        key = self._key(kind, payload, tier)

        if key in self._cache:
            self.usage.record(tier, cached=True)
            return schema.model_validate(self._cache[key])

        raw = self._offline(kind, payload) if self.offline else self._provider(kind, payload, tier)
        self.usage.record(tier, cached=False)
        self._cache[key] = raw
        return schema.model_validate(raw)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _key(kind: str, payload: dict, tier: str) -> str:
        blob = json.dumps({"k": kind, "p": payload, "t": tier}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _offline(self, kind: str, payload: dict) -> dict:
        handler = self._offline_handlers.get(kind)
        if handler is None:
            raise RuntimeError(
                f"No offline handler registered for {kind!r}. "
                "Set OPENAI_API_KEY or ANTHROPIC_API_KEY to use a real provider."
            )
        return handler(payload)

    def _provider(self, kind: str, payload: dict, tier: str) -> dict:
        """Call the configured provider and return a dict matching the schema."""
        model = self._model_name(tier)
        prompt = json.dumps(payload, default=str)

        if os.getenv("ANTHROPIC_API_KEY"):
            from langchain_anthropic import ChatAnthropic  # noqa: PLC0415

            llm = ChatAnthropic(model=model, temperature=0)
        else:
            from langchain_openai import ChatOpenAI  # noqa: PLC0415

            llm = ChatOpenAI(model=model, temperature=0)

        for attempt in range(3):
            try:
                reply = llm.invoke(
                    [
                        ("system", f"You are the {kind} agent. Reply with JSON only."),
                        ("human", prompt),
                    ]
                )
                return json.loads(reply.content)
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(0.6 * (attempt + 1))
        raise RuntimeError("unreachable")

    @staticmethod
    def _model_name(tier: str) -> str:
        if os.getenv("ANTHROPIC_API_KEY"):
            return {
                "small": "claude-haiku-4-5-20251001",
                "medium": "claude-sonnet-5",
                "large": "claude-opus-5",
            }[tier]
        return {"small": "gpt-4o-mini", "medium": "gpt-4o", "large": "o3"}[tier]
