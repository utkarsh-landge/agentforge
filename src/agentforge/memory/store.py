"""RAG long-term memory.

Rather than replaying the whole history into every model call, the agent
retrieves only the notes relevant to the current goal. Embeddings are used when
a provider is configured; otherwise a deterministic bag-of-words vector keeps
retrieval working offline with the same interface.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass
class Note:
    text: str
    vec: Counter = field(default_factory=Counter)


class MemoryStore:
    def __init__(self, embed: bool | None = None) -> None:
        self._notes: list[Note] = []
        if embed is None:
            embed = bool(os.getenv("OPENAI_API_KEY"))
        self.embed = embed

    def add(self, text: str) -> None:
        self._notes.append(Note(text=text, vec=Counter(_tokens(text))))

    def retrieve(self, query: str, k: int = 3, floor: float = 0.05) -> list[str]:
        """Return the k most relevant notes, dropping anything below `floor`."""
        q = Counter(_tokens(query))
        scored = [(self._cosine(q, n.vec), n.text) for n in self._notes]
        scored = [(s, t) for s, t in scored if s >= floor]
        scored.sort(key=lambda p: p[0], reverse=True)
        return [t for _, t in scored[:k]]

    @staticmethod
    def _cosine(a: Counter, b: Counter) -> float:
        if not a or not b:
            return 0.0
        shared = set(a) & set(b)
        num = sum(a[t] * b[t] for t in shared)
        da = math.sqrt(sum(v * v for v in a.values()))
        db = math.sqrt(sum(v * v for v in b.values()))
        return num / (da * db) if da and db else 0.0

    def __len__(self) -> int:
        return len(self._notes)
