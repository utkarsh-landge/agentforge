"""Durable workflow state.

A long-running graph must survive a restart, so LangGraph checkpoints after
every node. Postgres is used when available for durable structured state,
Redis for fast transient state, and an in-memory saver otherwise so tests and
the offline demo need no infrastructure.
"""

from __future__ import annotations

import os


def build_checkpointer():
    """Return the best available checkpointer, with a clear fallback order."""
    pg = os.getenv("POSTGRES_URL")
    if pg:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver  # noqa: PLC0415

            return PostgresSaver.from_conn_string(pg), "postgres"
        except Exception:
            pass

    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            from langgraph.checkpoint.redis import RedisSaver  # noqa: PLC0415

            return RedisSaver.from_conn_string(redis_url), "redis"
        except Exception:
            pass

    from langgraph.checkpoint.memory import MemorySaver  # noqa: PLC0415

    return MemorySaver(), "memory"
