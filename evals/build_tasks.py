"""Generate the evaluation set.

150 tasks across three shapes, because the interesting question is not "does it
work on easy input" but "what happens when it does not".

  well_sourced  — the corpus names the category, so a first-pass keyword
                  search finds it. Both configurations should pass.
  indirect      — the corpus holds enough facts, but none of them name the
                  category, so the narrow first-pass query misses everything.
                  Recoverable only if something widens the search and tries
                  again — which is what the critic loop does.
  trap          — the corpus invites an absolute claim ("cheapest", "always")
                  that nothing in the evidence supports.

Success is checked the same way for every configuration: a final answer exists,
it cites at least `min_evidence` distinct corpus facts, and it makes no absolute
claim the corpus does not support.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(1742)

DOMAINS = [
    ("vector database", ["pgvector", "Pinecone", "Weaviate", "Milvus", "Qdrant", "Chroma"]),
    ("message queue", ["RabbitMQ", "Kafka", "SQS", "NATS", "Redis Streams", "Pulsar"]),
    ("cache layer", ["Redis", "Memcached", "Hazelcast", "Dragonfly", "KeyDB", "Valkey"]),
    ("query engine", ["Trino", "Presto", "DuckDB", "ClickHouse", "Druid", "Spark SQL"]),
    ("object store", ["S3", "GCS", "R2", "MinIO", "Azure Blob", "Backblaze B2"]),
    ("orchestrator", ["Airflow", "Dagster", "Prefect", "Temporal", "Argo", "Luigi"]),
]

FACTS = [
    "{x} is billed per request rather than per provisioned node",
    "{x} supports horizontal scaling across availability zones",
    "{x} exposes a managed control plane with no servers to run",
    "{x} is open source under a permissive licence",
    "{x} keeps p99 read latency under ten milliseconds at moderate load",
    "{x} replicates writes synchronously to a second region",
    "{x} requires an operator to run it inside Kubernetes",
    "{x} charges separately for storage and for egress",
]

CRITERIA = ["cost", "latency", "operational burden", "scalability", "durability", "ecosystem"]


def build(n: int = 150) -> list[dict]:
    tasks: list[dict] = []
    for i in range(n):
        domain, options = DOMAINS[i % len(DOMAINS)]
        shape = ("well_sourced", "indirect", "trap")[i % 3]
        picks = random.sample(options, 4)
        criterion = CRITERIA[i % len(CRITERIA)]

        if shape == "well_sourced":
            corpus = [
                f"{random.choice(FACTS).format(x=p)} ({domain})" for p in picks for _ in range(2)
            ]
            min_evidence = 3
        elif shape == "indirect":
            # Enough evidence to succeed, but the category word is absent, so
            # the narrow first-pass query cannot reach it.
            corpus = [random.choice(FACTS).format(x=p) for p in picks]
            min_evidence = 2
        else:  # trap
            corpus = [f"{random.choice(FACTS).format(x=p)} ({domain})" for p in picks[:2]]
            corpus.append(f"Pricing for {picks[0]} and {picks[1]} varies by region and commitment")
            min_evidence = 2

        tasks.append(
            {
                "id": f"t{i:03d}",
                "shape": shape,
                "goal": f"Compare {domain} options on {criterion} and recommend one",
                "corpus": corpus,
                "min_evidence": min_evidence,
                "forbidden_absolutes": ["always", "never", "cheapest", "best", "guaranteed"],
            }
        )
    return tasks


if __name__ == "__main__":
    out = Path(__file__).parent / "tasks.jsonl"
    tasks = build()
    out.write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    shapes = {}
    for t in tasks:
        shapes[t["shape"]] = shapes.get(t["shape"], 0) + 1
    print(f"wrote {len(tasks)} tasks to {out.name}: {shapes}")
