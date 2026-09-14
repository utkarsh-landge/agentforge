from __future__ import annotations

import argparse
import json

from .graph import run


def main() -> int:
    p = argparse.ArgumentParser(prog="agentforge")
    p.add_argument("goal", help="what you want the agents to accomplish")
    p.add_argument("--corpus", help="path to a newline-separated corpus file")
    p.add_argument("--approve", action="store_true", help="pre-approve risky actions")
    p.add_argument("--json", action="store_true", help="emit the full final state")
    args = p.parse_args()

    corpus = []
    if args.corpus:
        corpus = [ln.strip() for ln in open(args.corpus) if ln.strip()]

    final = run(args.goal, corpus=corpus, approved=args.approve)

    if args.json:
        print(json.dumps(final, default=str, indent=2))
        return 0

    for line in final.get("trace", []):
        print(f"  {line}")
    if final.get("awaiting_approval"):
        print(f"\nPAUSED for approval: {final['awaiting_approval']}")
        return 2
    print("\n" + (final.get("final") or "(no output)"))
    return 0
