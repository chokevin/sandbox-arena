#!/usr/bin/env python3
"""Benchmark — track model performance over time.

Run this after each model improvement (SFT, DPO, RL) to measure progress.
Results are appended to a benchmark history file for trend analysis.

Usage:
    # Benchmark a model against all hard coding challenges
    python benchmark.py --model gpt-4o --arena coding_hard

    # Compare two models
    python benchmark.py --model gpt-4o --arena coding_hard
    python benchmark.py --model gpt-4o-mini --arena coding_hard
    python benchmark.py --show  # display comparison
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dojo import Dojo


BENCHMARK_FILE = "trajectories/benchmark_history.jsonl"


def run_benchmark(model: str, arena_name: str, max_steps: int,
                  mode: str, api_key: str, base_url: str):
    """Run a full benchmark and return results."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    dojo = Dojo(arena_name=arena_name, mode=mode, max_steps=max_steps)
    scenarios = dojo.arena.scenarios()

    system_prompt = (
        "You are an expert Python programmer. "
        "Output ONLY executable Python code — no markdown fences, no explanations."
    )

    results = []
    total_start = time.time()

    for idx, scenario in enumerate(scenarios):
        challenge = dojo.reset(idx)
        scenario_id = scenario.get("id") or scenario.get("ticker") or str(idx)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Solve this challenge:\n\n{challenge}"},
        ]

        best_score = 0
        solved = False
        attempts = 0

        for attempt in range(1, max_steps + 1):
            attempts = attempt
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0.2,
            )
            code = response.choices[0].message.content.strip()
            if code.startswith("```python"):
                code = code[len("```python"):].strip()
            if code.startswith("```"):
                code = code[3:].strip()
            if code.endswith("```"):
                code = code[:-3].strip()

            result = dojo.step(code)
            best_score = max(best_score, result.score)

            if result.info["passed"]:
                solved = True
                break

            if not result.done:
                messages.append({"role": "assistant", "content": code})
                messages.append({
                    "role": "user",
                    "content": f"Score: {result.score}. Feedback:\n{result.observation}\n\nFix and resubmit."
                })

        results.append({
            "scenario_id": scenario_id,
            "solved": solved,
            "best_score": best_score,
            "attempts": attempts,
        })

        icon = "✅" if solved else "❌"
        print(f"  {icon} {scenario_id:<25} score={best_score:>6.1f} attempts={attempts}")

    elapsed = time.time() - total_start
    solved_count = sum(1 for r in results if r["solved"])
    avg_score = sum(r["best_score"] for r in results) / len(results) if results else 0

    return {
        "model": model,
        "arena": arena_name,
        "timestamp": int(time.time()),
        "elapsed": round(elapsed, 1),
        "max_steps": max_steps,
        "mode": mode,
        "total": len(results),
        "solved": solved_count,
        "solve_rate": round(solved_count / len(results) * 100, 1) if results else 0,
        "avg_score": round(avg_score, 1),
        "results": results,
    }


def show_history(arena_filter: str = None):
    """Display benchmark history."""
    path = Path(BENCHMARK_FILE)
    if not path.exists():
        print("No benchmark history found.")
        return

    entries = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            if arena_filter and entry["arena"] != arena_filter:
                continue
            entries.append(entry)

    if not entries:
        print("No matching entries.")
        return

    # Group by arena
    arenas = {}
    for e in entries:
        arenas.setdefault(e["arena"], []).append(e)

    for arena_name, runs in arenas.items():
        print(f"\n{'='*65}")
        print(f"  {arena_name.upper()} Benchmark History")
        print(f"{'='*65}")
        print(f"  {'Model':<20} {'Solve Rate':>10} {'Avg Score':>10} {'Time':>8} {'Date':>12}")
        print(f"  {'─'*60}")

        for run in sorted(runs, key=lambda x: x["timestamp"]):
            from datetime import datetime
            date = datetime.fromtimestamp(run["timestamp"]).strftime("%Y-%m-%d")
            print(f"  {run['model']:<20} {run['solve_rate']:>9.1f}% {run['avg_score']:>9.1f} {run['elapsed']:>7.1f}s {date:>12}")

        print(f"  {'─'*60}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark — track model performance over time")
    parser.add_argument("--model", help="Model to benchmark")
    parser.add_argument("--arena", default="coding_hard", help="Arena to benchmark on")
    parser.add_argument("--max-steps", type=int, default=5, help="Max attempts per challenge")
    parser.add_argument("--mode", default="cluster", choices=["local", "cluster"])
    parser.add_argument("--show", action="store_true", help="Show benchmark history")
    args = parser.parse_args()

    if args.show:
        show_history(arena_filter=args.arena if args.arena != "coding_hard" else None)
        return

    if not args.model:
        print("Specify --model or use --show to view history")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        print("Set OPENAI_API_KEY")
        sys.exit(1)

    print(f"Benchmarking {args.model} on {args.arena} (max {args.max_steps} attempts)")
    print(f"{'='*60}")

    result = run_benchmark(
        model=args.model, arena_name=args.arena, max_steps=args.max_steps,
        mode=args.mode, api_key=api_key, base_url=base_url,
    )

    print(f"\n{'='*60}")
    print(f"  Model:      {result['model']}")
    print(f"  Arena:      {result['arena']}")
    print(f"  Solve Rate: {result['solve_rate']}% ({result['solved']}/{result['total']})")
    print(f"  Avg Score:  {result['avg_score']}")
    print(f"  Time:       {result['elapsed']}s")
    print(f"{'='*60}")

    # Append to history
    path = Path(BENCHMARK_FILE)
    path.parent.mkdir(exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"\n  Saved to {path}")
    print(f"  View history: python benchmark.py --show")


if __name__ == "__main__":
    main()
