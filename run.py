#!/usr/bin/env python3
"""Sandbox Arena — evaluate agents locally or at scale on AKS.

Usage:
    python run.py local trading agents/examples/momentum.py
    python run.py local coding agents/examples/solutions.py
    python run.py cluster trading agents/examples/momentum.py --parallel 50
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from arenas.base import EvalResult


def load_arena(arena_name: str):
    """Load an arena by name."""
    if arena_name == "trading":
        from arenas.trading.arena import TradingArena
        return TradingArena()
    elif arena_name == "coding":
        from arenas.coding.arena import CodingArena
        return CodingArena()
    elif arena_name == "blackjack":
        from arenas.blackjack.arena import BlackjackArena
        return BlackjackArena()
    elif arena_name == "survival":
        from arenas.survival.arena import SurvivalArena
        return SurvivalArena()
    else:
        print(f"Unknown arena: {arena_name}")
        print("Available: trading, coding, blackjack, survival")
        sys.exit(1)


def run_local_eval(script: str, timeout: int = 30) -> EvalResult:
    """Run an evaluation script locally in a subprocess."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=timeout
            )
            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout.strip().split("\n")[-1])
                    return EvalResult(
                        score=data["score"],
                        passed=data["passed"],
                        details=data.get("details", {}),
                        stdout=result.stdout,
                    )
                except (json.JSONDecodeError, KeyError) as e:
                    return EvalResult(score=0, passed=False, error=f"Bad output: {e}", stdout=result.stdout)
            else:
                return EvalResult(score=0, passed=False, error=result.stderr, stderr=result.stderr)
        except subprocess.TimeoutExpired:
            return EvalResult(score=0, passed=False, error="Timeout")
        finally:
            Path(f.name).unlink(missing_ok=True)


def run_cluster_eval(script: str, template: str, namespace: str) -> EvalResult:
    """Run an evaluation script in an Agent Sandbox pod."""
    from k8s_agent_sandbox import SandboxClient

    try:
        with SandboxClient(template_name=template, namespace=namespace) as sandbox:
            sandbox.write("eval.py", script)
            result = sandbox.run("python3 eval.py", timeout=60)
            stdout = result.stdout
            if result.exit_code == 0:
                try:
                    data = json.loads(stdout.strip().split("\n")[-1])
                    return EvalResult(
                        score=data["score"],
                        passed=data["passed"],
                        details=data.get("details", {}),
                        stdout=stdout,
                    )
                except (json.JSONDecodeError, KeyError) as e:
                    return EvalResult(score=0, passed=False, error=f"Bad output: {e}", stdout=stdout)
            else:
                return EvalResult(score=0, passed=False, error=result.stderr, stderr=result.stderr)
    except Exception as e:
        return EvalResult(score=0, passed=False, error=str(e))


def main():
    parser = argparse.ArgumentParser(description="Sandbox Arena — evaluate agents")
    parser.add_argument("mode", choices=["local", "cluster"], help="Run locally or on AKS")
    parser.add_argument("arena", help="Arena name (trading, coding)")
    parser.add_argument("agent", help="Path to agent Python file")
    parser.add_argument("--parallel", type=int, default=5, help="Max parallel evaluations (cluster mode)")
    parser.add_argument("--template", default="arena-sandbox", help="SandboxTemplate name")
    parser.add_argument("--namespace", default="default", help="Kubernetes namespace")
    parser.add_argument("--timeout", type=int, default=30, help="Per-eval timeout in seconds")
    args = parser.parse_args()

    arena = load_arena(args.arena)
    scenarios = arena.scenarios()

    with open(args.agent) as f:
        agent_code = f.read()

    print(f"{'='*60}")
    print(f"Sandbox Arena — {arena.name}")
    print(f"{'='*60}")
    print(f"Agent:     {args.agent}")
    print(f"Mode:      {args.mode}")
    print(f"Scenarios: {len(scenarios)}")
    if args.mode == "cluster":
        print(f"Parallel:  {args.parallel}")
        print(f"Template:  {args.template}")
    print(f"{'='*60}\n")

    # Generate eval scripts
    scripts = [(arena.eval_script(agent_code, s), s) for s in scenarios]

    # Run evaluations
    start = time.time()
    results = []

    if args.mode == "local":
        for i, (script, scenario) in enumerate(scripts):
            result = run_local_eval(script, timeout=args.timeout)
            results.append(result)
            icon = "✅" if result.passed else ("💥" if result.error else "❌")
            label = scenario.get("ticker") or scenario.get("id") or f"#{i}"
            print(f"  {icon} {label:15s} score={result.score}")
    else:
        def run_one(item):
            idx, (script, scenario) = item
            return idx, scenario, run_cluster_eval(script, args.template, args.namespace)

        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = pool.map(run_one, enumerate(scripts))
            for idx, scenario, result in futures:
                results.append(result)
                icon = "✅" if result.passed else ("💥" if result.error else "❌")
                label = scenario.get("ticker") or scenario.get("id") or f"#{idx}"
                print(f"  {icon} {label:15s} score={result.score}")

    elapsed = time.time() - start

    # Aggregate
    summary = arena.aggregate(results)
    print(f"\n{'='*60}")
    print(f"Results ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  Passed:    {summary['passed']}/{summary['total']}")
    print(f"  Avg Score: {summary['avg_score']:.1f}")
    print(f"  Min/Max:   {summary['min_score']:.1f} / {summary['max_score']:.1f}")
    if summary["errors"]:
        print(f"  Errors:    {summary['errors']}")
    print(f"  Time:      {elapsed:.1f}s ({elapsed/len(results):.2f}s/eval)")


if __name__ == "__main__":
    main()
