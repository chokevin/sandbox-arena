#!/usr/bin/env python3
"""Tournament — evaluate multiple agents across multiple arenas in parallel.

This is the key differentiator over MLE-Dojo: scale.
MLE-Dojo runs 1 agent × 1 challenge sequentially.
sandbox-arena runs N agents × M challenges in parallel across sandbox pods.

Usage:
    # Run all example agents against all arenas
    python tournament.py --mode cluster --parallel 50

    # Specific agents and arenas
    python tournament.py --agents agents/examples/momentum.py agents/examples/buy_and_hold.py \
                         --arenas trading --mode cluster
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentResult:
    agent_name: str
    arena: str
    scenario_id: str
    score: float
    passed: bool
    error: str = ""


def load_arena(name: str):
    if name == "trading":
        from arenas.trading.arena import TradingArena
        return TradingArena()
    elif name == "coding":
        from arenas.coding.arena import CodingArena
        return CodingArena()
    elif name == "coding_hard":
        from arenas.coding_hard.arena import HardCodingArena
        return HardCodingArena()
    elif name == "blackjack":
        from arenas.blackjack.arena import BlackjackArena
        return BlackjackArena()
    elif name == "survival":
        from arenas.survival.arena import SurvivalArena
        return SurvivalArena()
    else:
        raise ValueError(f"Unknown arena: {name}")


def discover_agents(agent_paths: list[str]) -> dict[str, str]:
    """Load agent files. Returns {name: code}."""
    agents = {}
    for path in agent_paths:
        p = Path(path)
        if p.is_dir():
            for f in sorted(p.glob("*.py")):
                agents[f.stem] = f.read_text()
        elif p.is_file():
            agents[p.stem] = p.read_text()
    return agents


def eval_single(arena, agent_name: str, agent_code: str,
                scenario: dict, mode: str, template: str, namespace: str) -> AgentResult:
    """Evaluate one agent on one scenario."""
    scenario_id = scenario.get("id") or scenario.get("ticker") or scenario.get("seed", "?")
    script = arena.eval_script(agent_code, scenario)

    try:
        if mode == "local":
            import subprocess
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(script)
                f.flush()
                proc = subprocess.run(
                    [sys.executable, f.name],
                    capture_output=True, text=True, timeout=30
                )
                Path(f.name).unlink(missing_ok=True)
                if proc.returncode == 0:
                    data = json.loads(proc.stdout.strip().split("\n")[-1])
                    return AgentResult(agent_name, arena.name, str(scenario_id),
                                      data["score"], data["passed"])
                else:
                    return AgentResult(agent_name, arena.name, str(scenario_id),
                                      0, False, error=proc.stderr[:200])
        else:
            from k8s_agent_sandbox import SandboxClient
            with SandboxClient(template_name=template, namespace=namespace) as sandbox:
                sandbox.write("eval.py", script)
                result = sandbox.run("python3 eval.py", timeout=60)
                if result.exit_code == 0:
                    data = json.loads(result.stdout.strip().split("\n")[-1])
                    return AgentResult(agent_name, arena.name, str(scenario_id),
                                      data["score"], data["passed"])
                else:
                    return AgentResult(agent_name, arena.name, str(scenario_id),
                                      0, False, error=result.stderr[:200])
    except Exception as e:
        return AgentResult(agent_name, arena.name, str(scenario_id),
                          0, False, error=str(e)[:200])


def print_leaderboard(results: list[AgentResult], arena_name: str):
    """Print a leaderboard for one arena."""
    # Group by agent
    agent_scores = {}
    for r in results:
        if r.arena != arena_name:
            continue
        if r.agent_name not in agent_scores:
            agent_scores[r.agent_name] = []
        agent_scores[r.agent_name].append(r)

    if not agent_scores:
        return

    # Rank by average score
    rankings = []
    for agent, scores in agent_scores.items():
        avg = sum(s.score for s in scores) / len(scores)
        passed = sum(1 for s in scores if s.passed)
        total = len(scores)
        errors = sum(1 for s in scores if s.error)
        rankings.append((avg, passed, total, errors, agent))

    rankings.sort(reverse=True)

    print(f"\n  {'─'*55}")
    print(f"  {'Rank':<6} {'Agent':<25} {'Avg Score':>10} {'Pass':>6} {'Err':>5}")
    print(f"  {'─'*55}")

    for i, (avg, passed, total, errors, agent) in enumerate(rankings, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        err_str = f"({errors}!)" if errors else ""
        print(f"  {medal}{i:<4} {agent:<25} {avg:>9.1f} {passed:>3}/{total} {err_str:>5}")

    print(f"  {'─'*55}")


def main():
    parser = argparse.ArgumentParser(description="Tournament — multi-agent parallel evaluation")
    parser.add_argument("--agents", nargs="+", default=["agents/examples"],
                        help="Agent files or directories")
    parser.add_argument("--arenas", nargs="+",
                        default=["trading", "coding", "blackjack", "survival"],
                        help="Arenas to compete in")
    parser.add_argument("--mode", default="cluster", choices=["local", "cluster"])
    parser.add_argument("--parallel", type=int, default=10,
                        help="Max concurrent evaluations")
    parser.add_argument("--template", default="arena-sandbox")
    parser.add_argument("--namespace", default="default")
    args = parser.parse_args()

    # Discover agents
    agents = discover_agents(args.agents)
    if not agents:
        print("No agents found!")
        sys.exit(1)

    # Build evaluation tasks
    tasks = []
    for arena_name in args.arenas:
        arena = load_arena(arena_name)
        for scenario in arena.scenarios():
            for agent_name, agent_code in agents.items():
                # Skip agents incompatible with arena
                if arena_name in ("trading",) and "strategy" not in agent_code:
                    continue
                if arena_name in ("coding", "coding_hard") and "def fib" not in agent_code and "def play" not in agent_code and "def survive" not in agent_code and "def strategy" not in agent_code:
                    # Coding agents define the solution functions directly
                    if arena_name == "coding" and not any(fn in agent_code for fn in ["fib", "fizzbuzz", "is_palindrome", "two_sum", "reverse_words"]):
                        continue
                if arena_name == "blackjack" and "def play" not in agent_code:
                    continue
                if arena_name == "survival" and "def survive" not in agent_code:
                    continue
                tasks.append((arena, agent_name, agent_code, scenario))

    total = len(tasks)
    print(f"{'='*60}")
    print(f"Tournament")
    print(f"{'='*60}")
    print(f"  Agents:      {len(agents)} ({', '.join(agents.keys())})")
    print(f"  Arenas:      {', '.join(args.arenas)}")
    print(f"  Evaluations: {total}")
    print(f"  Mode:        {args.mode}")
    print(f"  Parallel:    {args.parallel}")
    print(f"{'='*60}\n")

    # Execute all evaluations in parallel
    start = time.time()
    results: list[AgentResult] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(eval_single, arena, name, code, scenario,
                       args.mode, args.template, args.namespace): (name, arena.name)
            for arena, name, code, scenario in tasks
        }

        for future in as_completed(futures):
            agent_name, arena_name = futures[future]
            result = future.result()
            results.append(result)
            completed += 1

            icon = "✅" if result.passed else ("💥" if result.error else "❌")
            if completed % 10 == 0 or completed == total:
                print(f"  [{completed}/{total}] {icon} {result.agent_name} × {result.arena}:{result.scenario_id}")

    elapsed = time.time() - start

    # Leaderboards per arena
    print(f"\n{'='*60}")
    print(f"Leaderboards ({elapsed:.1f}s, {total} evaluations)")
    print(f"{'='*60}")

    for arena_name in args.arenas:
        arena_results = [r for r in results if r.arena == arena_name]
        if arena_results:
            print(f"\n  📊 {arena_name.upper()}")
            print_leaderboard(results, arena_name)

    # Overall rankings
    print(f"\n  📊 OVERALL")
    agent_overall = {}
    for r in results:
        if r.agent_name not in agent_overall:
            agent_overall[r.agent_name] = {"scores": [], "passed": 0, "total": 0}
        agent_overall[r.agent_name]["scores"].append(r.score)
        agent_overall[r.agent_name]["passed"] += int(r.passed)
        agent_overall[r.agent_name]["total"] += 1

    overall_ranked = sorted(
        agent_overall.items(),
        key=lambda x: sum(x[1]["scores"]) / len(x[1]["scores"]),
        reverse=True,
    )
    print(f"\n  {'─'*55}")
    print(f"  {'Rank':<6} {'Agent':<25} {'Avg Score':>10} {'Pass':>6}")
    print(f"  {'─'*55}")
    for i, (agent, data) in enumerate(overall_ranked, 1):
        avg = sum(data["scores"]) / len(data["scores"])
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"  {medal}{i:<4} {agent:<25} {avg:>9.1f} {data['passed']:>3}/{data['total']}")
    print(f"  {'─'*55}")

    # Save results
    results_path = Path("trajectories") / f"tournament_{int(time.time())}.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "elapsed": elapsed,
            "total_evals": total,
            "results": [
                {"agent": r.agent_name, "arena": r.arena,
                 "scenario": r.scenario_id, "score": r.score,
                 "passed": r.passed, "error": r.error}
                for r in results
            ],
        }, f, indent=2)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
