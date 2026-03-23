#!/usr/bin/env python3
"""RL Training Loop — iteratively improve a trading strategy using sandbox evaluation.

This demonstrates the core RL training pattern:
1. Start with a parameterized strategy
2. Generate N variants (mutate parameters)
3. Evaluate each variant in a sandbox (parallel, isolated)
4. Select the best performers
5. Mutate from the winners → next generation
6. Repeat until convergence

This uses evolutionary strategies (ES) rather than gradient-based RL
because the strategy is a simple Python function with tunable parameters,
not a neural network. The pattern is the same though — the sandbox
execution step is identical whether you're doing ES, PPO, or any other RL.

Usage:
    # Local mode (no cluster)
    python train.py --arena trading --generations 20 --population 10

    # Cluster mode (AKS + Agent Sandbox)
    python train.py --arena trading --generations 20 --population 50 --mode cluster
"""

import argparse
import copy
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from arenas.base import EvalResult


def load_arena(arena_name: str):
    if arena_name == "trading":
        from arenas.trading.arena import TradingArena
        return TradingArena()
    elif arena_name == "coding":
        from arenas.coding.arena import CodingArena
        return CodingArena()
    else:
        print(f"Unknown arena: {arena_name}")
        sys.exit(1)


# --- Parameterized strategy template ---
# The RL loop mutates these parameters each generation

STRATEGY_TEMPLATE = '''
def strategy(prices, position, cash):
    """Parameterized trading strategy."""
    if len(prices) < {lookback}:
        return "hold"

    window = prices[-{lookback}:]
    current = prices[-1]
    ma = sum(window) / len(window)

    # Momentum signal
    momentum = (current - prices[-{lookback}]) / prices[-{lookback}]

    # Volatility signal
    variance = sum((p - ma) ** 2 for p in window) / len(window)
    vol = variance ** 0.5
    vol_ratio = vol / ma if ma > 0 else 0

    if momentum > {buy_threshold} and vol_ratio < {max_vol}:
        return "buy" if cash > 0 else "hold"
    elif momentum < {sell_threshold} or vol_ratio > {panic_vol}:
        return "sell" if position > 0 else "hold"
    return "hold"
'''


def random_params():
    """Generate random strategy parameters."""
    return {
        "lookback": random.randint(3, 15),
        "buy_threshold": round(random.uniform(0.01, 0.15), 4),
        "sell_threshold": round(random.uniform(-0.15, -0.01), 4),
        "max_vol": round(random.uniform(0.02, 0.10), 4),
        "panic_vol": round(random.uniform(0.08, 0.20), 4),
    }


def mutate_params(params: dict, mutation_rate: float = 0.3) -> dict:
    """Mutate parameters slightly."""
    new = copy.deepcopy(params)
    for key in new:
        if random.random() < mutation_rate:
            if key == "lookback":
                new[key] = max(2, new[key] + random.randint(-2, 2))
            else:
                delta = new[key] * random.uniform(-0.3, 0.3)
                new[key] = round(new[key] + delta, 4)
    return new


def params_to_code(params: dict) -> str:
    """Convert parameters to executable strategy code."""
    return STRATEGY_TEMPLATE.format(**params)


def evaluate_agent_local(arena, agent_code: str, timeout: int = 30) -> float:
    """Evaluate an agent locally, return average score across scenarios."""
    import subprocess
    import tempfile
    from pathlib import Path

    scores = []
    for scenario in arena.scenarios():
        script = arena.eval_script(agent_code, scenario)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            f.flush()
            try:
                result = subprocess.run(
                    [sys.executable, f.name],
                    capture_output=True, text=True, timeout=timeout
                )
                if result.returncode == 0:
                    data = json.loads(result.stdout.strip().split("\n")[-1])
                    scores.append(data["score"])
                else:
                    scores.append(-100)  # Penalize crashes
            except (subprocess.TimeoutExpired, Exception):
                scores.append(-100)
            finally:
                Path(f.name).unlink(missing_ok=True)

    return sum(scores) / len(scores) if scores else -100


def evaluate_agent_cluster(arena, agent_code: str, template: str, namespace: str) -> float:
    """Evaluate an agent in a sandbox pod, return average score."""
    from k8s_agent_sandbox import SandboxClient

    scores = []
    for scenario in arena.scenarios():
        script = arena.eval_script(agent_code, scenario)
        try:
            with SandboxClient(template_name=template, namespace=namespace) as sandbox:
                sandbox.write("eval.py", script)
                result = sandbox.run("python3 eval.py", timeout=60)
                if result.exit_code == 0:
                    data = json.loads(result.stdout.strip().split("\n")[-1])
                    scores.append(data["score"])
                else:
                    scores.append(-100)
        except Exception:
            scores.append(-100)

    return sum(scores) / len(scores) if scores else -100


def main():
    parser = argparse.ArgumentParser(description="RL Training Loop with Sandbox Arena")
    parser.add_argument("--arena", default="trading", help="Arena to train on")
    parser.add_argument("--generations", type=int, default=20, help="Number of generations")
    parser.add_argument("--population", type=int, default=10, help="Population size per generation")
    parser.add_argument("--elite", type=int, default=3, help="Top N to keep each generation")
    parser.add_argument("--mode", default="local", choices=["local", "cluster"])
    parser.add_argument("--parallel", type=int, default=5, help="Max parallel evals")
    parser.add_argument("--template", default="arena-sandbox")
    parser.add_argument("--namespace", default="default")
    args = parser.parse_args()

    arena = load_arena(args.arena)

    print(f"{'='*60}")
    print(f"RL Training Loop — {arena.name}")
    print(f"{'='*60}")
    print(f"Generations:  {args.generations}")
    print(f"Population:   {args.population}")
    print(f"Elite:        {args.elite}")
    print(f"Mode:         {args.mode}")
    print(f"{'='*60}\n")

    # Initialize population with random parameters
    population = [random_params() for _ in range(args.population)]
    best_ever = {"score": -999, "params": None, "generation": 0}

    total_start = time.time()

    for gen in range(1, args.generations + 1):
        gen_start = time.time()

        # Evaluate all agents in this generation
        if args.mode == "local":
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                codes = [params_to_code(p) for p in population]
                scores = list(pool.map(
                    lambda c: evaluate_agent_local(arena, c),
                    codes
                ))
        else:
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                codes = [params_to_code(p) for p in population]
                scores = list(pool.map(
                    lambda c: evaluate_agent_cluster(arena, c, args.template, args.namespace),
                    codes
                ))

        # Rank by score
        ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        gen_best_score = ranked[0][0]
        gen_best_params = ranked[0][1]
        gen_avg = sum(s for s, _ in ranked) / len(ranked)
        gen_elapsed = time.time() - gen_start

        # Track best ever
        if gen_best_score > best_ever["score"]:
            best_ever = {"score": gen_best_score, "params": gen_best_params, "generation": gen}

        # Print generation summary
        bar = "█" * int(max(0, gen_best_score) / 2) + "░" * int(max(0, 50 - gen_best_score) / 2)
        print(f"  Gen {gen:3d} | best={gen_best_score:7.2f} avg={gen_avg:7.2f} | {bar} | {gen_elapsed:.1f}s")

        # Select elite and breed next generation
        elite = [params for _, params in ranked[:args.elite]]
        next_gen = list(elite)  # Keep elite unchanged
        while len(next_gen) < args.population:
            parent = random.choice(elite)
            child = mutate_params(parent)
            next_gen.append(child)

        population = next_gen

    total_elapsed = time.time() - total_start

    # Final results
    print(f"\n{'='*60}")
    print(f"Training Complete ({total_elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"Best score:      {best_ever['score']:.2f}")
    print(f"Best generation: {best_ever['generation']}")
    print(f"Best parameters:")
    for k, v in best_ever["params"].items():
        print(f"  {k:20s}: {v}")

    print(f"\nBest strategy code:")
    print("-" * 40)
    print(params_to_code(best_ever["params"]))
    print("-" * 40)

    # Save best agent
    best_code = params_to_code(best_ever["params"])
    with open("agents/trained_best.py", "w") as f:
        f.write(best_code)
    print(f"\nSaved best agent to agents/trained_best.py")
    print(f"Evaluate it: python run.py local trading agents/trained_best.py")


if __name__ == "__main__":
    main()
