#!/usr/bin/env python3
"""Run Pokemon battle evaluation at scale using Agent Sandbox.

Each battle runs in an isolated sandbox pod on Kubernetes.

Usage:
    python run_eval.py --agent agents/example_agent.py --battles-per-opponent 100

Requires:
    - agent-sandbox controller installed on your cluster
    - sandbox-router deployed
    - SandboxTemplate 'poke-battle-sandbox' applied
    - kubectl configured
"""

import argparse
import asyncio
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from k8s_agent_sandbox import SandboxClient


OPPONENTS = ["random", "max_damage", "heuristic"]


def run_battle_in_sandbox(agent_code: str, opponent: str, battle_id: int) -> dict:
    """Run a single battle inside a sandbox pod."""
    template = os.environ.get("SANDBOX_TEMPLATE", "poke-battle-sandbox")
    namespace = os.environ.get("SANDBOX_NAMESPACE", "default")

    battle_script = f'''
import asyncio
import importlib.util
import sys

# Write agent code
with open("/tmp/agent.py", "w") as f:
    f.write("""{agent_code}""")

# Load agent
spec = importlib.util.spec_from_file_location("agent_mod", "/tmp/agent.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from poke_env.player import Player, RandomPlayer
AgentClass = None
for name in dir(mod):
    cls = getattr(mod, name)
    if isinstance(cls, type) and issubclass(cls, Player) and cls is not Player and cls is not RandomPlayer:
        AgentClass = cls
        break

if not AgentClass:
    print("ERROR: No agent found")
    sys.exit(1)

# Create opponent
opponent_type = "{opponent}"
if opponent_type == "max_damage":
    class Opponent(Player):
        def choose_move(self, battle):
            if battle.available_moves:
                return self.create_order(max(battle.available_moves, key=lambda m: m.base_power))
            return self.choose_random_move(battle)
elif opponent_type == "heuristic":
    class Opponent(Player):
        def choose_move(self, battle):
            if battle.available_moves:
                opp = battle.opponent_active_pokemon
                def score(m):
                    p = m.base_power or 0
                    mult = opp.damage_multiplier(m) if opp else 1.0
                    return p * mult
                best = max(battle.available_moves, key=score)
                if score(best) > 0:
                    return self.create_order(best)
            return self.choose_random_move(battle)
else:
    Opponent = RandomPlayer

async def battle():
    agent = AgentClass(battle_format="gen9randombattle")
    opp = Opponent(battle_format="gen9randombattle")
    await agent.battle_against(opp, n_battles=1)
    won = agent.n_won_battles > 0
    print(f"RESULT:won={won}")

asyncio.run(battle())
'''

    try:
        with SandboxClient(
            template_name=template,
            namespace=namespace,
        ) as sandbox:
            sandbox.write("battle.py", battle_script)
            result = sandbox.run("python3 battle.py", timeout=120)
            stdout = result.stdout

            won = "RESULT:won=True" in stdout
            return {
                "battle_id": battle_id,
                "opponent": opponent,
                "won": won,
                "success": True,
                "output": stdout,
            }
    except Exception as e:
        return {
            "battle_id": battle_id,
            "opponent": opponent,
            "won": False,
            "success": False,
            "output": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent in sandboxed battles")
    parser.add_argument("--agent", required=True, help="Path to agent Python file")
    parser.add_argument("--battles-per-opponent", type=int, default=10,
                        help="Battles per opponent")
    parser.add_argument("--opponents", default="all",
                        help="Comma-separated opponents or 'all'")
    parser.add_argument("--parallel", type=int, default=10,
                        help="Max concurrent sandbox battles")
    args = parser.parse_args()

    with open(args.agent) as f:
        agent_code = f.read()

    opponents = OPPONENTS if args.opponents == "all" else args.opponents.split(",")
    total_battles = len(opponents) * args.battles_per_opponent

    print(f"=== Poke-Sandbox Evaluation ===")
    print(f"Agent: {args.agent}")
    print(f"Opponents: {', '.join(opponents)}")
    print(f"Battles per opponent: {args.battles_per_opponent}")
    print(f"Total battles: {total_battles}")
    print(f"Max parallel: {args.parallel}")
    print(f"===============================\n")

    # Build battle tasks
    tasks = []
    battle_id = 0
    for opponent in opponents:
        for _ in range(args.battles_per_opponent):
            tasks.append((agent_code, opponent, battle_id))
            battle_id += 1

    # Execute in parallel using thread pool
    start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(run_battle_in_sandbox, *t) for t in tasks]
        for i, future in enumerate(futures):
            result = future.result()
            results.append(result)
            status = "✅" if result["won"] else "❌"
            print(f"  [{i+1}/{total_battles}] vs {result['opponent']}: {status}")

    elapsed = time.time() - start

    # Aggregate results
    print(f"\n=== Results ({elapsed:.1f}s) ===")
    for opponent in opponents:
        opp_results = [r for r in results if r["opponent"] == opponent]
        wins = sum(1 for r in opp_results if r["won"])
        total = len(opp_results)
        errors = sum(1 for r in opp_results if not r["success"])
        print(f"  vs {opponent:15s}: {wins}/{total} wins ({100*wins/total:.1f}%)"
              + (f"  ({errors} errors)" if errors else ""))

    total_wins = sum(1 for r in results if r["won"])
    print(f"\n  Overall: {total_wins}/{total_battles} wins "
          f"({100*total_wins/total_battles:.1f}%)")
    print(f"  Time: {elapsed:.1f}s ({elapsed/total_battles:.2f}s/battle)")


if __name__ == "__main__":
    main()
