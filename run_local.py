#!/usr/bin/env python3
"""Run Pokemon battles locally — no cluster needed.

Usage:
    python run_local.py --agent agents/example_agent.py --battles 10
"""

import argparse
import asyncio
import importlib.util
import sys

from poke_env.player import RandomPlayer


def load_agent_class(path: str):
    """Dynamically load a Player subclass from a Python file."""
    spec = importlib.util.spec_from_file_location("agent_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the first Player subclass that isn't Player itself
    from poke_env.player import Player
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (isinstance(attr, type)
                and issubclass(attr, Player)
                and attr is not Player
                and attr is not RandomPlayer):
            return attr

    raise ValueError(f"No Player subclass found in {path}")


async def main():
    parser = argparse.ArgumentParser(description="Run Pokemon battles locally")
    parser.add_argument("--agent", required=True, help="Path to agent Python file")
    parser.add_argument("--battles", type=int, default=10, help="Number of battles")
    parser.add_argument("--opponent", default="random",
                        choices=["random", "max_damage", "heuristic"],
                        help="Opponent type")
    args = parser.parse_args()

    AgentClass = load_agent_class(args.agent)

    # Create players
    agent = AgentClass(battle_format="gen9randombattle")

    if args.opponent == "max_damage":
        from opponents.max_damage_player import MaxDamagePlayer
        opponent = MaxDamagePlayer(battle_format="gen9randombattle")
    elif args.opponent == "heuristic":
        from opponents.heuristic_player import HeuristicPlayer
        opponent = HeuristicPlayer(battle_format="gen9randombattle")
    else:
        opponent = RandomPlayer(battle_format="gen9randombattle")

    print(f"Battle: {AgentClass.__name__} vs {opponent.__class__.__name__}")
    print(f"Running {args.battles} battles...")

    # Run battles
    await agent.battle_against(opponent, n_battles=args.battles)

    wins = agent.n_won_battles
    total = args.battles
    print(f"\nResults: {wins}/{total} wins ({100*wins/total:.1f}%)")
    print(f"Agent: {AgentClass.__name__}")
    print(f"Opponent: {opponent.__class__.__name__}")


if __name__ == "__main__":
    asyncio.run(main())
