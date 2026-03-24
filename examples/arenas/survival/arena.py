"""Survival arena — agents manage resources to survive as long as possible.

A roguelike survival simulation: the agent starts with health, food, and energy.
Each turn, the agent chooses an action. Random events happen.
Goal: survive the most turns.
"""

import json

from arenas.base import Arena


SCENARIOS = [
    {"seed": 42, "max_turns": 200, "difficulty": "normal"},
    {"seed": 99, "max_turns": 200, "difficulty": "normal"},
    {"seed": 7, "max_turns": 200, "difficulty": "hard"},
    {"seed": 256, "max_turns": 200, "difficulty": "hard"},
    {"seed": 1000, "max_turns": 200, "difficulty": "extreme"},
]


class SurvivalArena(Arena):
    name = "survival"

    def scenarios(self) -> list[dict]:
        return SCENARIOS

    def eval_script(self, agent_code: str, scenario: dict) -> str:
        return f'''
import json
import random

random.seed({scenario["seed"]})

# --- Agent code (untrusted) ---
{agent_code}
# --- End agent code ---

health = 100
food = 50
energy = 50
shelter = 0
turn = 0
max_turns = {scenario["max_turns"]}

diff_mult = {{"normal": 1.0, "hard": 1.5, "extreme": 2.0}}["{scenario["difficulty"]}"]
events = 0

while turn < max_turns and health > 0:
    turn += 1

    try:
        action = survive(health, food, energy, shelter, turn)
    except Exception:
        action = "rest"

    if action == "forage":
        food += random.randint(5, 20)
        energy -= 10
    elif action == "rest":
        energy = min(100, energy + 20)
        health = min(100, health + 5)
    elif action == "explore":
        energy -= 15
        if random.random() < 0.3:
            food += random.randint(10, 30)
        if random.random() < 0.1:
            shelter += 10
    elif action == "build":
        energy -= 20
        food -= 5
        shelter = min(100, shelter + 15)

    # Daily costs
    food -= int(3 * diff_mult)
    energy -= int(2 * diff_mult)

    if food <= 0:
        health -= int(10 * diff_mult)
        food = 0
    if energy <= 0:
        health -= int(5 * diff_mult)
        energy = 0

    # Random events
    roll = random.random()
    if roll < 0.05 * diff_mult:
        dmg = random.randint(10, 30)
        block = min(dmg, shelter // 2)
        health -= (dmg - block)
        events += 1
    elif roll < 0.08 * diff_mult:
        food = max(0, food - random.randint(5, 15))
        events += 1
    elif roll > 0.95:
        food += random.randint(10, 25)
        events += 1

    health = max(0, health)

survival_pct = turn / max_turns * 100

result = {{
    "score": round(survival_pct, 1),
    "passed": turn >= max_turns,
    "details": {{
        "seed": {scenario["seed"]},
        "difficulty": "{scenario["difficulty"]}",
        "survived_turns": turn,
        "max_turns": max_turns,
        "final_health": health,
        "events": events,
    }}
}}
print(json.dumps(result))
'''
