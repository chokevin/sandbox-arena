"""Blackjack arena — train agents to play optimal blackjack.

Uses a simple built-in blackjack simulator (no external deps).
Each scenario is a different random seed for 100-hand sessions.
"""

import json

from arenas.base import Arena


SCENARIOS = [
    {"seed": 42, "hands": 100},
    {"seed": 123, "hands": 100},
    {"seed": 456, "hands": 100},
    {"seed": 789, "hands": 100},
    {"seed": 1337, "hands": 100},
]


class BlackjackArena(Arena):
    name = "blackjack"

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

def card_value(card):
    if card in ("J", "Q", "K"):
        return 10
    if card == "A":
        return 11
    return card

def hand_value(hand):
    total = sum(card_value(c) for c in hand)
    aces = sum(1 for c in hand if c == "A")
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total

def new_deck():
    cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, "J", "Q", "K", "A"] * 4
    random.shuffle(cards)
    return cards

wins = 0
losses = 0
draws = 0
total_reward = 0.0

for _ in range({scenario["hands"]}):
    deck = new_deck()
    player_hand = [deck.pop(), deck.pop()]
    dealer_hand = [deck.pop(), deck.pop()]

    # Player turn
    while hand_value(player_hand) < 21:
        try:
            action = play(hand_value(player_hand), card_value(dealer_hand[0]), len(player_hand))
        except Exception:
            action = "stand"
        if action == "hit":
            player_hand.append(deck.pop())
        else:
            break

    player_total = hand_value(player_hand)

    # Dealer turn (stands on 17+)
    if player_total <= 21:
        while hand_value(dealer_hand) < 17:
            dealer_hand.append(deck.pop())

    dealer_total = hand_value(dealer_hand)

    if player_total > 21:
        losses += 1
        total_reward -= 1
    elif dealer_total > 21:
        wins += 1
        total_reward += 1
    elif player_total > dealer_total:
        wins += 1
        total_reward += 1
    elif player_total < dealer_total:
        losses += 1
        total_reward -= 1
    else:
        draws += 1

total = wins + losses + draws
win_rate = wins / total * 100 if total > 0 else 0

result = {{
    "score": round(win_rate, 1),
    "passed": total_reward > 0,
    "details": {{
        "seed": {scenario["seed"]},
        "hands": total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(win_rate, 1),
        "net_reward": total_reward,
    }}
}}
print(json.dumps(result))
'''
