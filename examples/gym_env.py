"""Gym-compatible environment wrapper for RL training.

This wraps any sandbox-arena Arena as a standard Gymnasium environment,
enabling training with stable-baselines3, Ray RLlib, CleanRL, or any
RL framework that uses the Gym interface.

For LLM agents: use dojo.py (multi-turn text interface).
For RL policies (PPO, SAC, DQN): use this gym wrapper.

The observation is the scenario state, the action is the agent's decision,
and the reward comes from the arena's scoring.

Usage with stable-baselines3:
    from gym_env import SandboxGymEnv
    from stable_baselines3 import PPO

    env = SandboxGymEnv(arena="trading")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)
"""

import json
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


class TradingGymEnv(gym.Env):
    """Gym environment for the trading arena.

    Observation: [current_price, ma_5, ma_10, position, cash_ratio, day_ratio]
    Action: 0=hold, 1=buy, 2=sell
    Reward: daily portfolio return
    """

    metadata = {"render_modes": []}

    def __init__(self, prices=None, initial_cash=10000.0):
        super().__init__()

        if prices is None:
            # Default price series (can be overridden)
            from arenas.trading.arena import SAMPLE_SCENARIOS
            self.all_scenarios = SAMPLE_SCENARIOS
            prices = self.all_scenarios[0]["prices"]
        else:
            self.all_scenarios = [{"ticker": "custom", "prices": prices}]

        self.prices = prices
        self.initial_cash = initial_cash

        # Action: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)

        # Observation: [price_norm, ma5_ratio, ma10_ratio, has_position, cash_ratio, progress]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Optionally cycle through scenarios
        if hasattr(self, "scenario_idx"):
            self.scenario_idx = (self.scenario_idx + 1) % len(self.all_scenarios)
        else:
            self.scenario_idx = 0

        self.prices = self.all_scenarios[self.scenario_idx]["prices"]
        self.day = 0
        self.cash = self.initial_cash
        self.position = 0
        self.initial_portfolio = self.initial_cash

        return self._get_obs(), {}

    def _get_obs(self):
        price = self.prices[self.day]
        # Moving averages
        window5 = self.prices[max(0, self.day-4):self.day+1]
        window10 = self.prices[max(0, self.day-9):self.day+1]
        ma5 = sum(window5) / len(window5)
        ma10 = sum(window10) / len(window10)

        portfolio = self.cash + self.position * price
        return np.array([
            price / self.prices[0],           # normalized price
            price / ma5 if ma5 > 0 else 1.0,  # price vs MA5
            price / ma10 if ma10 > 0 else 1.0, # price vs MA10
            float(self.position > 0),          # has position
            self.cash / self.initial_cash,     # cash ratio
            self.day / len(self.prices),       # progress
        ], dtype=np.float32)

    def step(self, action):
        price = self.prices[self.day]
        prev_portfolio = self.cash + self.position * price

        # Execute action
        if action == 1 and self.cash >= price:  # buy
            shares = int(self.cash // price)
            if shares > 0:
                self.position += shares
                self.cash -= shares * price
        elif action == 2 and self.position > 0:  # sell
            self.cash += self.position * price
            self.position = 0

        self.day += 1
        done = self.day >= len(self.prices) - 1

        # Reward = portfolio return this step
        new_price = self.prices[self.day] if not done else self.prices[-1]
        new_portfolio = self.cash + self.position * new_price
        reward = (new_portfolio - prev_portfolio) / prev_portfolio

        # Bonus reward at end for overall performance
        if done:
            total_return = (new_portfolio - self.initial_cash) / self.initial_cash
            reward += total_return

        return self._get_obs(), reward, done, False, {
            "portfolio": new_portfolio,
            "cash": self.cash,
            "position": self.position,
            "day": self.day,
        }


class BlackjackGymEnv(gym.Env):
    """Gym environment for the blackjack arena.

    Observation: [hand_total_norm, dealer_showing_norm, num_cards_norm]
    Action: 0=stand, 1=hit
    Reward: +1 win, -1 loss, 0 draw
    """

    metadata = {"render_modes": []}

    def __init__(self, seed=42):
        super().__init__()
        import random
        self.rng = random.Random(seed)

        self.action_space = spaces.Discrete(2)  # 0=stand, 1=hit
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3,), dtype=np.float32
        )
        self.reset()

    def _card_value(self, card):
        if card in ("J", "Q", "K"):
            return 10
        if card == "A":
            return 11
        return card

    def _hand_value(self, hand):
        total = sum(self._card_value(c) for c in hand)
        aces = sum(1 for c in hand if c == "A")
        while total > 21 and aces:
            total -= 10
            aces -= 1
        return total

    def _new_deck(self):
        cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, "J", "Q", "K", "A"] * 4
        self.rng.shuffle(cards)
        return cards

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = __import__("random").Random(seed)

        self.deck = self._new_deck()
        self.player_hand = [self.deck.pop(), self.deck.pop()]
        self.dealer_hand = [self.deck.pop(), self.deck.pop()]
        self.done = False

        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self._hand_value(self.player_hand) / 21.0,
            self._card_value(self.dealer_hand[0]) / 11.0,
            len(self.player_hand) / 10.0,
        ], dtype=np.float32)

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        if action == 1:  # hit
            self.player_hand.append(self.deck.pop())
            if self._hand_value(self.player_hand) > 21:
                self.done = True
                return self._get_obs(), -1.0, True, False, {"result": "bust"}
            return self._get_obs(), 0, False, False, {}

        # Stand — dealer plays
        self.done = True
        player_total = self._hand_value(self.player_hand)

        while self._hand_value(self.dealer_hand) < 17:
            self.dealer_hand.append(self.deck.pop())

        dealer_total = self._hand_value(self.dealer_hand)

        if dealer_total > 21:
            reward = 1.0
            result = "dealer_bust"
        elif player_total > dealer_total:
            reward = 1.0
            result = "win"
        elif player_total < dealer_total:
            reward = -1.0
            result = "loss"
        else:
            reward = 0.0
            result = "draw"

        return self._get_obs(), reward, True, False, {"result": result}
