#!/usr/bin/env python3
"""Train a commodity trading agent using RL.

Trains a policy to trade Gold, Oil, Wheat, and Natural Gas
using REINFORCE with a deeper policy network.

Usage:
    python train_commodities.py --episodes 5000
    python train_commodities.py --episodes 10000 --lr 0.005
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path


class TradingPolicy:
    """Two-layer neural network policy for continuous action spaces."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64, lr: float = 0.005):
        self.lr = lr
        self.act_dim = act_dim

        # Layer 1: obs_dim → hidden
        self.w1 = np.random.randn(obs_dim, hidden) * np.sqrt(2.0 / obs_dim)
        self.b1 = np.zeros(hidden)

        # Layer 2: hidden → hidden
        self.w2 = np.random.randn(hidden, hidden) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(hidden)

        # Mean head: hidden → act_dim
        self.w_mu = np.random.randn(hidden, act_dim) * 0.01
        self.b_mu = np.zeros(act_dim)

        # Log-std (learnable)
        self.log_std = np.zeros(act_dim)

    def forward(self, obs: np.ndarray):
        """Forward pass. Returns action mean and std."""
        h1 = np.tanh(obs @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        mu = np.tanh(h2 @ self.w_mu + self.b_mu)  # tanh → [-1, 1]
        std = np.exp(np.clip(self.log_std, -2, 0.5))
        return mu, std, h1, h2

    def select_action(self, obs: np.ndarray):
        mu, std, _, _ = self.forward(obs)
        noise = np.random.randn(self.act_dim) * std
        action = np.clip(mu + noise, -1, 1)
        return action, mu, std

    def update(self, trajectories: list):
        """REINFORCE with baseline subtraction."""
        # Compute all returns
        all_returns = []
        for traj in trajectories:
            G = 0
            returns = []
            for r in reversed(traj["rewards"]):
                G = r + 0.99 * G
                returns.insert(0, G)
            all_returns.extend(returns)

        if not all_returns:
            return

        baseline = np.mean(all_returns)
        std_returns = np.std(all_returns) + 1e-8

        # Numerical gradient approximation (simple but effective)
        param_groups = [
            ("w1", self.w1), ("b1", self.b1),
            ("w2", self.w2), ("b2", self.b2),
            ("w_mu", self.w_mu), ("b_mu", self.b_mu),
            ("log_std", self.log_std),
        ]

        for traj in trajectories:
            G = 0
            for t in range(len(traj["rewards"]) - 1, -1, -1):
                G = traj["rewards"][t] + 0.99 * G
                advantage = (G - baseline) / std_returns

                obs = traj["observations"][t]
                action = traj["actions"][t]
                mu, std, h1, h2 = self.forward(obs)

                # Policy gradient for Gaussian policy
                diff = (action - mu) / (std ** 2 + 1e-10)

                # Backprop through mean head
                d_mu = diff * advantage * self.lr / len(trajectories)
                self.w_mu += np.outer(h2, d_mu)
                self.b_mu += d_mu

                # Backprop through log_std
                d_log_std = ((action - mu) ** 2 / (std ** 2 + 1e-10) - 1) * advantage
                self.log_std += d_log_std * self.lr * 0.1 / len(trajectories)
                self.log_std = np.clip(self.log_std, -2, 0.5)

    def save(self, path: str):
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
                 w_mu=self.w_mu, b_mu=self.b_mu, log_std=self.log_std)

    def load(self, path: str):
        data = np.load(path)
        self.w1, self.b1 = data["w1"], data["b1"]
        self.w2, self.b2 = data["w2"], data["b2"]
        self.w_mu, self.b_mu = data["w_mu"], data["b_mu"]
        self.log_std = data["log_std"]


def collect_episode(env, policy):
    obs, _ = env.reset()
    observations, actions, rewards = [], [], []
    done = False
    while not done:
        action, mu, std = policy.select_action(obs)
        observations.append(obs.copy())
        actions.append(action.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "total_reward": sum(rewards),
        "info": info,
    }


def main():
    parser = argparse.ArgumentParser(description="Train commodity trading agent")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--trading-days", type=int, default=252)
    args = parser.parse_args()

    from arenas.commodities.gym_env import CommodityTradingEnv

    env = CommodityTradingEnv(n_days=args.trading_days)
    eval_env = CommodityTradingEnv(n_days=args.trading_days)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = TradingPolicy(obs_dim, act_dim, hidden=args.hidden, lr=args.lr)

    print(f"{'='*70}")
    print(f"Commodity Trading RL")
    print(f"{'='*70}")
    print(f"  Commodities:  Gold, Oil, Wheat, Natural Gas")
    print(f"  Trading days: {args.trading_days}")
    print(f"  Episodes:     {args.episodes}")
    print(f"  Batch:        {args.batch}")
    print(f"  Hidden dim:   {args.hidden}")
    print(f"  Obs dim:      {obs_dim}")
    print(f"  Act dim:      {act_dim}")
    print(f"{'='*70}\n")

    start = time.time()
    history = []

    for ep in range(0, args.episodes, args.batch):
        batch = [collect_episode(env, policy) for _ in range(args.batch)]
        policy.update(batch)

        if (ep + args.batch) % args.eval_interval == 0 or ep == 0:
            eval_results = [collect_episode(eval_env, policy) for _ in range(20)]

            returns = [e["info"].get("total_return", 0) for e in eval_results]
            sharpes = [e["info"].get("sharpe", 0) for e in eval_results]
            drawdowns = [e["info"].get("drawdown", 0) for e in eval_results]

            avg_return = np.mean(returns)
            avg_sharpe = np.mean(sharpes)
            avg_dd = np.mean(drawdowns)

            # Visual bar for returns
            bar_len = int(np.clip((avg_return + 20) * 1.25, 0, 50))
            bar = "█" * bar_len + "░" * (50 - bar_len)

            print(f"  Ep {ep+args.batch:>5d} | ret={avg_return:>7.1f}% | "
                  f"sharpe={avg_sharpe:>5.2f} | dd={avg_dd:>5.1f}% | {bar}")

            history.append({
                "episode": ep + args.batch,
                "avg_return": round(avg_return, 2),
                "avg_sharpe": round(avg_sharpe, 3),
                "avg_drawdown": round(avg_dd, 2),
                "std_return": round(np.std(returns), 2),
            })

    elapsed = time.time() - start

    # Final evaluation
    print(f"\n{'='*70}")
    print(f"Final Evaluation (50 episodes)")
    print(f"{'='*70}")

    final_results = [collect_episode(eval_env, policy) for _ in range(50)]
    returns = [e["info"].get("total_return", 0) for e in final_results]
    sharpes = [e["info"].get("sharpe", 0) for e in final_results]
    drawdowns = [e["info"].get("drawdown", 0) for e in final_results]
    final_values = [e["info"].get("final_value", 100000) for e in final_results]

    print(f"  Avg Return:    {np.mean(returns):>7.2f}% ± {np.std(returns):.2f}%")
    print(f"  Avg Sharpe:    {np.mean(sharpes):>7.3f} ± {np.std(sharpes):.3f}")
    print(f"  Avg Drawdown:  {np.mean(drawdowns):>7.2f}%")
    print(f"  Avg Final Val: ${np.mean(final_values):>,.0f}")
    print(f"  Best Return:   {np.max(returns):>7.2f}%")
    print(f"  Worst Return:  {np.min(returns):>7.2f}%")
    print(f"  Win Rate:      {sum(1 for r in returns if r > 0) / len(returns) * 100:.0f}%")
    print(f"  Training Time: {elapsed:.1f}s")

    # Compare to buy-and-hold baseline
    print(f"\n  vs Buy-and-Hold baseline:")
    baseline_returns = []
    for _ in range(50):
        benv = CommodityTradingEnv(n_days=args.trading_days)
        obs, _ = benv.reset()
        # Buy equal weight on day 1, hold
        action = np.ones(4) * 1.0  # max long everything
        done = False
        while not done:
            obs, r, terminated, truncated, info = benv.step(action)
            action = np.ones(4) * 1.0  # keep holding
            done = terminated or truncated
        baseline_returns.append(info.get("total_return", 0))

    print(f"  Buy-Hold Avg:  {np.mean(baseline_returns):>7.2f}%")
    print(f"  RL Agent Avg:  {np.mean(returns):>7.2f}%")
    diff = np.mean(returns) - np.mean(baseline_returns)
    print(f"  Alpha:         {diff:>+7.2f}%")

    # Save
    policy.save("agents/trained_commodity_policy")
    Path("trajectories").mkdir(exist_ok=True)
    with open(f"trajectories/commodity_training_{int(time.time())}.json", "w") as f:
        json.dump({
            "episodes": args.episodes,
            "history": history,
            "final_avg_return": round(np.mean(returns), 2),
            "final_avg_sharpe": round(np.mean(sharpes), 3),
            "baseline_avg_return": round(np.mean(baseline_returns), 2),
            "alpha": round(diff, 2),
        }, f, indent=2)

    print(f"\n  Policy saved: agents/trained_commodity_policy.npz")


if __name__ == "__main__":
    main()
