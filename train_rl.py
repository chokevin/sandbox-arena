#!/usr/bin/env python3
"""Train an RL policy using Gym environments.

This trains a neural network policy (not an LLM) on the trading or
blackjack environments using simple policy gradient (REINFORCE).
No external RL library required — pure numpy + stdlib.

For production training, swap this for stable-baselines3 PPO or Ray RLlib.

Usage:
    python train_rl.py --env trading --episodes 1000
    python train_rl.py --env blackjack --episodes 5000
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path


class SimplePolicy:
    """Simple linear policy with softmax action selection."""

    def __init__(self, obs_dim: int, act_dim: int, lr: float = 0.01):
        self.weights = np.random.randn(obs_dim, act_dim) * 0.01
        self.bias = np.zeros(act_dim)
        self.lr = lr

    def forward(self, obs: np.ndarray) -> np.ndarray:
        logits = obs @ self.weights + self.bias
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()

    def select_action(self, obs: np.ndarray) -> int:
        probs = self.forward(obs)
        return np.random.choice(len(probs), p=probs)

    def update(self, trajectories: list):
        """REINFORCE policy gradient update."""
        grad_w = np.zeros_like(self.weights)
        grad_b = np.zeros_like(self.bias)

        for traj in trajectories:
            # Compute returns with discount
            returns = []
            G = 0
            for r in reversed(traj["rewards"]):
                G = r + 0.99 * G
                returns.insert(0, G)
            returns = np.array(returns)

            # Normalize returns
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            for obs, action, G in zip(traj["observations"], traj["actions"], returns):
                probs = self.forward(obs)
                # Policy gradient: ∇log(π(a|s)) * G
                one_hot = np.zeros(len(probs))
                one_hot[action] = 1
                grad = np.outer(obs, (one_hot - probs)) * G
                grad_w += grad
                grad_b += (one_hot - probs) * G

        # Average over trajectories
        n = max(len(trajectories), 1)
        self.weights += self.lr * grad_w / n
        self.bias += self.lr * grad_b / n

    def save(self, path: str):
        np.savez(path, weights=self.weights, bias=self.bias)

    def load(self, path: str):
        data = np.load(path)
        self.weights = data["weights"]
        self.bias = data["bias"]


def collect_episode(env, policy) -> dict:
    """Run one episode and collect trajectory."""
    obs, _ = env.reset()
    observations, actions, rewards = [], [], []

    done = False
    while not done:
        action = policy.select_action(obs)
        observations.append(obs.copy())
        actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "total_reward": sum(rewards),
    }


def main():
    parser = argparse.ArgumentParser(description="Train RL policy on gym environments")
    parser.add_argument("--env", default="trading", choices=["trading", "blackjack"])
    parser.add_argument("--episodes", type=int, default=1000, help="Total training episodes")
    parser.add_argument("--batch", type=int, default=20, help="Episodes per update")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--eval-interval", type=int, default=100, help="Evaluate every N episodes")
    args = parser.parse_args()

    # Create environment
    if args.env == "trading":
        from gym_env import TradingGymEnv
        env = TradingGymEnv()
        eval_env = TradingGymEnv()
    else:
        from gym_env import BlackjackGymEnv
        env = BlackjackGymEnv()
        eval_env = BlackjackGymEnv(seed=999)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    policy = SimplePolicy(obs_dim, act_dim, lr=args.lr)

    print(f"{'='*60}")
    print(f"RL Training — {args.env}")
    print(f"{'='*60}")
    print(f"  Obs dim:    {obs_dim}")
    print(f"  Act dim:    {act_dim}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Batch size: {args.batch}")
    print(f"  LR:         {args.lr}")
    print(f"{'='*60}\n")

    start = time.time()
    history = []

    for ep in range(0, args.episodes, args.batch):
        # Collect batch of trajectories
        batch = [collect_episode(env, policy) for _ in range(args.batch)]

        # Update policy
        policy.update(batch)

        avg_reward = sum(t["total_reward"] for t in batch) / len(batch)

        # Evaluate
        if (ep + args.batch) % args.eval_interval == 0 or ep == 0:
            eval_rewards = []
            for _ in range(50):
                traj = collect_episode(eval_env, policy)
                eval_rewards.append(traj["total_reward"])

            eval_avg = np.mean(eval_rewards)
            eval_std = np.std(eval_rewards)

            bar_len = int(max(0, min(50, (eval_avg + 1) * 25)))
            bar = "█" * bar_len + "░" * (50 - bar_len)

            print(f"  Ep {ep+args.batch:>5d} | train={avg_reward:>7.3f} | "
                  f"eval={eval_avg:>7.3f}±{eval_std:.3f} | {bar}")

            history.append({
                "episode": ep + args.batch,
                "train_reward": round(avg_reward, 4),
                "eval_reward": round(eval_avg, 4),
                "eval_std": round(eval_std, 4),
            })

    elapsed = time.time() - start

    # Save policy
    policy_path = f"agents/trained_{args.env}_policy"
    Path("agents").mkdir(exist_ok=True)
    policy.save(policy_path)

    # Save history
    history_path = Path("trajectories") / f"rl_training_{args.env}_{int(time.time())}.json"
    history_path.parent.mkdir(exist_ok=True)
    with open(history_path, "w") as f:
        json.dump({"env": args.env, "episodes": args.episodes, "history": history}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training Complete ({elapsed:.1f}s)")
    print(f"{'='*60}")
    if history:
        print(f"  Final eval: {history[-1]['eval_reward']:.3f}")
        print(f"  First eval: {history[0]['eval_reward']:.3f}")
        improvement = history[-1]["eval_reward"] - history[0]["eval_reward"]
        print(f"  Improvement: {improvement:+.3f}")
    print(f"  Policy saved: {policy_path}.npz")
    print(f"  History: {history_path}")


if __name__ == "__main__":
    main()
