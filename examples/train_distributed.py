#!/usr/bin/env python3
"""Distributed commodity trading RL — episodes run in sandbox pods.

The training loop runs locally (or on a GPU node). Each episode is
dispatched to a sandbox pod from the warm pool, executed in isolation,
and rewards are collected back for policy update.

This is the key advantage over local training:
  - Local: 10 episodes/batch × 3s each = 30s per update
  - Sandbox: 50 episodes/batch × 3s each IN PARALLEL = 3s per update

10x faster training through parallelism.

Usage:
    # Requires agent-sandbox deployed on AKS
    kubectl apply -f sandbox/

    python train_distributed.py --episodes 5000 --batch 50 --parallel 50
"""

import argparse
import json
import time
import base64
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# Inline the environment + policy into a script that runs inside the sandbox
EPISODE_SCRIPT_TEMPLATE = '''
import json
import numpy as np
import sys

# === Embedded environment ===
COMMODITIES = {commodities_json}
START_PRICES = {start_prices_json}

def generate_prices(params, start_price, n_days, rng):
    dt = 1 / 252
    prices = np.zeros(n_days)
    prices[0] = start_price
    vol = params["sigma"]
    long_term = start_price
    for i in range(1, n_days):
        mr = params["mean_rev"] * (np.log(long_term) - np.log(prices[i-1])) * dt
        if i > 1:
            ret = np.log(prices[i-1] / prices[i-2])
            vol = 0.9 * vol + 0.1 * abs(ret) * np.sqrt(252)
            vol = np.clip(vol, params["sigma"] * 0.5, params["sigma"] * 2.0)
        drift = (params["mu"] - 0.5 * vol**2) * dt + mr
        shock = vol * np.sqrt(dt) * rng.standard_normal()
        prices[i] = prices[i-1] * np.exp(drift + shock)
    return prices

def compute_features(prices, idx):
    lookback = min(idx, 20) if idx >= 20 else max(idx, 1)
    window = prices[max(0, idx-lookback+1):idx+1]
    current = prices[idx]
    ret_1d = (prices[idx] / prices[max(0, idx-1)] - 1) if idx > 0 else 0
    ret_5d = (prices[idx] / prices[max(0, idx-5)] - 1) if idx >= 5 else 0
    ret_20d = (prices[idx] / prices[max(0, idx-20)] - 1) if idx >= 20 else 0
    ma5 = np.mean(prices[max(0, idx-4):idx+1])
    ma20 = np.mean(window)
    if len(window) > 1:
        log_rets = np.diff(np.log(window))
        vol = np.std(log_rets) * np.sqrt(252) if len(log_rets) > 0 else 0
    else:
        vol = 0
    if idx >= 14:
        deltas = np.diff(prices[idx-14:idx+1])
        gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        losses = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rsi = 100 - 100 / (1 + gains / (losses + 1e-10))
    else:
        rsi = 50
    if len(window) > 1:
        std = np.std(window)
        bb_pos = (current - ma20) / (2 * std + 1e-10)
    else:
        bb_pos = 0
    return np.array([ret_1d, ret_5d, ret_20d, current/ma5-1, current/ma20-1,
                     vol, rsi/100-0.5, np.clip(bb_pos,-2,2)/2], dtype=np.float32)

# === Run episode with embedded policy weights ===
weights = np.load('/tmp/policy.npz')
w1, b1 = weights['w1'], weights['b1']
w2, b2 = weights['w2'], weights['b2']
w_mu, b_mu = weights['w_mu'], weights['b_mu']
log_std = weights['log_std']

def forward(obs):
    h1 = np.tanh(obs @ w1 + b1)
    h2 = np.tanh(h1 @ w2 + b2)
    mu = np.tanh(h2 @ w_mu + b_mu)
    std = np.exp(np.clip(log_std, -2, 0.5))
    return mu, std

# Run one episode
rng = np.random.default_rng({seed})
n_days = {n_days}
commodity_names = list(COMMODITIES.keys())
prices_dict = {{}}
for name, params in COMMODITIES.items():
    prices_dict[name] = generate_prices(params, START_PRICES[name], n_days, rng)

cash = 100000.0
positions = {{n: 0.0 for n in commodity_names}}
initial_cash = cash
peak = cash
day = 20

observations = []
actions_list = []
rewards = []

while day < n_days - 1:
    # Observation
    features = []
    for name in commodity_names:
        features.append(compute_features(prices_dict[name], day))
    pv = cash + sum(positions[n] * prices_dict[n][day] for n in commodity_names)
    port_state = np.array([cash/pv if pv>0 else 1, np.clip(pv/initial_cash-1,-1,5),
                           np.clip((peak-pv)/peak if peak>0 else 0,0,1),
                           (n_days-day)/n_days], dtype=np.float32)
    obs = np.concatenate(features + [port_state])
    observations.append(obs.tolist())

    # Action
    mu, std = forward(obs)
    action = np.clip(mu + np.random.randn(len(mu)) * std, -1, 1)
    actions_list.append(action.tolist())

    # Execute trades
    pv_before = pv
    for i, name in enumerate(commodity_names):
        target_weight = action[i] * 0.25
        target_value = pv_before * target_weight
        current_value = positions[name] * prices_dict[name][day]
        trade_value = target_value - current_value
        if abs(trade_value) > 1:
            price = prices_dict[name][day]
            spread_bps = COMMODITIES[name]["spread_bps"]
            spread = price * spread_bps / 10000
            units = trade_value / price
            positions[name] += units
            cash -= trade_value + np.sign(trade_value) * spread * abs(units)
            cash -= abs(trade_value) * 2 / 10000

    day += 1
    pv_after = cash + sum(positions[n] * prices_dict[n][day] for n in commodity_names)
    peak = max(peak, pv_after)
    daily_ret = (pv_after - pv_before) / pv_before if pv_before > 0 else 0
    dd = (peak - pv_after) / peak
    reward = daily_ret * 100 - dd * 10

    if day >= n_days - 1:
        portfolio_values_arr = [initial_cash]
        total_ret = (pv_after / initial_cash - 1) * 100
        reward += total_ret / 10

    rewards.append(float(reward))

total_return = (pv_after / initial_cash - 1) * 100
result = {{
    "observations": observations,
    "actions": actions_list,
    "rewards": rewards,
    "total_reward": sum(rewards),
    "total_return": round(total_return, 2),
    "final_value": round(pv_after, 2),
}}
print(json.dumps(result))
'''


class TradingPolicy:
    """Same policy as train_commodities.py."""

    def __init__(self, obs_dim, act_dim, hidden=64, lr=0.005):
        self.lr = lr
        self.act_dim = act_dim
        self.w1 = np.random.randn(obs_dim, hidden) * np.sqrt(2.0 / obs_dim)
        self.b1 = np.zeros(hidden)
        self.w2 = np.random.randn(hidden, hidden) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(hidden)
        self.w_mu = np.random.randn(hidden, act_dim) * 0.01
        self.b_mu = np.zeros(act_dim)
        self.log_std = np.zeros(act_dim)

    def save(self, path):
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
                 w_mu=self.w_mu, b_mu=self.b_mu, log_std=self.log_std)

    def load(self, path):
        data = np.load(path)
        self.w1, self.b1 = data["w1"], data["b1"]
        self.w2, self.b2 = data["w2"], data["b2"]
        self.w_mu, self.b_mu = data["w_mu"], data["b_mu"]
        self.log_std = data["log_std"]

    def update(self, trajectories):
        all_returns = []
        for traj in trajectories:
            G = 0
            for r in reversed(traj["rewards"]):
                G = r + 0.99 * G
                all_returns.append(G)
        if not all_returns:
            return
        baseline = np.mean(all_returns)
        std_ret = np.std(all_returns) + 1e-8

        for traj in trajectories:
            G = 0
            for t in range(len(traj["rewards"]) - 1, -1, -1):
                G = traj["rewards"][t] + 0.99 * G
                advantage = (G - baseline) / std_ret
                obs = np.array(traj["observations"][t])
                action = np.array(traj["actions"][t])

                h1 = np.tanh(obs @ self.w1 + self.b1)
                h2 = np.tanh(h1 @ self.w2 + self.b2)
                mu = np.tanh(h2 @ self.w_mu + self.b_mu)
                std = np.exp(np.clip(self.log_std, -2, 0.5))

                diff = (action - mu) / (std ** 2 + 1e-10)
                d_mu = diff * advantage * self.lr / len(trajectories)
                self.w_mu += np.outer(h2, d_mu)
                self.b_mu += d_mu

                d_log_std = ((action - mu)**2 / (std**2 + 1e-10) - 1) * advantage
                self.log_std += d_log_std * self.lr * 0.1 / len(trajectories)
                self.log_std = np.clip(self.log_std, -2, 0.5)


def run_episode_in_sandbox(policy_path: str, seed: int, n_days: int,
                           template: str, namespace: str) -> dict:
    """Run one episode inside a sandbox pod."""
    from k8s_agent_sandbox import SandboxClient
    from arenas.commodities.gym_env import COMMODITIES, START_PRICES

    script = EPISODE_SCRIPT_TEMPLATE.format(
        commodities_json=json.dumps(COMMODITIES),
        start_prices_json=json.dumps(START_PRICES),
        seed=seed,
        n_days=n_days,
    )

    try:
        with SandboxClient(template_name=template, namespace=namespace) as sandbox:
            # Upload policy weights
            with open(policy_path, "rb") as f:
                sandbox.write("policy.npz", f.read())
            sandbox.write("episode.py", script)
            result = sandbox.run("python3 episode.py", timeout=120)

            if result.exit_code == 0:
                return json.loads(result.stdout.strip().split("\n")[-1])
            else:
                return {"rewards": [], "total_reward": -100,
                        "total_return": -100, "error": result.stderr[:200]}
    except Exception as e:
        return {"rewards": [], "total_reward": -100,
                "total_return": -100, "error": str(e)[:200]}


def main():
    parser = argparse.ArgumentParser(description="Distributed commodity RL training")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--parallel", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--trading-days", type=int, default=252)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--template", default="arena-sandbox")
    parser.add_argument("--namespace", default="default")
    args = parser.parse_args()

    obs_dim = 36  # 4 commodities × 8 features + 4 portfolio state
    act_dim = 4   # position per commodity
    policy = TradingPolicy(obs_dim, act_dim, hidden=args.hidden, lr=args.lr)

    print(f"{'='*70}")
    print(f"Distributed Commodity Trading RL (AKS Sandbox)")
    print(f"{'='*70}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Batch:      {args.batch} (run in parallel across sandbox pods)")
    print(f"  Parallel:   {args.parallel} concurrent pods")
    print(f"  Template:   {args.template}")
    print(f"{'='*70}\n")

    start = time.time()
    history = []
    policy_path = "/tmp/sandbox_policy.npz"

    for ep in range(0, args.episodes, args.batch):
        # Save current weights
        policy.save(policy_path)

        # Dispatch episodes to sandbox pods in parallel
        batch_start = time.time()
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = [
                pool.submit(run_episode_in_sandbox, policy_path,
                           seed=ep + i, n_days=args.trading_days,
                           template=args.template, namespace=args.namespace)
                for i in range(args.batch)
            ]
            trajectories = [f.result() for f in futures]

        batch_time = time.time() - batch_start

        # Filter out errors
        valid = [t for t in trajectories if "observations" in t]
        errors = len(trajectories) - len(valid)

        if valid:
            policy.update(valid)

        # Log
        if (ep + args.batch) % args.eval_interval == 0 or ep == 0:
            returns = [t.get("total_return", -100) for t in valid]
            avg_ret = np.mean(returns) if returns else -100
            print(f"  Ep {ep+args.batch:>5d} | ret={avg_ret:>7.1f}% | "
                  f"batch={batch_time:.1f}s | errors={errors} | "
                  f"episodes/s={len(valid)/batch_time:.1f}")
            history.append({
                "episode": ep + args.batch,
                "avg_return": round(avg_ret, 2),
                "batch_time": round(batch_time, 2),
                "errors": errors,
            })

    elapsed = time.time() - start
    policy.save("agents/trained_commodity_distributed")

    print(f"\n{'='*70}")
    print(f"Training Complete ({elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"  Total episodes: {args.episodes}")
    print(f"  Throughput: {args.episodes / elapsed:.1f} episodes/s")
    print(f"  Policy saved: agents/trained_commodity_distributed.npz")

    Path("trajectories").mkdir(exist_ok=True)
    with open(f"trajectories/distributed_training_{int(time.time())}.json", "w") as f:
        json.dump({"episodes": args.episodes, "elapsed": elapsed, "history": history}, f, indent=2)


if __name__ == "__main__":
    main()
