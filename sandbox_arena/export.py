"""Export trajectories in LeWM-compatible format.

LeWorldModel expects:
  - observations: (N_trajectories, T, *obs_shape) — states at each timestep
  - actions: (N_trajectories, T, action_dim) — actions taken
  
Optionally includes rewards for RL-compatible world models.

Output formats:
  - .npz (NumPy compressed) — fastest for local training
  - .jsonl (JSON lines) — portable, human-readable

Usage:
    from sandbox_arena.export import collect_and_export

    # Collect trajectories from any env and export
    collect_and_export(
        env_code=open("my_env.py").read(),
        env_class="MyEnv",
        policy_fn=my_policy,
        n_episodes=1000,
        output="trajectories/dataset.npz",
    )
"""

import json
import numpy as np
from pathlib import Path
from typing import Callable, Optional


def trajectories_to_npz(trajectories: list[dict], output: str):
    """Save trajectories as .npz (NumPy compressed arrays).

    Compatible with LeWM and similar world model training pipelines.

    Output arrays:
        observations: (N, T, obs_dim) — padded to max trajectory length
        actions: (N, T, act_dim)
        rewards: (N, T) — optional, for RL
        lengths: (N,) — actual length of each trajectory
    """
    obs_list = [np.array(t["observations"], dtype=np.float32) for t in trajectories]
    act_list = [np.array(t["actions"], dtype=np.float32) for t in trajectories]
    rew_list = [np.array(t["rewards"], dtype=np.float32) for t in trajectories]

    # Pad to max length
    max_len = max(len(o) for o in obs_list)
    obs_dim = obs_list[0].shape[-1] if obs_list[0].ndim > 1 else 1
    act_dim = act_list[0].shape[-1] if act_list[0].ndim > 1 else 1

    N = len(trajectories)
    observations = np.zeros((N, max_len, obs_dim), dtype=np.float32)
    actions = np.zeros((N, max_len, act_dim), dtype=np.float32)
    rewards = np.zeros((N, max_len), dtype=np.float32)
    lengths = np.zeros(N, dtype=np.int32)

    for i, (o, a, r) in enumerate(zip(obs_list, act_list, rew_list)):
        T_obs = len(o)
        T_act = len(a)
        T = min(T_obs, T_act)
        lengths[i] = T

        if o.ndim == 1:
            o = o.reshape(-1, 1)
        if a.ndim == 1:
            a = a.reshape(-1, 1)

        observations[i, :T] = o[:T]
        actions[i, :T] = a[:T]
        rewards[i, :T] = r[:T]

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output,
                        observations=observations,
                        actions=actions,
                        rewards=rewards,
                        lengths=lengths)

    print(f"  Saved {N} trajectories to {output}")
    print(f"  Shape: obs={observations.shape}, act={actions.shape}")
    print(f"  Avg length: {lengths.mean():.0f}, max: {lengths.max()}")

    return output


def trajectories_to_jsonl(trajectories: list[dict], output: str):
    """Save trajectories as .jsonl (one JSON object per line)."""
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for t in trajectories:
            record = {
                "observations": t["observations"],
                "actions": t["actions"],
                "rewards": t["rewards"],
            }
            if "info" in t:
                record["info"] = t["info"]
            f.write(json.dumps(record) + "\n")

    print(f"  Saved {len(trajectories)} trajectories to {output}")
    return output


def collect_and_export(
    env_code: str,
    env_class: str,
    policy_fn: Callable,
    n_episodes: int = 1000,
    output: str = "trajectories/dataset.npz",
    parallel: int = 10,
    mode: str = "local",
    template: str = "arena-sandbox",
    namespace: str = "default",
    max_steps: int = 1000,
    seed_offset: int = 0,
    format: str = "auto",
):
    """Collect trajectories and export in LeWM-compatible format.

    Args:
        env_code: Python source defining the Gym environment
        env_class: Class name to instantiate
        policy_fn: function(obs) -> action
        n_episodes: number of episodes to collect
        output: output file path (.npz or .jsonl)
        parallel: max concurrent episodes
        mode: "local" or "cluster"
        format: "npz", "jsonl", or "auto" (from extension)
    """
    from sandbox_arena.batch import batch_rollout

    print(f"Collecting {n_episodes} trajectories ({mode} mode, {parallel} parallel)...")

    trajectories = batch_rollout(
        env_code=env_code,
        env_class=env_class,
        policy_fn=policy_fn,
        n_episodes=n_episodes,
        parallel=parallel,
        mode=mode,
        template=template,
        namespace=namespace,
        max_steps=max_steps,
        seed_offset=seed_offset,
    )

    # Filter out errors
    valid = [t for t in trajectories if "observations" in t and t["observations"]]
    print(f"  Collected {len(valid)}/{n_episodes} valid trajectories")

    if not valid:
        print("  No valid trajectories collected!")
        return None

    # Determine format
    if format == "auto":
        format = "npz" if output.endswith(".npz") else "jsonl"

    if format == "npz":
        return trajectories_to_npz(valid, output)
    else:
        return trajectories_to_jsonl(valid, output)
