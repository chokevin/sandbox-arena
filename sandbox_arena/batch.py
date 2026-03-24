"""Batch rollout — run many episodes in parallel across sandbox pods.

This is the key scaling primitive. Instead of running episodes sequentially,
dispatch N episodes to N sandbox pods simultaneously.
"""

import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable


def batch_rollout(
    env_code: str,
    env_class: str,
    policy_fn: Callable,
    n_episodes: int = 50,
    parallel: int = 50,
    template: str = "arena-sandbox",
    namespace: str = "default",
    mode: str = "cluster",
    max_steps: int = 1000,
    seed_offset: int = 0,
) -> list[dict]:
    """Run N episodes in parallel, each in its own sandbox pod.

    Args:
        env_code: Python source code defining the Gym environment
        env_class: Class name to instantiate
        policy_fn: function(obs: np.ndarray) -> action (np.ndarray or list)
        n_episodes: total episodes to run
        parallel: max concurrent sandbox pods
        template: SandboxTemplate name
        namespace: K8s namespace
        mode: "cluster" or "local"
        max_steps: max steps per episode
        seed_offset: starting seed for reproducibility

    Returns:
        List of episode results: {observations, actions, rewards, info}
    """

    def run_one_episode(episode_idx: int) -> dict:
        """Run a single episode."""
        from sandbox_arena.env import SandboxEnv

        env = SandboxEnv(template=template, namespace=namespace, mode=mode)
        env.setup(env_code=env_code, env_class=env_class)

        seed = seed_offset + episode_idx
        obs, info = env.reset(seed=seed)

        observations = [obs.tolist()]
        actions = []
        rewards = []

        for step in range(max_steps):
            action = policy_fn(obs)
            if isinstance(action, np.ndarray):
                action = action.tolist()

            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs.tolist())
            actions.append(action)
            rewards.append(float(reward))

            if terminated or truncated:
                break

        env.close()

        return {
            "episode": episode_idx,
            "seed": seed,
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "total_reward": sum(rewards),
            "steps": len(rewards),
            "info": info,
        }

    # Dispatch episodes in parallel
    results = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(run_one_episode, i): i
            for i in range(n_episodes)
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({
                    "episode": futures[future],
                    "error": str(e),
                    "rewards": [],
                    "total_reward": 0,
                })

    return sorted(results, key=lambda r: r.get("episode", 0))
