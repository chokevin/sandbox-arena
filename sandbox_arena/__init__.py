"""sandbox-arena: Run Gym environments in isolated Kubernetes sandbox pods.

Usage:
    from sandbox_arena import SandboxEnv

    env = SandboxEnv(template="python-runtime", namespace="default")
    obs, info = env.reset(env_code="from my_env import MyEnv", env_class="MyEnv")
    obs, reward, done, truncated, info = env.step(action)

    results = env.batch_rollout(policy_fn, n_episodes=50, parallel=50)
"""

from sandbox_arena.env import SandboxEnv
from sandbox_arena.batch import batch_rollout

__all__ = ["SandboxEnv", "batch_rollout"]
