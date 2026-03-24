# sandbox-arena

Run Gym environments in isolated Kubernetes sandbox pods. Train RL agents at scale.

## Install

```bash
pip install sandbox-arena                 # core (local mode)
pip install sandbox-arena[cluster]        # + AKS Agent Sandbox support
```

## Usage

### Single episode

```python
from sandbox_arena import SandboxEnv

env = SandboxEnv(mode="cluster")  # or mode="local" for dev
env.setup(env_code=open("my_env.py").read(), env_class="MyEnv")

obs, info = env.reset()
obs, reward, done, truncated, info = env.step(action)
env.close()
```

### Parallel episodes (the key feature)

```python
from sandbox_arena import batch_rollout

results = batch_rollout(
    env_code=open("my_env.py").read(),
    env_class="MyEnv",
    policy_fn=my_model.get_action,
    n_episodes=50,
    parallel=50,       # 50 sandbox pods simultaneously
    mode="cluster",    # each episode in its own isolated pod
)

for r in results:
    print(f"Episode {r['episode']}: reward={r['total_reward']:.2f}")
```

### Export trajectories (for world model training)

```python
from sandbox_arena import collect_and_export

collect_and_export(
    env_code=open("my_env.py").read(),
    env_class="MyEnv",
    policy_fn=my_policy,
    n_episodes=1000,
    output="trajectories.npz",
    mode="cluster",
)
```

## Prerequisites (cluster mode)

```bash
# AKS cluster with agent-sandbox installed
kubectl apply -f sandbox/sandbox-template.yaml
kubectl apply -f sandbox/warm-pool.yaml
```

## Why sandbox pods?

| | Local subprocess | Sandbox pod (AKS) |
|---|---|---|
| Isolation | Process-level | VM-level (Kata/MSHV) |
| Parallelism | Limited by CPU cores | 100s of pods on spot nodes |
| Untrusted code | Risky | Safe (hardware isolation) |
| Cost | Free | ~$0.01/episode on spot |
| Setup | None | AKS + agent-sandbox |

Use **local** mode for development. Use **cluster** mode when you need scale, isolation, or both.

## Examples

The `examples/` directory contains training scripts, arenas, and tools built on top of the core API:

```
examples/
├── arenas/              # 8 environments (trading, coding, games)
├── train_commodities.py # RL training for commodity trading
├── train_rl.py          # Gym env RL (snake, gridworld)
├── dojo_agent.py        # Multi-turn LLM coding agent
├── tournament.py        # Multi-agent leaderboards
└── visualize.py         # Training curves
```

## API Reference

### `SandboxEnv(template, namespace, mode)`

Gym-like environment where `step()` runs inside a sandbox pod.

- `setup(env_code, env_class)` — provide environment as Python source
- `reset(seed)` → `(obs, info)`
- `step(action)` → `(obs, reward, terminated, truncated, info)`
- `close()` — release the sandbox pod

### `batch_rollout(env_code, env_class, policy_fn, n_episodes, parallel, mode)`

Run N episodes in parallel. Returns list of `{observations, actions, rewards, total_reward, info}`.

### `collect_and_export(env_code, env_class, policy_fn, n_episodes, output, mode)`

Collect trajectories and save as `.npz` (compatible with LeWM and similar world model training).
