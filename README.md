# poke-sandbox

Pokemon Battle AI Dojo — train and evaluate battle agents in isolated sandboxes on Kubernetes.

```
┌──────────────┐      ┌─────────────────────────────────────┐
│  You submit  │      │  AKS + Agent Sandbox                │
│  a battle    │ ───> │                                     │
│  strategy    │      │  Pod₁: Your agent vs RandomPlayer   │
│  (Python)    │      │  Pod₂: Your agent vs MaxDamage      │
│              │      │  Pod₃: Your agent vs SmartBot       │
│              │ <─── │  ...                                │
│  Get back:   │      │  Pod₅₀: Your agent vs HeuristicBot │
│  73% win rate│      │                                     │
└──────────────┘      └─────────────────────────────────────┘
```

## What This Is

A platform that evaluates Pokemon battle AI agents at scale. You write a Python battle strategy, submit it, and the platform runs it against many opponents in parallel — each battle in an isolated [Agent Sandbox](https://github.com/kubernetes-sigs/agent-sandbox) pod.

This demonstrates the **sandboxed code evaluation** pattern used in RL training for code-generation models, but with Pokemon battles instead of coding problems.

## How It Works

1. **You write an agent** — a Python class that extends `poke-env`'s `Player` and implements `choose_move()`
2. **The orchestrator** dispatches your agent to N sandbox pods in parallel
3. **Each sandbox** runs a local Pokemon Showdown server + your agent vs an opponent
4. **Results** are collected and aggregated (win rate, turns per game, etc.)

Each sandbox is Kata/MSHV isolated — your agent code can't escape, cheat, or crash other battles.

## Quick Start

### Prerequisites

- Python 3.10+
- A Kubernetes cluster with [agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox) installed
- `kubectl` configured

### Local Mode (no cluster needed)

Run a single battle locally to test your agent:

```bash
pip install -r requirements.txt
python run_local.py --agent agents/example_agent.py --opponent random --battles 10
```

### Cluster Mode (sandboxed evaluation)

Evaluate your agent at scale in isolated sandboxes:

```bash
python run_eval.py --agent agents/example_agent.py --opponents all --battles-per-opponent 100
```

## Writing an Agent

Create a Python file in `agents/`:

```python
from poke_env.player import Player

class MyAgent(Player):
    def choose_move(self, battle):
        # Your strategy here!
        # battle.available_moves — list of moves you can use
        # battle.available_switches — Pokemon you can switch to
        # battle.opponent_active_pokemon — the opponent's current Pokemon

        # Example: pick the move with highest base power
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda m: m.base_power)
            return self.create_order(best_move)

        # No moves available, switch to a random Pokemon
        return self.choose_random_move(battle)
```

## Project Structure

```
poke-sandbox/
├── agents/                    # Your battle agents go here
│   ├── example_agent.py       # Simple max-damage agent
│   └── random_agent.py        # Random baseline
├── opponents/                 # Built-in opponents to battle against
│   ├── random_player.py
│   ├── max_damage_player.py
│   └── heuristic_player.py
├── sandbox/                   # Kubernetes sandbox configs
│   ├── sandbox-template.yaml  # SandboxTemplate for battle pods
│   └── warm-pool.yaml         # SandboxWarmPool for fast battles
├── docker/
│   └── Dockerfile             # Battle runtime image
├── orchestrator.py            # Dispatches battles to sandboxes
├── run_local.py               # Run battles locally (no cluster)
├── run_eval.py                # Run evaluation on cluster
└── requirements.txt
```

## Architecture

```
run_eval.py (your machine)
    │
    ├── Creates SandboxClaims (one per battle)
    ├── Writes agent code + opponent code into each sandbox
    ├── Runs battles via SandboxClient
    └── Collects results
          │
          ▼
    SandboxWarmPool (Kubernetes)
    ┌─────────────────────────┐
    │ Pod: Showdown + poke-env│ ── battle ──> result (win/loss)
    │ Pod: Showdown + poke-env│ ── battle ──> result (win/loss)
    │ Pod: Showdown + poke-env│ ── battle ──> result (win/loss)
    │ ...                     │
    └─────────────────────────┘
```

## Why Agent Sandbox?

- **Isolation:** Each battle runs in a Kata/MSHV VM — agents can't cheat, read opponents' code, or crash the cluster
- **Speed:** Warm pool provides sub-second sandbox creation — no cold-start delays between battles
- **Scale:** Run hundreds of battles in parallel on spot nodes (cheap)
- **Safety:** Untrusted agent code is contained — perfect for tournaments or public submissions
