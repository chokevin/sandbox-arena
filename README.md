# sandbox-arena

Gym-style RL training platform with isolated execution on Kubernetes. Train agents against diverse environments — trading, coding, games — with episodes running in parallel across sandboxed pods.

## What this is

A framework for **reinforcement learning** where the environment runs in isolated [Agent Sandbox](https://github.com/kubernetes-sigs/agent-sandbox) pods on AKS. Your model trains on a GPU node, sandbox-arena provides the `step() → reward` signal.

```
Your model (GPU node)              sandbox-arena (sandbox pods)
┌──────────────────┐              ┌───────────────────────────┐
│ Decision          │  actions    │ Pod₁: market sim → reward │
│ Transformer       │ ──────────>│ Pod₂: market sim → reward │
│ (168K params)     │             │ ...                       │
│                   │  rewards   │ Pod₅₀: market sim → reward│
│ Update weights   │ <──────────│                           │
└──────────────────┘              └───────────────────────────┘
```

## Training Approaches

| Approach | Model | Environment | Training |
|----------|-------|-------------|----------|
| **Decision Transformer** | Causal transformer (PyTorch) | Commodity trading | Collect → SFT → RL |
| **Policy RL** | Neural net (REINFORCE) | Trading, Snake, GridWorld | REINFORCE with baseline |
| **Evolutionary** | Parameterized strategies | Trading | Mutation + selection |
| **LLM Agent** | GPT-4o / DeepSeek / Llama | Coding challenges | Multi-turn dojo with feedback |

## Quick Start

```bash
pip install -r requirements.txt

# Train a Decision Transformer on commodity trading
python train_transformer.py --phase all

# Train a simple RL policy on Snake
python train_rl.py --env snake --episodes 2000

# Run the LLM coding dojo
export OPENAI_API_KEY=$(gh auth token)
export OPENAI_BASE_URL=https://models.inference.ai.azure.com
python dojo_agent.py --arena coding_hard --all

# Tournament: pit all agents against each other
python tournament.py --mode local

# Visualize training results
python visualize.py
```

## Environments

| Arena | Type | Observation | Action | Metric |
|-------|------|-------------|--------|--------|
| `commodities` | Multi-asset trading | Price features + technicals (36d) | Position sizing (4d continuous) | Sharpe ratio |
| `trading` | Simple strategies | Price history | buy/sell/hold | Return % |
| `coding` | Test suites | Problem description | Python code | Pass rate |
| `coding_hard` | Hard algorithms | Problem description | Python code | Pass rate |
| `blackjack` | Card game | Hand + dealer (3d) | hit/stand | Win rate |
| `survival` | Resource management | Health/food/energy/shelter | forage/rest/explore/build | Survival % |
| `snake` | Classic game | Grid (192d) | up/right/down/left | Score |
| `gridworld` | Maze navigation | Position + walls (8d) | up/right/down/left | Solved % |

## Architecture

```
sandbox-arena/
├── Environments
│   └── arenas/              8 environments (trading, coding, games)
│       ├── commodities/     Multi-commodity Gym env with realistic dynamics
│       ├── coding_hard/     LeetCode-style problems for LLM agents
│       ├── snake/           Classic Snake as Gym env
│       └── ...
│
├── Training
│   ├── train_transformer.py Decision Transformer (collect → SFT → RL)
│   ├── train_commodities.py NN policy for commodity trading (REINFORCE)
│   ├── train_rl.py          Gym env RL for games (Snake, GridWorld)
│   ├── train.py             Evolutionary strategy optimization
│   └── train_distributed.py Distributed RL via Agent Sandbox pods
│
├── Evaluation
│   ├── dojo.py              Multi-turn Gym-style environment
│   ├── dojo_agent.py        LLM agent with iterative feedback
│   ├── tournament.py        Multi-agent parallel leaderboards
│   ├── benchmark.py         Track model performance over time
│   └── visualize.py         Training curves + evaluation charts
│
├── Infrastructure
│   ├── providers.py         Pluggable LLM backends (OpenAI, vLLM, HF)
│   ├── collector.py         Trajectory collection (SFT + DPO datasets)
│   ├── sandbox/             K8s SandboxTemplate + WarmPool configs
│   ├── deploy.sh            Deploy training as K8s Job on AKS
│   └── Dockerfile           Container image for sandbox pods
│
└── Agents
    └── agents/examples/     Example agents for each arena
```

## Running on AKS (Cluster Mode)

All commands support `--mode cluster` to run episodes in isolated Agent Sandbox pods:

```bash
# Prerequisites
kubectl apply -f sandbox/

# Distributed commodity RL (50 parallel sandbox pods)
python train_distributed.py --episodes 5000 --batch 50 --parallel 50

# Tournament across sandbox pods
python tournament.py --mode cluster --parallel 50

# LLM dojo with sandboxed code execution
python dojo_agent.py --arena coding_hard --all --mode cluster
```

## Results

**Decision Transformer** (commodity trading, first run):
- 168K params, 3-layer causal transformer
- SFT loss: 0.048 → 0.027
- Returns: -0.17% (needs better training data + more RL episodes)

**Simple NN Policy** (commodity trading, 8000 episodes):
- Returns: -3.1% → ~0% (learned to stop losing)
- 56% win rate

**LLM Agent** (coding_hard, GPT-4o):
- 4/5 solved first attempt
- LRU Cache: failed all 5 attempts (75% — shows where model improvement is needed)

**Evolutionary Strategy** (trading):
- Returns: 13.2% → 15.5% over 15 generations

## Model Providers

```bash
# API models (GitHub Models — free)
export OPENAI_API_KEY=$(gh auth token)
export OPENAI_BASE_URL=https://models.inference.ai.azure.com

# Self-hosted models (vLLM on AKS GPU nodes)
export VLLM_BASE_URL=http://vllm-svc.default.svc.cluster.local:8000/v1
export VLLM_MODEL=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
```
