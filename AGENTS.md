# sandbox-arena Agent Instructions

## Default Mode

All experimentation runs in **cluster mode** (`--mode cluster`). This executes
agent code in isolated Agent Sandbox pods on AKS — not locally.

```bash
# Evaluate an agent
python run.py cluster <arena> <agent-file>

# Run the dojo (multi-turn LLM agent)
python dojo_agent.py --arena <arena> --all --mode cluster

# Train with evolutionary strategy
python train.py --arena <arena> --mode cluster --parallel 50

# Train commodity trading RL
python train_commodities.py --episodes 8000

# Train distributed across sandbox pods
python train_distributed.py --episodes 5000 --batch 50 --parallel 50
```

## Prerequisites for cluster mode

1. AKS cluster with agent-sandbox controller + extensions installed
2. sandbox-router deployed
3. SandboxTemplate and WarmPool applied:
   ```bash
   kubectl apply -f sandbox/
   ```
4. `pip install k8s-agent-sandbox` in your environment
5. `kubectl` configured to reach the cluster

## Arenas

| Arena | Agent interface | Metric |
|-------|----------------|--------|
| `trading` | `strategy(prices, position, cash) → buy/sell/hold` | Return % |
| `coding` | Define functions (fib, fizzbuzz, etc.) | Pass rate |
| `coding_hard` | Harder algorithm problems (LRU cache, sliding window) | Pass rate |
| `blackjack` | `play(hand_total, dealer_showing, num_cards) → hit/stand` | Win rate |
| `survival` | `survive(health, food, energy, shelter, turn) → action` | Survival % |
| `commodities` | Gym env: continuous position sizing across Gold/Oil/Wheat/NatGas | Sharpe ratio |
| `snake` | Gym env: 0=up, 1=right, 2=down, 3=left | Score |
| `gridworld` | Gym env: navigate maze 0=up, 1=right, 2=down, 3=left | Solved % |

## Training data

The dojo agent automatically saves trajectories to `trajectories/`:
- `trajectories_*.jsonl` — full conversations
- `sft_*.jsonl` — solved challenges (for supervised fine-tuning)
- `dpo_*.jsonl` — preference pairs (for RLHF/DPO)

## Next: AlphaAgent Integration

To integrate [AlphaAgent](https://github.com/gregorizeidler/AlphaAgent) (PPO + FinBERT sentiment + ensemble):
1. Build AlphaAgent Docker image → push to ACR
2. Create SandboxTemplate pointing to that image
3. Wrap AlphaAgent's `train_agent.py` as a sandbox-arena episode
4. Distribute episode collection across sandbox warm pool
