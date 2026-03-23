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

## Training data

The dojo agent automatically saves trajectories to `trajectories/`:
- `trajectories_*.jsonl` — full conversations
- `sft_*.jsonl` — solved challenges (for supervised fine-tuning)
- `dpo_*.jsonl` — preference pairs (for RLHF/DPO)
