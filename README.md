# sandbox-arena

Evaluate untrusted code at scale in isolated sandboxes on AKS.

Submit an agent (Python), pick an arena (trading, Pokemon, coding), and the platform evaluates it in parallel across sandboxed Kubernetes pods — each execution isolated via [Agent Sandbox](https://github.com/kubernetes-sigs/agent-sandbox) + Kata/MSHV.

```
You submit           AKS + Agent Sandbox
a Python agent  ───> ┌─────────────────────────────────┐
                     │ SandboxWarmPool (spot nodes)     │
                     │                                  │
                     │ Pod₁: run agent (isolated VM)    │
                     │ Pod₂: run agent (isolated VM)    │
                     │ ...                              │
                     │ Pod₅₀: run agent (isolated VM)   │
                     │                                  │
Get results    <───  │ Aggregate: score, rank, stats    │
                     └─────────────────────────────────┘
```

## Arenas

| Arena | What it evaluates | Environment | Metric |
|-------|------------------|-------------|--------|
| **trading** | RL trading strategies on historical stock data | FinRL / custom gym | Sharpe ratio, total return |
| **pokemon** | Battle AI agents via Pokemon Showdown | poke-env | Win rate |
| **coding** | Code solutions against test suites | pytest | Pass rate |

Each arena defines an `evaluate()` function. The platform handles sandboxing, parallelism, and result aggregation.

## Quick Start

### Prerequisites

- Python 3.10+
- An AKS cluster with [agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox) installed
- `kubectl` configured, `pip install k8s-agent-sandbox`

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Write an agent

**Trading agent** (`agents/my_trader.py`):
```python
def strategy(prices, position, cash):
    """Simple momentum strategy.
    
    Args:
        prices: list of historical closing prices (most recent last)
        position: current shares held
        cash: current cash balance
    Returns:
        action: "buy", "sell", or "hold"
    """
    if len(prices) < 5:
        return "hold"
    if prices[-1] > prices[-5]:  # price going up
        return "buy" if cash > 0 else "hold"
    else:
        return "sell" if position > 0 else "hold"
```

**Coding agent** (`agents/my_solver.py`):
```python
def solve(problem: str) -> str:
    """Given a coding problem description, return Python code that solves it."""
    # Your logic here — could call an LLM, use heuristics, etc.
    return "print('hello world')"
```

### 3. Evaluate locally (no cluster)

```bash
python run.py local trading agents/my_trader.py
python run.py local coding agents/my_solver.py
```

### 4. Evaluate at scale on AKS

```bash
# Apply sandbox configs
kubectl apply -f sandbox/

# Run evaluation across sandboxed pods
python run.py cluster trading agents/my_trader.py --parallel 50
```

## Architecture

```
run.py
  │
  ├── local mode:   runs evaluate() in-process
  │
  └── cluster mode: for each evaluation run:
        ├── SandboxClient claims a pod from warm pool
        ├── Writes agent code + arena environment into pod
        ├── Runs evaluation inside sandbox
        ├── Collects result (score, stdout, stderr)
        └── Releases pod back to pool
```

### Why Agent Sandbox?

- **Isolation:** Each evaluation in a Kata/MSHV VM — agents can't cheat, read other agents, or crash the cluster
- **Speed:** Warm pool = sub-second pod claims (no cold start)
- **Scale:** 100s of parallel evaluations on spot nodes (cheap)
- **Safety:** Run untrusted code from anyone — tournaments, public submissions, RL training loops

## Project Structure

```
sandbox-arena/
├── arenas/
│   ├── base.py              # Arena interface
│   ├── trading/
│   │   ├── arena.py          # Trading evaluation logic
│   │   ├── data/             # Sample stock data
│   │   └── README.md
│   ├── pokemon/
│   │   ├── arena.py          # Pokemon battle evaluation
│   │   └── README.md
│   └── coding/
│       ├── arena.py          # Code evaluation against test suites
│       ├── problems/         # Sample problems + tests
│       └── README.md
├── sandbox/
│   ├── sandbox-template.yaml # SandboxTemplate for eval pods
│   └── warm-pool.yaml        # SandboxWarmPool config
├── agents/                   # Your agents go here
│   └── examples/
├── run.py                    # CLI entrypoint
├── orchestrator.py           # Dispatches evaluations to sandboxes
└── requirements.txt
```
