#!/usr/bin/env python3
"""Interactive Dojo — LLM agent iteratively solves challenges.

The LLM reads the challenge, writes code, gets feedback, and improves.
This is the multi-turn version of run.py.

Usage:
    # With GitHub Models (free)
    export OPENAI_API_KEY=$(gh auth token)
    export OPENAI_BASE_URL=https://models.inference.ai.azure.com

    python dojo_agent.py --arena coding --scenario 0
    python dojo_agent.py --arena trading --scenario 0
    python dojo_agent.py --arena blackjack --all
"""

import argparse
import os
import sys
import time

from dojo import Dojo

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai")
    sys.exit(1)


SYSTEM_PROMPT = """You are an expert Python programmer competing in a coding dojo.
You will be given a challenge and must write Python code to solve it.
After each submission, you'll get feedback (score, errors, hints).
Use the feedback to improve your solution.

Rules:
- Output ONLY executable Python code — no markdown fences, no explanations.
- The code must be self-contained (stdlib only).
- Read the feedback carefully and fix specific errors."""


def strip_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    if code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return code


def run_dojo_session(client, model: str, dojo: Dojo, scenario_index: int, trajectory=None):
    """Run one multi-turn dojo session."""
    challenge = dojo.reset(scenario_index)
    label = dojo.current_scenario.get("id") or dojo.current_scenario.get("ticker") or f"#{scenario_index}"

    print(f"\n{'='*60}")
    print(f"Challenge: {label}")
    print(f"{'='*60}")
    print(challenge)
    print(f"{'='*60}\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Solve this challenge:\n\n{challenge}"},
    ]

    # Record trajectory
    if trajectory:
        from collector import Turn
        trajectory.turns.append(Turn(role="system", content=SYSTEM_PROMPT))
        trajectory.turns.append(Turn(role="user", content=f"Solve this challenge:\n\n{challenge}"))

    for attempt in range(1, dojo.max_steps + 1):
        # Ask LLM for code
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0.2,
        )
        code = strip_fences(response.choices[0].message.content)

        print(f"  Attempt {attempt}/{dojo.max_steps}")
        print(f"  Code ({len(code)} chars): {code[:80]}...")

        # Submit to dojo
        result = dojo.step(code)

        icon = "✅" if result.info["passed"] else "❌"
        print(f"  {icon} Score: {result.score} | {result.observation.split(chr(10))[0]}")

        # Record trajectory
        if trajectory:
            trajectory.turns.append(Turn(role="assistant", content=code))
            trajectory.attempts = attempt
            trajectory.best_score = result.info["best_score"]

        if result.done:
            if result.info["passed"]:
                print(f"  🎉 Solved in {attempt} attempt(s)!")
                if trajectory:
                    trajectory.solved = True
            else:
                print(f"  ⛔ Not solved after {attempt} attempts. Best: {result.info['best_score']}")
            return result

        # Feed result back to LLM for next attempt
        feedback = f"Your solution scored {result.score}. Here's the feedback:\n\n{result.observation}\n\nPlease fix and resubmit."
        messages.append({"role": "assistant", "content": code})
        messages.append({"role": "user", "content": feedback})

        if trajectory:
            trajectory.turns.append(Turn(role="user", content=feedback))

    return result


def main():
    parser = argparse.ArgumentParser(description="Dojo Agent — LLM iteratively solves challenges")
    parser.add_argument("--arena", default="coding", help="Arena (coding, trading, blackjack, survival)")
    parser.add_argument("--scenario", type=int, default=None, help="Scenario index (default: all)")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--max-steps", type=int, default=3, help="Max attempts per challenge")
    parser.add_argument("--mode", default="local", choices=["local", "cluster"])
    parser.add_argument("--model", default=None, help="Model name override")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    model = args.model or os.environ.get("OPENAI_MODEL", "gpt-4o")

    if not api_key:
        print("Set OPENAI_API_KEY (or: export OPENAI_API_KEY=$(gh auth token))")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)
    dojo = Dojo(arena_name=args.arena, mode=args.mode, max_steps=args.max_steps)
    scenarios = dojo.arena.scenarios()

    if args.scenario is not None:
        indices = [args.scenario]
    elif args.all:
        indices = list(range(len(scenarios)))
    else:
        indices = [0]

    print(f"Dojo Agent — {args.arena} arena, {len(indices)} challenge(s), max {args.max_steps} attempts each")
    print(f"Model: {model}")

    start = time.time()
    results = []

    # Trajectory collection
    from collector import TrajectoryCollector
    collector = TrajectoryCollector()

    for idx in indices:
        scenario = scenarios[idx % len(scenarios)]
        scenario_id = scenario.get("id") or scenario.get("ticker") or str(idx)
        traj = collector.new_trajectory(arena=args.arena, scenario_id=scenario_id)
        traj_start = time.time()

        result = run_dojo_session(client, model, dojo, idx, trajectory=traj)
        traj.total_time = time.time() - traj_start
        results.append(result)

    elapsed = time.time() - start

    # Save trajectories
    traj_path = collector.save()
    sft_path, sft_count = collector.save_sft_dataset()
    dpo_path, dpo_count = collector.save_dpo_dataset()

    # Summary
    solved = sum(1 for r in results if r.info["passed"])
    print(f"\n{'='*60}")
    print(f"Dojo Complete ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  Solved: {solved}/{len(results)}")
    avg_score = sum(r.info["best_score"] for r in results) / len(results)
    print(f"  Avg Best Score: {avg_score:.1f}")
    total_steps = sum(r.info["step"] for r in results)
    print(f"  Total Attempts: {total_steps}")

    summary = collector.summary()
    print(f"\n  Training Data:")
    print(f"    Trajectories: {traj_path}")
    print(f"    SFT examples: {sft_count} ({sft_path})")
    print(f"    DPO pairs:    {dpo_count} ({dpo_path})")
    print(f"    Total turns:  {summary['total_turns']}")


if __name__ == "__main__":
    main()
