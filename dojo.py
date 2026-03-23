"""Multi-turn Dojo environment — LLM agents iteratively solve challenges.

Like MLE-Dojo's Gym interface but cloud-native. The agent gets:
1. A challenge description
2. Ability to submit code
3. Structured feedback (score, errors, hints)
4. Multiple attempts to iterate

Each code execution runs in an isolated sandbox (local subprocess or AKS pod).
"""

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class StepResult:
    """Result of a single step in the dojo."""
    observation: str
    score: float
    done: bool
    info: dict = field(default_factory=dict)


class Dojo:
    """Multi-turn environment for iterative agent problem-solving.

    Usage:
        dojo = Dojo(arena="coding", mode="local")
        obs = dojo.reset(scenario_index=0)  # get challenge description
        result = dojo.step(code)             # submit solution, get feedback
        # agent reads feedback, improves code
        result = dojo.step(better_code)      # try again
        # repeat until result.done or max_steps
    """

    def __init__(self, arena_name: str, mode: str = "local",
                 max_steps: int = 5, template: str = "arena-sandbox",
                 namespace: str = "default"):
        self.arena = self._load_arena(arena_name)
        self.mode = mode
        self.max_steps = max_steps
        self.template = template
        self.namespace = namespace

        self.current_scenario = None
        self.step_count = 0
        self.best_score = 0
        self.history = []

    @staticmethod
    def _load_arena(name: str):
        if name == "trading":
            from arenas.trading.arena import TradingArena
            return TradingArena()
        elif name == "coding":
            from arenas.coding.arena import CodingArena
            return CodingArena()
        elif name == "blackjack":
            from arenas.blackjack.arena import BlackjackArena
            return BlackjackArena()
        elif name == "survival":
            from arenas.survival.arena import SurvivalArena
            return SurvivalArena()
        else:
            raise ValueError(f"Unknown arena: {name}")

    def reset(self, scenario_index: int = 0) -> str:
        """Reset the environment and return the challenge description."""
        scenarios = self.arena.scenarios()
        self.current_scenario = scenarios[scenario_index % len(scenarios)]
        self.step_count = 0
        self.best_score = 0
        self.history = []

        return self._get_challenge_description()

    def _get_challenge_description(self) -> str:
        """Generate a human-readable challenge description."""
        s = self.current_scenario
        arena = self.arena.name

        if arena == "coding":
            return (
                f"Problem: {s['id']}\n"
                f"{s['description']}\n\n"
                f"Your code will be tested against {len(s['tests'])} test cases.\n"
                f"You have {self.max_steps} attempts."
            )
        elif arena == "trading":
            return (
                f"Trading Challenge: {s['ticker']}\n"
                f"You have {len(s['prices'])} days of price data.\n"
                f"Starting cash: $10,000. Goal: maximize total return.\n\n"
                f"Implement: def strategy(prices, position, cash) -> 'buy'|'sell'|'hold'\n"
                f"You have {self.max_steps} attempts."
            )
        elif arena == "blackjack":
            return (
                f"Blackjack Challenge (seed={s['seed']})\n"
                f"Play {s['hands']} hands of blackjack.\n\n"
                f"Implement: def play(hand_total, dealer_showing, num_cards) -> 'hit'|'stand'\n"
                f"You have {self.max_steps} attempts."
            )
        elif arena == "survival":
            return (
                f"Survival Challenge (difficulty={s['difficulty']}, seed={s['seed']})\n"
                f"Survive {s['max_turns']} turns managing health, food, energy, and shelter.\n\n"
                f"Implement: def survive(health, food, energy, shelter, turn) -> "
                f"'forage'|'rest'|'explore'|'build'\n"
                f"You have {self.max_steps} attempts."
            )
        return f"Challenge: {json.dumps(s)}"

    def step(self, agent_code: str) -> StepResult:
        """Submit code and get feedback."""
        self.step_count += 1
        script = self.arena.eval_script(agent_code, self.current_scenario)

        # Execute
        if self.mode == "local":
            result = self._run_local(script)
        else:
            result = self._run_cluster(script)

        # Parse result
        score = result.get("score", 0)
        passed = result.get("passed", False)
        error = result.get("error", "")
        details = result.get("details", {})
        self.best_score = max(self.best_score, score)

        done = passed or self.step_count >= self.max_steps

        # Build feedback
        feedback_parts = []
        if error:
            feedback_parts.append(f"Error: {error}")
        else:
            feedback_parts.append(f"Score: {score}")
            if passed:
                feedback_parts.append("PASSED!")
            else:
                feedback_parts.append(f"Not yet passing. {self.max_steps - self.step_count} attempts remaining.")

            # Arena-specific hints
            if self.arena.name == "coding" and details.get("errors"):
                feedback_parts.append("Failed tests:")
                for err in details["errors"][:3]:
                    feedback_parts.append(f"  {err}")
            elif self.arena.name == "trading":
                feedback_parts.append(
                    f"Return: {details.get('total_return_pct', 0):.1f}% | "
                    f"Trades: {details.get('trades', 0)}"
                )

        observation = "\n".join(feedback_parts)

        step_record = {
            "step": self.step_count,
            "score": score,
            "passed": passed,
            "code_length": len(agent_code),
        }
        self.history.append(step_record)

        return StepResult(
            observation=observation,
            score=score,
            done=done,
            info={
                "passed": passed,
                "step": self.step_count,
                "best_score": self.best_score,
                "details": details,
                "history": self.history,
            },
        )

    def _run_local(self, script: str, timeout: int = 30) -> dict:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            f.flush()
            try:
                proc = subprocess.run(
                    [sys.executable, f.name],
                    capture_output=True, text=True, timeout=timeout
                )
                if proc.returncode == 0:
                    return json.loads(proc.stdout.strip().split("\n")[-1])
                else:
                    return {"score": 0, "passed": False, "error": proc.stderr[:500]}
            except subprocess.TimeoutExpired:
                return {"score": 0, "passed": False, "error": "Timeout (30s)"}
            except (json.JSONDecodeError, IndexError) as e:
                return {"score": 0, "passed": False, "error": f"Bad output: {e}"}
            finally:
                Path(f.name).unlink(missing_ok=True)

    def _run_cluster(self, script: str) -> dict:
        from k8s_agent_sandbox import SandboxClient
        try:
            with SandboxClient(
                template_name=self.template, namespace=self.namespace
            ) as sandbox:
                sandbox.write("eval.py", script)
                result = sandbox.run("python3 eval.py", timeout=60)
                if result.exit_code == 0:
                    return json.loads(result.stdout.strip().split("\n")[-1])
                else:
                    return {"score": 0, "passed": False, "error": result.stderr[:500]}
        except Exception as e:
            return {"score": 0, "passed": False, "error": str(e)}
