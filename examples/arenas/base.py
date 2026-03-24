"""Base arena interface. All arenas implement this."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalResult:
    """Result of a single evaluation run."""
    score: float
    passed: bool
    details: dict = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    error: str = ""


class Arena:
    """Base class for evaluation arenas."""

    name: str = "base"

    def eval_script(self, agent_code: str, scenario: dict) -> str:
        """Generate a Python script that evaluates the agent in this arena.

        The script must print a JSON result to stdout with at least:
            {"score": float, "passed": bool, "details": {}}

        Args:
            agent_code: The user's agent source code
            scenario: Arena-specific scenario config (e.g. stock ticker, problem ID)
        Returns:
            A complete Python script as a string
        """
        raise NotImplementedError

    def scenarios(self) -> list[dict]:
        """Return the list of scenarios to evaluate against."""
        raise NotImplementedError

    def aggregate(self, results: list[EvalResult]) -> dict:
        """Aggregate results across all scenarios into a summary."""
        passed = sum(1 for r in results if r.passed)
        scores = [r.score for r in results if not r.error]
        return {
            "arena": self.name,
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "errors": sum(1 for r in results if r.error),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
        }
