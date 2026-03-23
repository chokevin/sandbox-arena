"""Trajectory collector — records dojo sessions as training data.

Each trajectory captures the full multi-turn conversation:
  challenge → code attempt → feedback → improved code → feedback → ...

Output format is JSONL (one trajectory per line), compatible with
SFT fine-tuning pipelines (OpenAI, Hugging Face TRL, axolotl).
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Turn:
    role: str  # "system", "user", "assistant"
    content: str


@dataclass 
class Trajectory:
    arena: str
    scenario_id: str
    solved: bool
    best_score: float
    attempts: int
    total_time: float
    turns: list[Turn] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_chat_format(self) -> list[dict]:
        """Convert to OpenAI chat format for SFT training."""
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def to_dpo_pairs(self) -> list[dict]:
        """Extract preference pairs for DPO training.
        
        Winning response = code that scored higher.
        Losing response = code that scored lower.
        """
        pairs = []
        attempts = []
        
        for i, turn in enumerate(self.turns):
            if turn.role == "assistant":
                # Find the next user turn which has the score feedback
                score = 0
                if i + 1 < len(self.turns) and self.turns[i + 1].role == "user":
                    feedback = self.turns[i + 1].content
                    if "Score:" in feedback:
                        try:
                            score = float(feedback.split("Score:")[1].split()[0])
                        except (ValueError, IndexError):
                            pass
                attempts.append({"code": turn.content, "score": score})

        # Create pairwise comparisons
        for i in range(len(attempts)):
            for j in range(i + 1, len(attempts)):
                if attempts[i]["score"] != attempts[j]["score"]:
                    winner = i if attempts[i]["score"] > attempts[j]["score"] else j
                    loser = j if winner == i else i
                    # Get the prompt that preceded the winner
                    pairs.append({
                        "prompt": self.turns[0].content if self.turns else "",
                        "chosen": attempts[winner]["code"],
                        "rejected": attempts[loser]["code"],
                    })
        return pairs


class TrajectoryCollector:
    """Collects trajectories from dojo sessions."""

    def __init__(self, output_dir: str = "trajectories"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectories: list[Trajectory] = []

    def new_trajectory(self, arena: str, scenario_id: str) -> Trajectory:
        traj = Trajectory(
            arena=arena,
            scenario_id=scenario_id,
            solved=False,
            best_score=0,
            attempts=0,
            total_time=0,
        )
        self.trajectories.append(traj)
        return traj

    def save(self, filename: Optional[str] = None):
        """Save all trajectories as JSONL."""
        if not filename:
            filename = f"trajectories_{int(time.time())}.jsonl"
        
        path = self.output_dir / filename
        with open(path, "w") as f:
            for traj in self.trajectories:
                record = {
                    "arena": traj.arena,
                    "scenario_id": traj.scenario_id,
                    "solved": traj.solved,
                    "best_score": traj.best_score,
                    "attempts": traj.attempts,
                    "total_time": traj.total_time,
                    "messages": traj.to_chat_format(),
                    "metadata": traj.metadata,
                }
                f.write(json.dumps(record) + "\n")
        
        return path

    def save_sft_dataset(self, filename: Optional[str] = None):
        """Save only successful trajectories as SFT training data.
        
        Format: one conversation per line, only including solved challenges.
        """
        if not filename:
            filename = f"sft_{int(time.time())}.jsonl"

        path = self.output_dir / filename
        count = 0
        with open(path, "w") as f:
            for traj in self.trajectories:
                if traj.solved:
                    record = {"messages": traj.to_chat_format()}
                    f.write(json.dumps(record) + "\n")
                    count += 1

        return path, count

    def save_dpo_dataset(self, filename: Optional[str] = None):
        """Save preference pairs for DPO training.
        
        Only from trajectories with multiple attempts (where we can compare).
        """
        if not filename:
            filename = f"dpo_{int(time.time())}.jsonl"

        path = self.output_dir / filename
        count = 0
        with open(path, "w") as f:
            for traj in self.trajectories:
                if traj.attempts > 1:
                    for pair in traj.to_dpo_pairs():
                        f.write(json.dumps(pair) + "\n")
                        count += 1

        return path, count

    def summary(self) -> dict:
        total = len(self.trajectories)
        solved = sum(1 for t in self.trajectories if t.solved)
        scores = [t.best_score for t in self.trajectories]
        attempts = [t.attempts for t in self.trajectories]
        return {
            "total_challenges": total,
            "solved": solved,
            "solve_rate": f"{solved/total*100:.1f}%" if total else "0%",
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "avg_attempts": sum(attempts) / len(attempts) if attempts else 0,
            "total_turns": sum(len(t.turns) for t in self.trajectories),
        }
