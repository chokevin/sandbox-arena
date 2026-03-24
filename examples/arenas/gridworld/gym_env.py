"""Grid World arena — navigate a maze to reach a goal.

Simple grid world with walls. Good for testing basic RL algorithms.
"""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


MAZES = {
    "simple": [
        "S....",
        ".###.",
        "...#.",
        ".#...",
        "....G",
    ],
    "medium": [
        "S.......",
        ".######.",
        "......#.",
        ".####.#.",
        ".#....#.",
        ".#.####.",
        ".#......",
        "......#G",
    ],
}


class GridWorldEnv(gym.Env):
    """Grid world maze navigation.

    Observation: [agent_r, agent_c, goal_r, goal_c, wall_up, wall_right, wall_down, wall_left]
    Action: 0=up, 1=right, 2=down, 3=left
    Reward: -0.01 per step, +1 reaching goal, -0.1 hitting wall
    """

    def __init__(self, maze_name: str = "simple"):
        super().__init__()
        maze_lines = MAZES.get(maze_name, MAZES["simple"])
        self.rows = len(maze_lines)
        self.cols = max(len(line) for line in maze_lines)
        self.walls = set()
        self.start = (0, 0)
        self.goal = (self.rows - 1, self.cols - 1)

        for r, line in enumerate(maze_lines):
            for c, ch in enumerate(line):
                if ch == "#":
                    self.walls.add((r, c))
                elif ch == "S":
                    self.start = (r, c)
                elif ch == "G":
                    self.goal = (r, c)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        self.max_steps = self.rows * self.cols * 3
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent = self.start
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        r, c = self.agent
        gr, gc = self.goal
        return np.array([
            r / self.rows, c / self.cols,
            gr / self.rows, gc / self.cols,
            float((r - 1, c) in self.walls or r - 1 < 0),
            float((r, c + 1) in self.walls or c + 1 >= self.cols),
            float((r + 1, c) in self.walls or r + 1 >= self.rows),
            float((r, c - 1) in self.walls or c - 1 < 0),
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        r, c = self.agent
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dr, dc = moves[action]
        new_r, new_c = r + dr, c + dc

        if (new_r < 0 or new_r >= self.rows or new_c < 0 or
                new_c >= self.cols or (new_r, new_c) in self.walls):
            reward = -0.1
        else:
            self.agent = (new_r, new_c)
            reward = -0.01

        if self.agent == self.goal:
            return self._get_obs(), 1.0, True, False, {"steps": self.steps, "solved": True}
        if self.steps >= self.max_steps:
            return self._get_obs(), -1.0, True, False, {"steps": self.steps, "solved": False}
        return self._get_obs(), reward, False, False, {"steps": self.steps}
