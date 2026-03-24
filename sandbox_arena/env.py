"""SandboxEnv — Gym-compatible environment that runs inside a sandbox pod.

The user provides:
  1. Environment code (a Gym env as a Python string or file)
  2. Actions (via step())

sandbox-arena handles:
  1. Creating a sandbox pod from the warm pool
  2. Uploading the environment code
  3. Sending actions, receiving observations + rewards
  4. Cleaning up the pod

This is a thin wrapper around k8s_agent_sandbox.SandboxClient.
"""

import json
import numpy as np
from typing import Callable, Optional


class SandboxEnv:
    """Gym-like environment that executes inside a Kubernetes sandbox pod.

    Usage:
        env = SandboxEnv(template="python-runtime")

        # Option 1: Provide env as inline code
        env.setup(env_code=open("my_env.py").read(), env_class="MyTradingEnv")

        # Option 2: Provide env as a pip-installable package
        env.setup(pip_install="my-trading-env", env_class="my_env:TradingEnv")

        obs, info = env.reset()
        obs, reward, done, truncated, info = env.step([0.5, -0.3, 0.1, 0.0])
        env.close()
    """

    def __init__(self, template: str = "arena-sandbox",
                 namespace: str = "default",
                 mode: str = "cluster"):
        """
        Args:
            template: SandboxTemplate name on the cluster
            namespace: Kubernetes namespace
            mode: "cluster" (real sandbox pods) or "local" (subprocess, for dev)
        """
        self.template = template
        self.namespace = namespace
        self.mode = mode
        self._sandbox = None
        self._env_code = None
        self._env_class = None
        self._episode_id = 0

    def setup(self, env_code: str = None, env_file: str = None,
              env_class: str = "Env", pip_install: str = None,
              setup_code: str = None):
        """Configure the environment that runs inside the sandbox.

        Args:
            env_code: Python source code defining the Gym environment
            env_file: Path to a .py file defining the environment
            env_class: Class name to instantiate (default: "Env")
            pip_install: pip package to install in the sandbox
            setup_code: Additional Python code to run before creating the env
        """
        if env_file:
            with open(env_file) as f:
                self._env_code = f.read()
        elif env_code:
            self._env_code = env_code
        else:
            self._env_code = ""

        self._env_class = env_class
        self._pip_install = pip_install
        self._setup_code = setup_code or ""

    def reset(self, seed: int = None, **kwargs) -> tuple:
        """Reset the environment. Creates a new sandbox pod if needed."""
        self._episode_id += 1

        if self.mode == "local":
            return self._reset_local(seed, **kwargs)
        else:
            return self._reset_cluster(seed, **kwargs)

    def step(self, action) -> tuple:
        """Send an action to the environment running in the sandbox."""
        if self.mode == "local":
            return self._step_local(action)
        else:
            return self._step_cluster(action)

    def close(self):
        """Clean up the sandbox pod."""
        if self._sandbox:
            try:
                self._sandbox.__exit__(None, None, None)
            except Exception:
                pass
            self._sandbox = None

    # === Cluster mode (real sandbox pods) ===

    def _reset_cluster(self, seed=None, **kwargs):
        """Create sandbox pod and initialize environment inside it."""
        from k8s_agent_sandbox import SandboxClient

        # Close previous sandbox if any
        self.close()

        # Create new sandbox
        self._sandbox = SandboxClient(
            template_name=self.template,
            namespace=self.namespace,
        )
        self._sandbox.__enter__()

        # Upload environment code
        if self._env_code:
            self._sandbox.write("sandbox_env.py", self._env_code)

        # Build and upload the runner script
        runner = self._build_runner_script(seed)
        self._sandbox.write("runner.py", runner)

        # Install dependencies if needed
        if self._pip_install:
            self._sandbox.run(f"pip install -q {self._pip_install}", timeout=120)

        # Initialize the environment
        result = self._sandbox.run("python3 runner.py reset", timeout=60)
        return self._parse_result(result.stdout)

    def _step_cluster(self, action):
        """Send action to sandbox and get result."""
        action_json = json.dumps(action if isinstance(action, list) else action.tolist())
        result = self._sandbox.run(
            f'python3 runner.py step \'{action_json}\'', timeout=60
        )
        return self._parse_step_result(result.stdout)

    # === Local mode (in-process, for development) ===

    def _reset_local(self, seed=None, **kwargs):
        """Run environment locally in-process."""
        import importlib.util
        import tempfile
        from pathlib import Path

        if self._env_code:
            self._tmpdir = tempfile.mkdtemp()
            Path(f"{self._tmpdir}/sandbox_env.py").write_text(self._env_code)

            spec = importlib.util.spec_from_file_location(
                "sandbox_env", f"{self._tmpdir}/sandbox_env.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            env_cls = getattr(mod, self._env_class)
            self._local_env = env_cls()

        obs, info = self._local_env.reset(seed=seed)
        if hasattr(obs, 'tolist'):
            obs = np.array(obs, dtype=np.float32)
        return obs, info

    def _step_local(self, action):
        if isinstance(action, list):
            action = np.array(action)
        obs, reward, terminated, truncated, info = self._local_env.step(action)
        if hasattr(obs, 'tolist'):
            obs = np.array(obs, dtype=np.float32)
        return obs, float(reward), bool(terminated), bool(truncated), info

    # === Helpers ===

    def _build_runner_script(self, seed=None) -> str:
        """Generate the Python script that runs inside the sandbox."""
        seed_str = str(seed) if seed is not None else "None"
        return f'''
import json
import sys
import pickle
import numpy as np

{self._setup_code}

# Import the environment
from sandbox_env import {self._env_class}

STATE_FILE = "/tmp/env_state.pkl"

def save_env(env):
    with open(STATE_FILE, "wb") as f:
        pickle.dump(env, f)

def load_env():
    with open(STATE_FILE, "rb") as f:
        return pickle.load(f)

def to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

cmd = sys.argv[1]

if cmd == "reset":
    env = {self._env_class}()
    obs, info = env.reset(seed={seed_str})
    save_env(env)
    result = {{"obs": to_json(obs), "info": {{k: to_json(v) for k, v in info.items()}}}}
    print(json.dumps(result))

elif cmd == "step":
    action = json.loads(sys.argv[2])
    env = load_env()
    obs, reward, terminated, truncated, info = env.step(np.array(action))
    save_env(env)
    result = {{
        "obs": to_json(obs),
        "reward": to_json(reward),
        "terminated": to_json(terminated),
        "truncated": to_json(truncated),
        "info": {{k: to_json(v) for k, v in info.items()}},
    }}
    print(json.dumps(result))
'''

    def _parse_result(self, stdout: str) -> tuple:
        data = json.loads(stdout.strip().split("\n")[-1])
        obs = np.array(data["obs"], dtype=np.float32)
        return obs, data.get("info", {})

    def _parse_step_result(self, stdout: str) -> tuple:
        data = json.loads(stdout.strip().split("\n")[-1])
        obs = np.array(data["obs"], dtype=np.float32)
        return (
            obs,
            float(data["reward"]),
            bool(data["terminated"]),
            bool(data["truncated"]),
            data.get("info", {}),
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
