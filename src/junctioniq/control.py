"""Signal control policies that share one interface: ``action = policy(env, obs)``.

``fixed`` and ``actuated`` are run by SUMO itself (the environment ignores the action),
so they appear here only to keep the evaluation loop uniform.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from junctioniq.sim.env import Control, JunctionEnv
from junctioniq.sim.signals import encode_joint

Policy = Callable[[JunctionEnv, np.ndarray], int | None]

# Name -> how the environment must run the signals for that policy.
CONTROL_MODE: dict[str, Control] = {
    "fixed": "fixed",
    "actuated": "actuated",
    "max-pressure": "agent",
    "random": "agent",
    "dqn": "agent",
}


def sumo_runs_it(_env: JunctionEnv, _obs: np.ndarray) -> None:
    return None


def max_pressure(env: JunctionEnv, _obs: np.ndarray) -> int:
    """Max-pressure control: each junction serves the phase with the largest queue imbalance.

    A well-known baseline with a stability guarantee (Varaiya, 2013). Ties keep the
    current phase, and the minimum green is enforced by the environment.
    """
    choice = []
    for pressures, ctrl in zip(env.pressures(), env.controllers, strict=True):
        best = max(pressures)
        choice.append(
            ctrl.current if pressures[ctrl.current] >= best else int(np.argmax(pressures))
        )
    return encode_joint(choice, env.sizes)


def random_policy(seed: int = 0) -> Policy:
    rng = np.random.default_rng(seed)

    def act(env: JunctionEnv, _obs: np.ndarray) -> int:
        return int(rng.integers(env.action_space.n))

    return act


def make_policy(name: str, model: Path | None = None) -> Policy:
    if name in ("fixed", "actuated"):
        return sumo_runs_it
    if name == "max-pressure":
        return max_pressure
    if name == "random":
        return random_policy()
    if name == "dqn":
        if model is None:
            raise ValueError("The dqn policy needs a trained model file (--model)")
        from junctioniq.rl.dqn import DQNAgent

        agent = DQNAgent.load(model)
        return lambda _env, obs: agent.act(obs, epsilon=0.0)
    raise ValueError(f"Unknown policy {name!r}. Choose from: {', '.join(CONTROL_MODE)}")
