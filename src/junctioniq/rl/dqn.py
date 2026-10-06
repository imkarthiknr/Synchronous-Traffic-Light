"""Double DQN (van Hasselt et al., 2016) in PyTorch.

Compared with the 2021 Keras learner, this version:

- keeps a separate target network and picks the next action with the online network
  (Double DQN), which curbs the over-estimation of plain Q-learning;
- trains on mini-batches in one call instead of fitting one sample at a time;
- uses a Huber loss, gradient clipping and an epsilon schedule over environment steps;
- treats a time-limit cut-off as a truncation, not a terminal state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class DQNConfig:
    hidden: int = 256
    lr: float = 5e-4
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 100_000
    learning_starts: int = 2_000
    train_every: int = 1
    target_update_tau: float = 0.005  # soft (Polyak) target update per learning step
    max_grad_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 40_000

    def epsilon(self, step: int) -> float:
        frac = min(1.0, step / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)


def q_network(n_obs: int, n_actions: int, hidden: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(n_obs, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_actions),
    )


class ReplayBuffer:
    """Fixed-size ring buffer of transitions, sampled uniformly."""

    def __init__(self, capacity: int, n_obs: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, n_obs), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_obs), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self._next = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, obs, action: int, reward: float, next_obs, terminated: bool) -> None:
        i = self._next
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.dones[i] = next_obs, float(terminated)
        self._next = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = self._rng.integers(0, self.size, size=batch_size)
        return {
            "obs": torch.from_numpy(self.obs[idx]),
            "actions": torch.from_numpy(self.actions[idx]),
            "rewards": torch.from_numpy(self.rewards[idx]),
            "next_obs": torch.from_numpy(self.next_obs[idx]),
            "dones": torch.from_numpy(self.dones[idx]),
        }


class DQNAgent:
    def __init__(self, n_obs: int, n_actions: int, config: DQNConfig | None = None, seed: int = 0):
        self.config = config or DQNConfig()
        self.n_obs, self.n_actions = n_obs, n_actions
        torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)
        self.online = q_network(n_obs, n_actions, self.config.hidden)
        self.target = q_network(n_obs, n_actions, self.config.hidden)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.lr)

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        return self.online(x).squeeze(0).numpy()

    def act(self, obs: np.ndarray, epsilon: float) -> int:
        if self._rng.random() < epsilon:
            return int(self._rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(obs)))

    def learn(self, batch: dict[str, torch.Tensor]) -> float:
        cfg = self.config
        q = self.online(batch["obs"]).gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_action = self.online(batch["next_obs"]).argmax(dim=1, keepdim=True)
            next_q = self.target(batch["next_obs"]).gather(1, next_action).squeeze(1)
            target = batch["rewards"] + cfg.gamma * (1.0 - batch["dones"]) * next_q
        loss = nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), cfg.max_grad_norm)
        self.optimizer.step()
        with torch.no_grad():
            for t, o in zip(self.target.parameters(), self.online.parameters(), strict=True):
                t.lerp_(o, cfg.target_update_tau)
        return float(loss.item())

    def save(self, path: Path, extra: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "n_obs": self.n_obs,
                "n_actions": self.n_actions,
                "config": asdict(self.config),
                "online": self.online.state_dict(),
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> DQNAgent:
        data = torch.load(path, map_location="cpu", weights_only=True)
        agent = cls(data["n_obs"], data["n_actions"], DQNConfig(**data["config"]))
        agent.online.load_state_dict(data["online"])
        agent.target.load_state_dict(data["online"])
        agent.online.eval()
        agent.extra = data.get("extra", {})
        return agent
