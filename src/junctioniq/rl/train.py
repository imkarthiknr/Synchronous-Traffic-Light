"""Training loop for the Double DQN agent."""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from junctioniq.rl.dqn import DQNAgent, DQNConfig, ReplayBuffer
from junctioniq.sim.env import JunctionEnv


@dataclass
class TrainConfig:
    scenario: str = "kowdenahalli"
    episodes: int = 120
    seed: int = 0
    reward: str = "queue"
    delta: int = 5
    yellow_time: int = 3
    min_green: int = 10
    # Each training day gets a random demand level in this range, so the agent sees
    # quiet and busy days rather than memorising one.
    demand_scale: tuple[float, float] = (0.6, 1.2)
    horizon: int | None = None  # seconds of demand per day; None = the scenario's full profile
    validate_every: int = 5
    validation_seeds: list[int] = field(default_factory=lambda: [900, 901])
    dqn: DQNConfig = field(default_factory=DQNConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> TrainConfig:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        dqn = DQNConfig(**data.pop("dqn", {}))
        if "demand_scale" in data:
            data["demand_scale"] = tuple(data["demand_scale"])
        return cls(**data, dqn=dqn)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["demand_scale"] = list(self.demand_scale)
        return d


def _env(cfg: TrainConfig, scale, drain: int = 0) -> JunctionEnv:
    return JunctionEnv(
        cfg.scenario,
        control="agent",
        reward=cfg.reward,
        delta=cfg.delta,
        yellow_time=cfg.yellow_time,
        min_green=cfg.min_green,
        demand_scale=scale,
        drain=drain,
        horizon=cfg.horizon,
    )


def _model_info(cfg: TrainConfig, episode: int, validation: dict | None) -> dict:
    """Stored with the weights so evaluation runs the environment the way training did."""
    return {
        "episode": episode,
        "validation": validation or {},
        "env": {
            "scenario": cfg.scenario,
            "reward": cfg.reward,
            "delta": cfg.delta,
            "yellow_time": cfg.yellow_time,
            "min_green": cfg.min_green,
        },
    }


def validate(agent: DQNAgent, cfg: TrainConfig) -> dict[str, float]:
    """Greedy runs on held-out days at normal demand; returns averaged metrics."""
    env = _env(cfg, 1.0, drain=1800)
    results = []
    try:
        for seed in cfg.validation_seeds:
            obs, _ = env.reset(seed=seed)
            done = False
            while not done:
                obs, _, term, trunc, info = env.step(agent.act(obs, epsilon=0.0))
                done = term or trunc
            results.append(info["metrics"])
    finally:
        env.close()
    return {k: float(np.mean([r[k] for r in results])) for k in results[0]}


def train(cfg: TrainConfig, out_dir: Path, log=print) -> Path:
    """Train, validate periodically and keep the best model. Returns the best model's path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    env = _env(cfg, cfg.demand_scale)
    n_obs, n_actions = env.observation_space.shape[0], int(env.action_space.n)
    agent = DQNAgent(n_obs, n_actions, cfg.dqn, seed=cfg.seed)
    buffer = ReplayBuffer(cfg.dqn.buffer_size, n_obs, seed=cfg.seed)

    log_file = out_dir / "training.csv"
    fields = [
        "episode", "steps", "epsilon", "return", "loss", "demand_scale",
        "mean_travel_time_s", "mean_junction_queue", "val_mean_travel_time_s",
        "val_mean_junction_queue", "seconds",
    ]  # fmt: skip
    best_path, best_score = out_dir / "best.pt", float("inf")
    step = 0
    with log_file.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for episode in range(1, cfg.episodes + 1):
            started = time.perf_counter()
            obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            done, ep_return, losses = False, 0.0, []
            while not done:
                eps = cfg.dqn.epsilon(step)
                action = agent.act(obs, eps)
                next_obs, reward, term, trunc, step_info = env.step(action)
                buffer.add(obs, action, reward, next_obs, term)
                obs, done = next_obs, term or trunc
                ep_return += reward
                step += 1
                if len(buffer) >= cfg.dqn.learning_starts and step % cfg.dqn.train_every == 0:
                    losses.append(agent.learn(buffer.sample(cfg.dqn.batch_size)))
            env.close_sim()
            metrics = step_info["metrics"]
            row = {
                "episode": episode,
                "steps": step,
                "epsilon": round(cfg.dqn.epsilon(step), 3),
                "return": round(ep_return, 2),
                "loss": round(float(np.mean(losses)), 4) if losses else "",
                "demand_scale": round(info["demand_scale"], 3),
                "mean_travel_time_s": round(metrics["mean_travel_time_s"], 1),
                "mean_junction_queue": round(metrics["mean_junction_queue"], 2),
            }
            if episode % cfg.validate_every == 0 or episode == cfg.episodes:
                val = validate(agent, cfg)
                row["val_mean_travel_time_s"] = round(val["mean_travel_time_s"], 1)
                row["val_mean_junction_queue"] = round(val["mean_junction_queue"], 2)
                # Unfinished vehicles count as a heavy penalty so gridlock never looks good.
                score = val["mean_travel_time_s"] + 10 * val["still_in_network"]
                if score < best_score:
                    best_score = score
                    agent.save(best_path, _model_info(cfg, episode, val))
            row["seconds"] = round(time.perf_counter() - started, 1)
            writer.writerow(row)
            fh.flush()
            log(
                f"episode {episode:>3}  eps {row['epsilon']:.2f}  return {row['return']:>8.1f}  "
                f"travel {row['mean_travel_time_s']:>6.1f}s"
                + (
                    f"  | val travel {row['val_mean_travel_time_s']:.1f}s"
                    if "val_mean_travel_time_s" in row
                    else ""
                )
            )
    env.close()
    agent.save(out_dir / "last.pt", _model_info(cfg, cfg.episodes, None))
    return best_path
