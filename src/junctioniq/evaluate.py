"""Compare signal controllers on the same simulated days.

Every controller sees exactly the same vehicles (same demand seed and SUMO seed), and
each day runs until the network is empty (or 30 extra minutes), so vehicles stuck in
a queue at the end still count.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from junctioniq.control import CONTROL_MODE, make_policy
from junctioniq.sim.env import JunctionEnv

METRICS = [
    ("mean_travel_time_s", "Mean travel time (s)"),
    ("mean_stopped_time_s", "Mean stopped time (s)"),
    ("mean_junction_queue", "Mean queue (veh)"),
    ("max_junction_queue", "Max queue (veh)"),
    ("teleported", "Teleports"),
    ("still_in_network", "Unfinished"),
]


@dataclass(frozen=True)
class PolicySpec:
    """A controller to evaluate, e.g. ``fixed``, ``fixed:20``, ``actuated``, ``dqn``."""

    name: str
    fixed_green: int | None = None

    @classmethod
    def parse(cls, text: str) -> PolicySpec:
        name, _, arg = text.partition(":")
        if name not in CONTROL_MODE:
            raise ValueError(f"Unknown policy {name!r}. Choose from: {', '.join(CONTROL_MODE)}")
        if arg and name != "fixed":
            raise ValueError(
                "Only the fixed policy takes an argument (green seconds), e.g. fixed:20"
            )
        return cls(name, int(arg) if arg else None)

    @property
    def label(self) -> str:
        if self.name == "fixed":
            return f"Fixed-time ({self.fixed_green or 42} s greens)"
        return {
            "actuated": "Actuated (SUMO)",
            "max-pressure": "Max-pressure",
            "random": "Random",
            "dqn": "Double DQN (ours)",
        }[self.name]


def run_day(
    spec: PolicySpec,
    seed: int,
    scale: float = 1.0,
    model: Path | None = None,
    scenario: str = "kowdenahalli",
    drain: int = 1800,
) -> tuple[dict[str, float], list[int]]:
    """Simulate one day with one controller. Returns the metrics and the per-second queue."""
    env_kwargs = {}
    if spec.name == "dqn":
        from junctioniq.rl.dqn import DQNAgent

        info = DQNAgent.load(model).extra.get("env", {}) if model else {}
        env_kwargs = {k: info[k] for k in ("delta", "yellow_time", "min_green") if k in info}
        scenario = info.get("scenario", scenario)
    env = JunctionEnv(
        scenario,
        control=CONTROL_MODE[spec.name],
        demand_scale=scale,
        drain=drain,
        fixed_green=spec.fixed_green,
        **env_kwargs,
    )
    policy = make_policy(spec.name, model)
    try:
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(policy(env, obs))
            done = term or trunc
        return info["metrics"], list(env.metrics.queue_trace)
    finally:
        env.close()


def evaluate(
    specs: list[PolicySpec],
    seeds: list[int],
    scale: float = 1.0,
    model: Path | None = None,
    out_dir: Path | None = None,
    log=print,
) -> list[dict]:
    rows, traces = [], {}
    for spec in specs:
        for seed in seeds:
            metrics, trace = run_day(spec, seed, scale, model)
            rows.append({"policy": spec.label, "seed": seed, "demand_scale": scale, **metrics})
            if seed == seeds[0]:
                traces[spec.label] = trace
        log(f"{spec.label:<28} {summarise(rows, spec.label)['mean_travel_time_s']}")
    if out_dir:
        write_outputs(rows, traces, out_dir, scale)
    return rows


def summarise(rows: list[dict], label: str) -> dict[str, str]:
    mine = [r for r in rows if r["policy"] == label]
    out = {}
    for key, _ in METRICS:
        values = np.array([r[key] for r in mine], dtype=float)
        if key in ("teleported", "still_in_network", "max_junction_queue"):
            out[key] = f"{values.mean():.1f}"
        else:
            out[key] = f"{values.mean():.1f} ± {values.std():.1f}"
    return out


def markdown_table(rows: list[dict]) -> str:
    labels = list(dict.fromkeys(r["policy"] for r in rows))
    head = "| Controller | " + " | ".join(title for _, title in METRICS) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in METRICS) + " |"
    body = []
    for label in labels:
        s = summarise(rows, label)
        body.append(f"| {label} | " + " | ".join(s[k] for k, _ in METRICS) + " |")
    return "\n".join([head, sep, *body])


def write_outputs(rows: list[dict], traces: dict[str, list[int]], out_dir: Path, scale: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    seeds = sorted({r["seed"] for r in rows})
    (out_dir / "summary.md").write_text(
        f"Demand scale {scale}, {len(seeds)} simulated days (seeds {seeds[0]} to {seeds[-1]}). "
        "Mean ± standard deviation across days.\n\n" + markdown_table(rows) + "\n",
        encoding="utf-8",
    )

    labels = list(dict.fromkeys(r["policy"] for r in rows))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    means = [np.mean([r["mean_travel_time_s"] for r in rows if r["policy"] == lb]) for lb in labels]
    stds = [np.std([r["mean_travel_time_s"] for r in rows if r["policy"] == lb]) for lb in labels]
    colors = ["#2a9d8f" if "DQN" in lb else "#8d99ae" for lb in labels]
    ax.barh(labels, means, xerr=stds, color=colors, capsize=3)
    ax.invert_yaxis()
    ax.set_xlabel("Mean travel time per vehicle (s), lower is better")
    ax.set_title(f"Kowdenahalli, {len(seeds)} simulated days")
    fig.tight_layout()
    fig.savefig(out_dir / "travel_time.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for label, trace in traces.items():
        minutes = np.arange(len(trace)) / 60
        smooth = np.convolve(trace, np.ones(60) / 60, mode="same")
        ax.plot(minutes, smooth, label=label, lw=2.2 if "DQN" in label else 1.3)
    ax.set_xlabel("Minutes")
    ax.set_ylabel("Vehicles queued at the 4 junctions")
    ax.set_title(f"Queues over one simulated day (seed {seeds[0]})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "queues.png", dpi=130)
    plt.close(fig)
