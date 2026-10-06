"""Plot a training run's learning curve: python scripts/plot_training.py runs/dqn/training.csv."""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(log: str, out: str = "docs/images/training.png", reference: str | None = None) -> None:
    with Path(log).open() as fh:
        rows = list(csv.DictReader(fh))
    ep = [int(r["episode"]) for r in rows]
    train_tt = [float(r["mean_travel_time_s"]) for r in rows]
    val = [
        (int(r["episode"]), float(r["val_mean_travel_time_s"]))
        for r in rows
        if r["val_mean_travel_time_s"]
    ]
    eps = [float(r["epsilon"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ep, train_tt, color="#adb5bd", lw=1, label="training day (random demand, exploring)")
    ax.plot(
        *zip(*val, strict=True),
        color="#2a9d8f",
        lw=2,
        marker="o",
        ms=3,
        label="validation days (greedy, normal demand)",
    )
    if reference:
        ax.axhline(
            float(reference),
            color="#6c757d",
            ls="--",
            lw=1,
            label="max-pressure on validation days",
        )
    ax.set_xlabel("Episode (one simulated hour)")
    ax.set_ylabel("Mean travel time (s)")
    ax.set_ylim(bottom=0)
    ax2 = ax.twinx()
    ax2.plot(ep, eps, color="#e9c46a", lw=1)
    ax2.set_ylabel("Exploration (epsilon)", color="#b08900")
    ax2.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Double DQN training")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
