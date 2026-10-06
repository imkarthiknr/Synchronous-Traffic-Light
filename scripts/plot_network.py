"""Draw a scenario's road network with its signals and entries: docs/images/network.png."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sumolib

from junctioniq import scenario


def main(name: str = "kowdenahalli", out: str = "docs/images/network.png") -> None:
    sc = scenario.load(name)
    net = sumolib.net.readNet(str(sc.net))
    fig, ax = plt.subplots(figsize=(4.6, 7.2))
    for edge in net.getEdges():
        xs, ys = zip(*edge.getShape(), strict=True)
        ax.plot(xs, ys, color="#5c677d", lw=2.5, solid_capstyle="round", zorder=1)
    for jid, label in sc.junctions.items():
        x, y = net.getNode(jid).getCoord()
        ax.scatter([x], [y], s=140, color="#e63946", edgecolor="white", zorder=3)
        ax.annotate(label, (x, y), xytext=(8, 6), textcoords="offset points", fontsize=8)
    for entry, edge_id in sc.entries.items():
        x, y = net.getEdge(edge_id).getFromNode().getCoord()
        ax.scatter([x], [y], marker="s", s=40, color="#2a9d8f", zorder=2)
        ax.annotate(
            entry.replace("_", " "),
            (x, y),
            xytext=(6, -10),
            textcoords="offset points",
            fontsize=7,
            color="#2a9d8f",
        )
    ax.set_aspect("equal")
    ax.set_xlabel("metres")
    ax.set_title(f"{sc.name}\nred: signals, green: entries/exits", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
