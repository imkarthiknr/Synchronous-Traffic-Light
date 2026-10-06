"""The ``junctioniq`` command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from junctioniq import __version__


def _seeds(text: str) -> list[int]:
    """'1000-1009' or '1,2,3'."""
    if "-" in text:
        lo, hi = (int(x) for x in text.split("-", 1))
        return list(range(lo, hi + 1))
    return [int(x) for x in text.split(",")]


def cmd_build_network(args) -> None:
    from junctioniq import scenario

    sc = scenario.load(args.scenario)
    path = sc.build_network()
    print(f"Wrote {path}")


def cmd_simulate(args) -> None:
    from junctioniq.evaluate import PolicySpec, run_day

    spec = PolicySpec.parse(args.policy)
    if args.gui:
        _run_gui(args, spec)
        return
    metrics, _ = run_day(spec, args.seed, args.scale, args.model, args.scenario)
    print(json.dumps({"policy": spec.label, "seed": args.seed, **metrics}, indent=2))


def _run_gui(args, spec) -> None:
    from junctioniq.control import CONTROL_MODE, make_policy
    from junctioniq.sim.env import JunctionEnv

    env = JunctionEnv(
        args.scenario,
        control=CONTROL_MODE[spec.name],
        demand_scale=args.scale,
        fixed_green=spec.fixed_green,
        gui=True,
    )
    policy = make_policy(spec.name, args.model)
    obs, _ = env.reset(seed=args.seed)
    done = False
    while not done:
        obs, _, term, trunc, _ = env.step(policy(env, obs))
        done = term or trunc
    env.close()


def cmd_train(args) -> None:
    from junctioniq.rl.train import TrainConfig, train

    cfg = TrainConfig.from_yaml(args.config) if args.config else TrainConfig()
    if args.episodes:
        cfg.episodes = args.episodes
    if args.seed is not None:
        cfg.seed = args.seed
    best = train(cfg, args.out)
    print(f"Best model: {best}")


def cmd_evaluate(args) -> None:
    from junctioniq.evaluate import PolicySpec, evaluate, markdown_table

    specs = [PolicySpec.parse(p) for p in args.policies]
    if any(s.name == "dqn" for s in specs) and not args.model:
        sys.exit("The dqn policy needs --model")
    rows = evaluate(specs, _seeds(args.seeds), args.scale, args.model, args.out)
    print()
    print(markdown_table(rows))
    if args.out:
        print(f"\nResults, table and plots in {args.out}")


def cmd_detect(args) -> None:
    try:
        from junctioniq.vision.model import ensure_model
        from junctioniq.vision.pipeline import run
    except ImportError:
        sys.exit("The vision extra is not installed. Run: uv sync --extra vision")
    model = args.model_file or ensure_model()
    summary = run(
        args.video,
        args.out,
        model,
        zones_file=args.zones,
        conf=args.conf,
        max_frames=args.max_frames,
        write_video=not args.no_video,
    )
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="junctioniq",
        description="Traffic signal control: video queues, SUMO simulation and RL.",
    )
    p.add_argument("--version", action="version", version=f"junctioniq {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("build-network", help="regenerate the SUMO network from the OSM extract")
    s.add_argument("--scenario", default="kowdenahalli")
    s.set_defaults(func=cmd_build_network)

    s = sub.add_parser("simulate", help="run one simulated day with one controller")
    s.add_argument(
        "--policy", default="actuated", help="fixed, fixed:20, actuated, max-pressure, random, dqn"
    )
    s.add_argument("--model", type=Path, help="trained model for --policy dqn")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--scale", type=float, default=1.0, help="demand multiplier")
    s.add_argument("--scenario", default="kowdenahalli")
    s.add_argument("--gui", action="store_true", help="watch it in sumo-gui")
    s.set_defaults(func=cmd_simulate)

    s = sub.add_parser("train", help="train the Double DQN agent")
    s.add_argument("--config", type=Path, help="YAML config (see configs/dqn.yaml)")
    s.add_argument("--out", type=Path, default=Path("runs/dqn"))
    s.add_argument("--episodes", type=int)
    s.add_argument("--seed", type=int)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("evaluate", help="compare controllers on the same simulated days")
    s.add_argument(
        "--policies", nargs="+", default=["fixed", "fixed:20", "actuated", "max-pressure"]
    )
    s.add_argument("--model", type=Path)
    s.add_argument("--seeds", default="1000-1009")
    s.add_argument("--scale", type=float, default=1.0)
    s.add_argument("--out", type=Path)
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("detect", help="count vehicles and queues in a video")
    s.add_argument("video", type=Path)
    s.add_argument("--out", type=Path, default=Path("outputs/vision"))
    s.add_argument("--zones", type=Path, help="YAML of named polygons for a fixed camera")
    s.add_argument("--conf", type=float, default=0.3)
    s.add_argument("--max-frames", type=int)
    s.add_argument("--no-video", action="store_true", help="skip the annotated video")
    s.add_argument("--model-file", type=Path, help="use a local YOLOX ONNX file")
    s.set_defaults(func=cmd_detect)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
