# Development Guide

How the project is put together and how to work on it. The [README](README.md) has the overview and results.

- [Setup](#setup)
- [Project layout](#project-layout)
- [The scenario](#the-scenario)
- [The environment](#the-environment)
- [Controllers](#controllers)
- [Double DQN](#double-dqn)
- [Evaluation](#evaluation)
- [Vision pipeline](#vision-pipeline)
- [Testing](#testing)
- [Continuous integration](#continuous-integration)
- [Common tasks](#common-tasks)
- [Troubleshooting](#troubleshooting)

## Setup

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/). SUMO comes from PyPI (`eclipse-sumo`), so there is nothing else to install.

```bash
uv sync --all-extras        # everything, including the vision extra
uv run pytest               # fast tests (about 15 s)
```

On Linux, PyPI's PyTorch wheels include about 3 GB of CUDA libraries. Training runs on the CPU, so if you don't have an NVIDIA GPU use the CPU build instead:

```bash
scripts/install_cpu.sh      # locked versions, CPU-only torch
export UV_NO_SYNC=1         # stop `uv run` from reinstalling the CUDA build
```

`SUMO_HOME` is set automatically from the `eclipse-sumo` package. Set it yourself only to use a system SUMO (for example for `sumo-gui` on Windows or macOS).

## Project layout

```
src/junctioniq/
├── scenario.py            scenario files, network rebuild, per-episode traffic demand
├── scenarios/kowdenahalli/
│   ├── scenario.yaml      junctions, entries/exits, vehicle mix, demand profile
│   ├── kowdenahalli.osm   OpenStreetMap extract (ODbL)
│   └── kowdenahalli.net.xml   SUMO network built from it
├── sim/
│   ├── sumo.py            finding SUMO; libsumo (fast) or TraCI (GUI)
│   ├── signals.py         per-junction signal state machine, joint action encoding
│   ├── env.py             Gymnasium environment
│   └── metrics.py         per-episode measures
├── control.py             fixed-time, actuated, max-pressure, random, DQN policies
├── rl/dqn.py              Double DQN agent and replay buffer
├── rl/train.py            training loop, validation, checkpoints
├── evaluate.py            same-day comparison of controllers, tables and plots
├── vision/                YOLOX detector, ByteTrack, camera-motion compensation, queues
└── cli.py                 the `junctioniq` command
configs/dqn.yaml           the training configuration used for the published results
results/                   the published evaluation (CSV, summary table, plots)
scripts/                   CPU install helper, network figure
```

## The scenario

[`scenario.yaml`](src/junctioniq/scenarios/kowdenahalli/scenario.yaml) is the single description of the place and its traffic:

- **Network.** The OSM extract covers two parallel east–west roads joined by two north–south roads, with four signalised junctions. `junctioniq build-network` rebuilds `kowdenahalli.net.xml` with the `netconvert` options listed in the file: left-hand traffic (as in India), the four OSM nodes as traffic lights, 42 s greens and 3 s yellows for the default plan. The rebuild is byte-for-byte reproducible and CI checks it.
- **Demand.** Each simulated day is an hour in three periods: a build-up, a peak flowing in from the north, and a later peak in the other direction. Arrivals are Poisson per entry and second, a vehicle never exits where it entered, and the vehicle mix is cars, two-wheelers and trucks. `write_demand(path, seed, scale)` writes a SUMO trips file, so a seed is a reproducible day and `scale` makes it busier or quieter.

To add a scenario, create a folder next to `kowdenahalli/` with the same files and pass `--scenario <name>`.

## The environment

[`JunctionEnv`](src/junctioniq/sim/env.py) is a Gymnasium environment for all signals at once.

| Part | Definition |
| --- | --- |
| Step | `delta` = 5 simulated seconds |
| Action | One joint action: a green phase for every junction. Four junctions with two greens each gives 16 actions, as in the 2021 project |
| Switching | Each junction has a `SignalController`. A request to change only happens after `min_green` (10 s) and always passes through a `yellow_time` (3 s) yellow. Requests during yellow are ignored |
| Observation | Per junction and approach lane: halted vehicles and all vehicles, each divided by the lane's capacity (length / 7.5 m); the current green as one-hot; whether the junction may switch now. 38 values in [0, 1] |
| Reward | `queue` (default): minus the average number of halted vehicles on the approaches during the step, divided by 10. Also `wait` (change in waiting time) and `speed` (mean relative speed, the 2021 "metric 1") |
| End | `truncated` at the horizon (a time limit, so the agent still bootstraps), or `terminated` once demand is over and the network is empty. With `drain > 0` the day continues after the horizon until the network empties or `drain` seconds pass |

`control` decides who runs the lights: `agent` (the action does), `fixed` (a fixed-time plan, the network's own or `fixed_green` seconds per green) or `actuated` (SUMO's gap-based actuated control with 10–60 s greens). In the last two the action is ignored.

The simulation runs in-process through **libsumo**, so one day of an hour plus drain takes 1–3 seconds. Only one libsumo simulation can be open per process: `reset()` closes the previous one, and you should `close()` an environment before opening another. `gui=True` switches to TraCI and `sumo-gui`.

## Controllers

All controllers share the signature `action = policy(env, obs)` ([`control.py`](src/junctioniq/control.py)):

- **Fixed-time**: `fixed` is the network's own 42 s plan; `fixed:N` uses N-second greens. `fixed:20` was the best equal-split plan in a sweep of 10–42 s on the validation days.
- **Actuated (SUMO)**: extends a green while vehicles keep arriving. Changing its maximum green between 20 and 60 s made little difference on this network.
- **Max-pressure**: each junction serves the green with the largest difference between queued vehicles it would serve and vehicles queued downstream (Varaiya, 2013). It uses the same yellow and minimum green as the agent.
- **Random**: a sanity check.
- **Double DQN**: the trained agent, acting greedily.

## Double DQN

[`rl/dqn.py`](src/junctioniq/rl/dqn.py): an MLP (38 → 256 → 256 → 16), a replay buffer of 100k transitions, Double DQN targets (the online network picks the next action, the target network values it), Huber loss, Adam, gradient clipping at 10 and a soft target update (τ = 0.005). Exploration decays linearly from 1.0 to 0.05 over 60k steps.

[`rl/train.py`](src/junctioniq/rl/train.py): each training day gets a new seed and a random demand level between 0.6× and 1.2×. Every 5 episodes the greedy agent runs on two held-out validation days (seeds 900 and 901) at normal demand, and the best one is saved as `best.pt`. The score is mean travel time plus 10 s per vehicle still in the network, so gridlock can't look good. Outputs in the run folder: `config.yaml`, `training.csv`, `best.pt`, `last.pt`. A model file stores the environment settings it was trained with, and evaluation reuses them.

```bash
uv run junctioniq train --config configs/dqn.yaml --out runs/dqn
```

200 episodes take about 25 minutes on two CPU cores.

## Evaluation

```bash
uv run junctioniq evaluate \
  --policies fixed fixed:20 actuated max-pressure random dqn \
  --model runs/dqn/best.pt --seeds 1000-1009 --out results
```

Every controller drives the same 10 days: the same vehicles, the same SUMO seed, run until the network is empty (at most 30 extra minutes). The evaluation seeds 1000–1009 were not used in training or model selection. Travel time counts from each vehicle's planned departure, so time spent waiting to enter a jammed network counts too. Outputs: `results.csv` (one row per controller and day), `summary.md` (mean ± standard deviation), `travel_time.png` and `queues.png`.

## Vision pipeline

`junctioniq detect VIDEO` ([`vision/`](src/junctioniq/vision)):

1. **Detection**: YOLOX-S (COCO, ONNX Runtime on the CPU), letterboxed to 640×640, decoded from the raw grid output, class-agnostic NMS. Cars, two-wheelers (motorbikes and bicycles) and heavy vehicles (buses and trucks) are kept.
2. **Tracking**: ByteTrack from Supervision gives each vehicle an ID across frames.
3. **Camera motion**: corner features outside vehicle boxes are tracked with Lucas–Kanade optical flow, and a similarity transform is fitted with RANSAC. This handles panning and zooming footage like the demo clip.
4. **Queues**: a vehicle is *queued* once it has been tracked for 1 s and its motion over the road, after removing camera motion, stays below 0.25 vehicle lengths per second (smoothed over 1 s).
5. **Zones** (optional, for fixed cameras): a YAML file of named polygons, e.g. one per approach lane. Each frame then gets a queue count per zone.

```yaml
# zones.yaml: pixel coordinates in the video
northbound: [[120, 340], [230, 340], [250, 120], [190, 120]]
southbound: [[260, 340], [380, 340], [300, 120], [255, 120]]
```

The model (35 MB) is downloaded to `~/.cache/junctioniq/models` on first use and checked against a pinned SHA-256. Set `JUNCTIONIQ_CACHE` to put it elsewhere, or pass `--model-file`. Speed is about 0.5 s per frame at 480×360 on a two-core CPU, so the 15-second demo clip takes about two minutes.

Known limits: small two-wheelers in low-resolution footage are often missed by a COCO-trained model, and auto-rickshaws are reported as cars or two-wheelers. See the [roadmap](ROADMAP.md).

## Testing

```bash
uv run pytest                  # 43 fast tests
uv run pytest -m model         # needs the YOLOX download
uv run pytest -m "not slow and not model"   # skip the evaluation test too
```

| File | Covers |
| --- | --- |
| `test_signals.py` | Green extraction, yellow states, minimum green, switching, joint action encoding |
| `test_scenario.py` | Loading, reproducible and scalable demand, no U-turns, network rebuild matches the committed file |
| `test_env.py` | Observation bounds, determinism per seed, yellow visible in SUMO, truncation, fixed/actuated draining, rewards, pressures |
| `test_dqn.py` | Epsilon schedule, replay buffer, the Double DQN target on a hand-built network, learning a bandit, save/load |
| `test_train_eval.py` | A short training run end to end, YAML config, policy parsing, evaluation outputs |
| `test_vision.py` | Letterbox, YOLOX decoding and NMS on synthetic outputs, queue logic, camera-pan compensation, motion estimation; the real model on the demo clip (`-m model`) |
| `test_cli.py` | Seed ranges, defaults, `simulate` output |

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) on pushes to `main` and pull requests:

- **test** (Python 3.11 and 3.13): CPU-only install, Ruff lint and format check, pytest, a network rebuild that must match the committed file, and an evaluation of three controllers on one light day.
- **vision**: downloads YOLOX (cached), runs the model test and `junctioniq detect` on the demo clip.

## Common tasks

| Task | Command |
| --- | --- |
| Watch a controller in SUMO's GUI | `uv run junctioniq simulate --policy max-pressure --gui` |
| One day's numbers | `uv run junctioniq simulate --policy dqn --model runs/dqn/best.pt --seed 3` |
| Busier traffic | add `--scale 1.3` to `simulate` or `evaluate` |
| Change the demand | edit `demand:` in `scenario.yaml` |
| Try another reward | `reward: wait` in a copy of `configs/dqn.yaml` |
| Redraw the network figure | `uv run python scripts/plot_network.py` |
| Lint and format | `uv run ruff check . --fix && uv run ruff format .` |

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `SUMO not found` | Run `uv sync`; or set `SUMO_HOME` to a SUMO install |
| `libsumo` error about a simulation already running | Close the previous environment first; only one libsumo simulation per process |
| `uv run` reinstalls the big CUDA torch | `export UV_NO_SYNC=1` after `scripts/install_cpu.sh` |
| `--gui` does nothing on a server | `sumo-gui` needs a display; use `simulate` without `--gui` |
| `detect` says the vision extra is missing | `uv sync --extra vision` |
| Model download fails or checksum mismatch | Download the `yolox_s.onnx` from the YOLOX release yourself and pass `--model-file` |
