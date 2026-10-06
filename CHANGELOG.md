# Changelog

All notable changes to this project. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-10-06

JunctionIQ milestone 1: the 2021 college project rewritten as one tested Python package.

### Added
- `junctioniq` command with `build-network`, `simulate`, `train`, `evaluate` and `detect`.
- Kowdenahalli scenario: network rebuilt from the OpenStreetMap extract with left-hand traffic,
  plus a seeded, time-varying demand model with cars, two-wheelers and trucks.
- Gymnasium environment for all four signals, with yellow phases, a minimum green, three reward
  options and travel-time, stop-time and queue metrics.
- Baselines: fixed-time plans, SUMO actuated control, max-pressure and random.
- Double DQN in PyTorch with replay, soft target updates, validation and checkpoints.
- Evaluation of controllers on the same held-out days, with tables and plots in `results/`.
- Vision pipeline: YOLOX (ONNX) detection, ByteTrack tracking, camera-motion compensation and
  queue counts, optionally per zone; annotated video and CSV output.
- A 15-second demo clip from the team's Bengaluru footage.
- Tests, Ruff, GitHub Actions, README, development guide, roadmap and notices.

### Changed
- Signals now switch through a 3 s yellow after at least 10 s of green. The 2021 agent set
  phases directly every 2 s with no yellow.
- The learner is a Double DQN with batched updates, replacing a Keras DQN that fitted one sample
  at a time and had no target network.
- Detection uses YOLOX through ONNX Runtime instead of darkflow on TensorFlow 1.x.
- SUMO is installed from PyPI; no hard-coded Windows paths.

### Removed
- The bundled Windows SUMO install, the committed Windows Python 3.6 virtualenv, darkflow and
  its configs, the YOLO weights links, and about 170 MB of videos (5,625 files and about 500 MB in
  total). They remain in the git history.

## [2021-12-31] - original project

"Synchronous Traffic Light": vehicle counting with darkflow YOLO (module 1) and DQN signal
control in SUMO for four Kowdenahalli junctions (module 2), built on two public 2018
repositories (see [NOTICE.md](NOTICE.md)).

[0.1.0]: https://github.com/imkarthiknr/Synchronous-Traffic-Light/compare/9afc6d2...main
