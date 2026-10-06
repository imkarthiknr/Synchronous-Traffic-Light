# Roadmap

JunctionIQ is planned in four milestones. Milestone 1 is in this repository now.

## Milestone 1: research core (done, v0.1.0)

- [x] Kowdenahalli scenario rebuilt from OpenStreetMap, left-hand traffic, reproducible demand
- [x] Gymnasium environment with safe switching (yellow, minimum green) and a joint action
- [x] Baselines: fixed-time (default and tuned), SUMO actuated, max-pressure, random
- [x] Double DQN in PyTorch, with validation-based model selection
- [x] Same-day evaluation on held-out seeds with published results
- [x] Vision: YOLOX + ByteTrack, camera-motion compensation, queue counts per zone
- [x] CLI, tests, CI and documentation

## Milestone 2: stronger learning

- [ ] PPO (Stable-Baselines3) and a per-junction (multi-agent) action space, which scales to
      larger networks than a joint action can
- [ ] Pressure and waiting-time rewards compared on the same evaluation days
- [ ] Robustness: demand far outside training, a lane closure, detector noise
- [ ] Hyper-parameter sweeps and several training seeds per result

## Milestone 3: from video to control

- [ ] Fine-tune the detector on Indian traffic (for example the IDD dataset), so small
      two-wheelers and auto-rickshaws are counted properly
- [ ] Feed queue counts from the vision pipeline into the environment's observation, and test
      the agent on queues measured from video instead of read from the simulator
- [ ] Zone calibration tool for a new fixed camera

## Milestone 4: product layer

- [ ] FastAPI service: run simulations and evaluations, stream junction state
- [ ] React 19 dashboard: live map of the junctions, queues, signal phases and controller
      comparison
- [ ] Docker image and a hosted demo
