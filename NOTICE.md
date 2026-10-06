# Notices and credits

## Code

The code in this repository is MIT licensed (see [LICENSE](LICENSE)). It was written in 2026
and contains no code from the projects below; they are credited as the public work the
2021 version of this repository was built on.

- **Module 1 (2021): vehicle counting.** Built on
  [muffyharsha/traffic-annotation](https://github.com/muffyharsha/traffic-annotation), which
  bundled [darkflow](https://github.com/thtrieu/darkflow) (GPL-3.0), a TensorFlow port of YOLO.
- **Module 2 (2021): reinforcement learning signal control.** Built on
  [strangest-quark/TraffiQ-Control](https://github.com/strangest-quark/TraffiQ-Control),
  "Intelligent Traffic Light Control Using Reinforcement Learning", including its Kowdenahalli
  scenario idea and the OpenStreetMap extract.

That earlier code is still in this repository's git history (commit `9afc6d2`) under its
original terms.

## Data

- **Road network.** `src/junctioniq/scenarios/kowdenahalli/kowdenahalli.osm` is
  © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, available under the
  [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
  `kowdenahalli.net.xml` is derived from it with SUMO's netconvert and is offered under the
  same licence.
- **Demo video.** `data/bangalore-traffic-15s.mp4` is a 15-second, silent excerpt of traffic
  footage from the original project team, included for the vision demo.

## Models and tools (downloaded or installed, not included)

- **YOLOX** object detector by Megvii, [Apache-2.0](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE).
  The COCO-trained `yolox_s.onnx` weights are downloaded on first use and checked against a
  pinned SHA-256.
- **Eclipse SUMO**, the traffic simulator, [EPL-2.0](https://www.eclipse.org/sumo/) (installed
  from PyPI as `eclipse-sumo`, `libsumo`, `traci`, `sumolib`).
- **PyTorch**, **Gymnasium**, **ONNX Runtime**, **OpenCV** and **Supervision** (including its
  ByteTrack implementation), each under its own licence.
