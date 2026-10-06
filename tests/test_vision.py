from pathlib import Path

import cv2
import numpy as np
import pytest
import supervision as sv

from junctioniq.vision.detector import GROUP_ID, decode_yolox, letterbox, postprocess
from junctioniq.vision.queues import IDENTITY, CameraMotion, QueueEstimator

ROOT = Path(__file__).parents[1]


def test_letterbox_keeps_aspect_and_pads():
    img = np.zeros((360, 480, 3), dtype=np.uint8)
    out, ratio = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert ratio == pytest.approx(640 / 480)
    assert out[-1, -1].tolist() == [114, 114, 114]


def _raw_with_box(size=640, cx=100.0, cy=60.0, w=40.0, h=20.0, cls=2, score=0.9):
    """A fake YOLOX output with one confident cell at stride 8."""
    n = sum((size // s) ** 2 for s in (8, 16, 32))
    raw = np.zeros((n, 85), dtype=np.float32)
    raw[:, 2:4] = -10  # tiny boxes elsewhere
    gx, gy = int(cx // 8), int(cy // 8)
    i = gy * (size // 8) + gx
    raw[i, 0], raw[i, 1] = cx / 8 - gx, cy / 8 - gy
    raw[i, 2], raw[i, 3] = np.log(w / 8), np.log(h / 8)
    raw[i, 4] = score
    raw[i, 5 + cls] = 1.0
    return raw


def test_decode_and_postprocess_round_trip():
    decoded = decode_yolox(_raw_with_box(), 640)
    dets = postprocess(decoded, ratio=2.0, conf=0.3)
    assert len(dets) == 1
    assert np.allclose(dets.xyxy[0], [40, 25, 60, 35])  # (80..120, 50..70) / 2
    assert dets.class_id[0] == GROUP_ID["car"]


def test_postprocess_drops_non_vehicles():
    decoded = decode_yolox(_raw_with_box(cls=0), 640)  # class 0 = person
    assert len(postprocess(decoded, ratio=1.0)) == 0


def _dets(boxes, ids):
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        class_id=np.zeros(len(boxes), dtype=int),
        tracker_id=np.array(ids),
    )


def test_stationary_vehicle_becomes_queued_moving_does_not():
    est = QueueEstimator(fps=10)
    for t in range(20):
        queued = est.update(
            _dets([[0, 0, 20, 20], [50 + 10 * t, 0, 70 + 10 * t, 20]], [1, 2]), IDENTITY
        )
    assert queued.tolist() == [True, False]


def test_camera_pan_is_compensated():
    est = QueueEstimator(fps=10)
    shift = np.array([[1, 0, 5], [0, 1, 0]], dtype=np.float32)  # camera pans 5 px per frame
    for t in range(20):
        queued = est.update(_dets([[5 * t, 0, 5 * t + 20, 20]], [1]), shift)
    assert queued.tolist() == [True]


def test_zone_counts():
    zone = sv.PolygonZone(np.array([[0, 0], [100, 0], [100, 100], [0, 100]]))
    est = QueueEstimator(fps=10, zones={"north": zone})
    dets = _dets([[10, 10, 30, 30], [200, 10, 220, 30]], [1, 2])
    assert est.zone_counts(dets, np.array([True, True])) == {"north": 1}


def test_camera_motion_recovers_translation():
    rng = np.random.default_rng(0)
    base = (rng.random((240, 320)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    frame1 = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    moved = cv2.warpAffine(base, np.float32([[1, 0, 6], [0, 1, -4]]), (320, 240))
    frame2 = cv2.cvtColor(moved, cv2.COLOR_GRAY2BGR)
    cm = CameraMotion(scale=1.0)
    cm.update(frame1, np.zeros((0, 4)))
    m = cm.update(frame2, np.zeros((0, 4)))
    assert m[0, 2] == pytest.approx(6, abs=0.5) and m[1, 2] == pytest.approx(-4, abs=0.5)


@pytest.mark.model
def test_pipeline_on_demo_clip(tmp_path):
    from junctioniq.vision.model import ensure_model
    from junctioniq.vision.pipeline import run

    summary = run(
        ROOT / "data" / "bangalore-traffic-15s.mp4",
        tmp_path,
        ensure_model(),
        max_frames=20,
        log=lambda *_: None,
    )
    assert summary["frames"] == 20
    assert summary["mean_vehicles"] > 20  # a dense Bengaluru street
    assert (tmp_path / "queues.csv").is_file() and (tmp_path / "annotated.mp4").is_file()
