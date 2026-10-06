"""Video in, queue counts out: detect, track, compensate camera motion, count queued vehicles.

Outputs, in ``out_dir``:

- ``queues.csv``: one row per frame with vehicle and queue counts (and per-zone queues),
- ``annotated.mp4``: the video with queued vehicles in red and moving ones in green,
- ``summary.json``: averages and the number of distinct vehicles by type.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import yaml

from junctioniq.vision.detector import GROUPS, YOLOXDetector
from junctioniq.vision.queues import CameraMotion, QueueEstimator

QUEUED_BGR = (60, 60, 230)
MOVING_BGR = (90, 200, 90)


def load_zones(path: Path | None) -> dict[str, sv.PolygonZone]:
    """Zones file: ``{zone_name: [[x, y], [x, y], ...]}`` in pixel coordinates of the video."""
    if path is None:
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {
        str(name): sv.PolygonZone(
            np.array(points, dtype=np.int32), triggering_anchors=(sv.Position.BOTTOM_CENTER,)
        )
        for name, points in data.items()
    }


def draw(frame: np.ndarray, detections: sv.Detections, queued: np.ndarray, zones, text: str):
    out = frame.copy()
    for name, zone in zones.items():
        cv2.polylines(out, [zone.polygon], True, (240, 200, 60), 2)
        x, y = zone.polygon[0]
        cv2.putText(
            out, name, (int(x), int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 200, 60), 1
        )
    for (x1, y1, x2, y2), q in zip(detections.xyxy.astype(int), queued, strict=True):
        cv2.rectangle(out, (x1, y1), (x2, y2), QUEUED_BGR if q else MOVING_BGR, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(out, (0, 0), (tw + 12, th + 12), (20, 20, 20), -1)
    cv2.putText(
        out, text, (6, th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )
    return out


def run(
    video: Path,
    out_dir: Path,
    model_path: Path,
    zones_file: Path | None = None,
    conf: float = 0.3,
    max_frames: int | None = None,
    write_video: bool = True,
    log=print,
) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLOXDetector(model_path, conf=conf)
    tracker = sv.ByteTrack(frame_rate=round(fps))
    motion = CameraMotion()
    zones = load_zones(zones_file)
    estimator = QueueEstimator(fps=fps, zones=zones)
    writer = (
        cv2.VideoWriter(
            str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if write_video
        else None
    )

    fields = ["frame", "time_s", "vehicles", "queued", *GROUPS, *(f"queue_{z}" for z in zones)]
    track_class: dict[int, int] = {}
    totals = defaultdict(list)
    frame_no = 0
    with (out_dir / "queues.csv").open("w", newline="") as fh:
        rows = csv.DictWriter(fh, fieldnames=fields)
        rows.writeheader()
        while max_frames is None or frame_no < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            detections = tracker.update_with_detections(detector(frame))
            transform = motion.update(frame, detections.xyxy)
            queued = estimator.update(detections, transform)
            for tid, cid in zip(detections.tracker_id, detections.class_id, strict=True):
                track_class.setdefault(int(tid), int(cid))

            by_group = {g: int((detections.class_id == i).sum()) for i, g in enumerate(GROUPS)}
            row = {
                "frame": frame_no,
                "time_s": round(frame_no / fps, 3),
                "vehicles": len(detections),
                "queued": int(queued.sum()),
                **by_group,
                **{f"queue_{k}": v for k, v in estimator.zone_counts(detections, queued).items()},
            }
            rows.writerow(row)
            totals["vehicles"].append(row["vehicles"])
            totals["queued"].append(row["queued"])
            if writer is not None:
                text = (
                    f"t={row['time_s']:5.1f}s  vehicles {row['vehicles']}  queued {row['queued']}  "
                    f"(car {by_group['car']}, 2W {by_group['two-wheeler']}, "
                    f"heavy {by_group['heavy']})"
                )
                writer.write(draw(frame, detections, queued, zones, text))
            frame_no += 1
            if frame_no % 50 == 0:
                log(f"  frame {frame_no}: {row['vehicles']} vehicles, {row['queued']} queued")
    cap.release()
    if writer is not None:
        writer.release()

    distinct = defaultdict(int)
    for cid in track_class.values():
        distinct[GROUPS[cid]] += 1
    summary = {
        "video": Path(video).name,
        "frames": frame_no,
        "fps": fps,
        "mean_vehicles": float(np.mean(totals["vehicles"])) if frame_no else 0.0,
        "mean_queued": float(np.mean(totals["queued"])) if frame_no else 0.0,
        "max_queued": int(max(totals["queued"], default=0)),
        "distinct_vehicles": dict(distinct),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
