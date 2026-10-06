"""From tracked vehicles to queue counts, with camera-motion compensation.

A vehicle is *queued* when it has been tracked for a moment and is (nearly) not moving
over the road. Hand-held or panning cameras make every vehicle appear to move, so the
camera's own motion is estimated from background features (with vehicles masked out)
and subtracted first, the same idea trackers such as BoT-SORT use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import supervision as sv

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


class CameraMotion:
    """Estimates a similarity transform between consecutive frames from background points."""

    def __init__(self, max_points: int = 400, scale: float = 0.5) -> None:
        self.max_points, self.scale = max_points, scale
        self._prev: np.ndarray | None = None

    def update(self, frame_bgr: np.ndarray, vehicle_boxes: np.ndarray) -> np.ndarray:
        """Return the 2x3 transform mapping previous-frame pixels to this frame."""
        small = cv2.resize(frame_bgr, None, fx=self.scale, fy=self.scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        prev, self._prev = self._prev, gray
        if prev is None:
            return IDENTITY.copy()
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        for x1, y1, x2, y2 in (vehicle_boxes * self.scale).astype(int):
            mask[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)] = 0
        pts = cv2.goodFeaturesToTrack(prev, self.max_points, 0.01, 6, mask=mask)
        if pts is None or len(pts) < 8:
            return IDENTITY.copy()
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None)
        good = status.reshape(-1) == 1
        if good.sum() < 8:
            return IDENTITY.copy()
        m, _ = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC)
        if m is None:
            return IDENTITY.copy()
        m = m.astype(np.float32)
        m[:, 2] /= self.scale  # translation back to full-resolution pixels
        return m


def apply(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:, :2].T + transform[:, 2]


@dataclass
class _Track:
    center: np.ndarray
    height: float
    age: int = 0
    lost: int = 0
    speed: float = 1.0  # smoothed, in vehicle heights per second


@dataclass
class QueueEstimator:
    """Marks tracked vehicles as queued and counts them, optionally per zone."""

    fps: float
    stopped_speed: float = 0.25  # vehicle heights per second
    min_age_s: float = 1.0
    smoothing_s: float = 1.0
    zones: dict[str, sv.PolygonZone] = field(default_factory=dict)
    _tracks: dict[int, _Track] = field(default_factory=dict)

    def update(self, detections: sv.Detections, transform: np.ndarray) -> np.ndarray:
        """Update with this frame's tracked detections; returns a boolean 'queued' mask."""
        alpha = min(1.0, 1.0 / max(1.0, self.smoothing_s * self.fps))
        centers = detections.get_anchors_coordinates(sv.Position.CENTER)
        heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
        queued = np.zeros(len(detections), dtype=bool)
        seen = set()
        for i, tid in enumerate(detections.tracker_id if len(detections) else []):
            tid = int(tid)
            seen.add(tid)
            track = self._tracks.get(tid)
            if track is None:
                self._tracks[tid] = _Track(centers[i].copy(), float(heights[i]))
                continue
            expected = apply(transform, track.center[None])[0]  # where it would be if parked
            moved = float(np.linalg.norm(centers[i] - expected))
            size = max(4.0, 0.5 * (track.height + float(heights[i])))
            speed = moved * self.fps / size
            track.speed = (1 - alpha) * track.speed + alpha * speed
            track.center, track.height = centers[i].copy(), float(heights[i])
            track.age += 1
            track.lost = 0
            queued[i] = track.age >= self.min_age_s * self.fps and track.speed < self.stopped_speed
        for tid in list(self._tracks):
            if tid in seen:
                continue
            track = self._tracks[tid]
            track.age, track.lost = 0, track.lost + 1  # ByteTrack may bring it back
            if track.lost > 2 * self.fps:
                del self._tracks[tid]
        return queued

    def zone_counts(self, detections: sv.Detections, queued: np.ndarray) -> dict[str, int]:
        if not self.zones:
            return {}
        return {
            name: int((zone.trigger(detections) & queued).sum())
            for name, zone in self.zones.items()
        }
