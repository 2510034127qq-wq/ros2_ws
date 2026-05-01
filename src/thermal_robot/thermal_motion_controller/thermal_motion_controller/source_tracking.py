"""Lightweight thermal source candidate tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


STATUS_CANDIDATE = "candidate"
STATUS_CONFIRMED = "confirmed"
STATUS_STALE = "stale"
STATUS_SUPPRESSED = "suppressed"


@dataclass
class SourceDetection:
    x: float
    y: float
    strength: float
    confidence: float
    sigma: float = 1.0


@dataclass
class TrackedSource:
    track_id: str
    x: float
    y: float
    strength: float
    sigma: float
    existence_probability: float
    confidence: float
    observations: int
    last_seen_s: float
    covariance_xx: float = 2.0
    covariance_xy: float = 0.0
    covariance_yy: float = 2.0
    status: str = STATUS_CANDIDATE
    consecutive_observations: int = 1


class SourceTrackerCore:
    """Nearest-neighbor multi-hypothesis tracker for source-like map peaks."""

    def __init__(
        self,
        ambient_temp: float = 22.0,
        min_temp_rise: float = 4.0,
        min_confidence: float = 0.12,
        max_detections: int = 12,
        gate_m: float = 1.25,
        merge_radius_m: float = 1.0,
        duplicate_radius_m: float = 2.0,
        confirm_probability: float = 0.75,
        confirm_observations: int = 5,
        confirm_covariance_max: float = 0.9,
        stale_after_s: float = 12.0,
        stale_decay_s: float = 20.0,
    ):
        self.ambient_temp = float(ambient_temp)
        self.min_temp_rise = float(min_temp_rise)
        self.min_confidence = float(min_confidence)
        self.max_detections = int(max_detections)
        self.gate_m = float(gate_m)
        self.merge_radius_m = float(merge_radius_m)
        self.duplicate_radius_m = float(duplicate_radius_m)
        self.confirm_probability = float(confirm_probability)
        self.confirm_observations = int(confirm_observations)
        self.confirm_covariance_max = float(confirm_covariance_max)
        self.stale_after_s = float(stale_after_s)
        self.stale_decay_s = max(1e-3, float(stale_decay_s))
        self._tracks: Dict[str, TrackedSource] = {}
        self._next_id = 1

    @property
    def tracks(self) -> List[TrackedSource]:
        return list(self._tracks.values())

    def extract_detections(
        self,
        temperature_mean: np.ndarray,
        confidence: np.ndarray,
        resolution: float,
        origin_x: float,
        origin_y: float,
    ) -> List[SourceDetection]:
        temp = np.asarray(temperature_mean, dtype=np.float32)
        conf = np.asarray(confidence, dtype=np.float32)
        if temp.ndim != 2 or conf.shape != temp.shape or temp.size == 0:
            return []
        hot = (temp >= self.ambient_temp + self.min_temp_rise) & (conf >= self.min_confidence)
        if temp.shape[0] >= 3 and temp.shape[1] >= 3:
            center = temp[1:-1, 1:-1]
            local = np.ones_like(center, dtype=bool)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    local &= center >= temp[1 + dy: temp.shape[0] - 1 + dy,
                                            1 + dx: temp.shape[1] - 1 + dx]
            peak_mask = np.zeros_like(hot, dtype=bool)
            peak_mask[1:-1, 1:-1] = local
            hot &= peak_mask
        ys, xs = np.where(hot)
        if len(xs) == 0:
            return []
        scores = (temp[ys, xs] - self.ambient_temp) * conf[ys, xs]
        order = np.argsort(scores)[::-1]
        detections: List[SourceDetection] = []
        for idx in order[: self.max_detections * 3]:
            x = origin_x + (float(xs[idx]) + 0.5) * resolution
            y = origin_y + (float(ys[idx]) + 0.5) * resolution
            det = SourceDetection(
                x=x,
                y=y,
                strength=float(temp[ys[idx], xs[idx]] - self.ambient_temp),
                confidence=float(conf[ys[idx], xs[idx]]),
                sigma=max(float(resolution) * 2.0, 0.5),
            )
            if any(math.hypot(det.x - prev.x, det.y - prev.y) < self.merge_radius_m for prev in detections):
                continue
            detections.append(det)
            if len(detections) >= self.max_detections:
                break
        return detections

    def update(self, detections: Sequence[SourceDetection], now_s: float) -> List[TrackedSource]:
        updated_ids = set()
        for det in sorted(detections, key=lambda d: d.confidence * d.strength, reverse=True):
            track = self._nearest_track(det)
            if track is None:
                track = self._new_track(det, now_s)
            else:
                self._update_track(track, det, now_s)
            updated_ids.add(track.track_id)

        for track in self._tracks.values():
            if track.track_id in updated_ids:
                continue
            dt = max(0.0, now_s - track.last_seen_s)
            track.consecutive_observations = 0
            if track.status != STATUS_SUPPRESSED:
                track.existence_probability *= math.exp(-dt / self.stale_decay_s)
                track.confidence *= math.exp(-dt / self.stale_decay_s)
            if dt >= self.stale_after_s and track.status != STATUS_SUPPRESSED:
                track.status = STATUS_STALE

        self._merge_close_tracks()
        self._promote_confirmed()
        self._suppress_duplicates()
        return self.tracks

    def update_from_map(
        self,
        temperature_mean: np.ndarray,
        confidence: np.ndarray,
        resolution: float,
        origin_x: float,
        origin_y: float,
        now_s: float,
    ) -> List[TrackedSource]:
        detections = self.extract_detections(temperature_mean, confidence, resolution, origin_x, origin_y)
        return self.update(detections, now_s)

    def _nearest_track(self, det: SourceDetection) -> Optional[TrackedSource]:
        best = None
        best_d = float("inf")
        for track in self._tracks.values():
            if track.status == STATUS_SUPPRESSED:
                continue
            cov = max(0.2, 0.5 * (track.covariance_xx + track.covariance_yy))
            d = math.hypot(det.x - track.x, det.y - track.y) / math.sqrt(cov)
            if d < best_d:
                best = track
                best_d = d
        if best is not None and math.hypot(det.x - best.x, det.y - best.y) <= self.gate_m:
            return best
        return None

    def _new_track(self, det: SourceDetection, now_s: float) -> TrackedSource:
        track_id = f"src_{self._next_id}"
        self._next_id += 1
        prob = min(0.65, 0.3 + 0.35 * det.confidence)
        track = TrackedSource(
            track_id=track_id,
            x=det.x,
            y=det.y,
            strength=det.strength,
            sigma=det.sigma,
            existence_probability=prob,
            confidence=det.confidence,
            observations=1,
            last_seen_s=now_s,
            covariance_xx=2.0,
            covariance_yy=2.0,
        )
        self._tracks[track_id] = track
        return track

    def _update_track(self, track: TrackedSource, det: SourceDetection, now_s: float) -> None:
        n = max(1, track.observations)
        alpha = min(0.6, 1.0 / (n + 1.0))
        track.x = (1.0 - alpha) * track.x + alpha * det.x
        track.y = (1.0 - alpha) * track.y + alpha * det.y
        track.strength = max(track.strength * 0.9, det.strength)
        track.sigma = (1.0 - alpha) * track.sigma + alpha * det.sigma
        track.confidence = max(track.confidence * 0.85, det.confidence)
        track.observations += 1
        track.consecutive_observations += 1
        track.last_seen_s = now_s
        track.existence_probability = min(
            0.99, track.existence_probability + 0.06 + 0.10 * det.confidence
        )
        cov = max(0.12, 2.0 / math.sqrt(track.observations))
        track.covariance_xx = cov
        track.covariance_yy = cov
        if track.status == STATUS_STALE:
            track.status = STATUS_CANDIDATE

    def _merge_close_tracks(self) -> None:
        tracks = sorted(
            self._tracks.values(),
            key=lambda t: (t.status == STATUS_CONFIRMED, t.existence_probability, t.observations),
            reverse=True,
        )
        for i, keep in enumerate(tracks):
            if keep.status == STATUS_SUPPRESSED:
                continue
            for drop in tracks[i + 1:]:
                if drop.status == STATUS_SUPPRESSED:
                    continue
                if math.hypot(keep.x - drop.x, keep.y - drop.y) > self.merge_radius_m:
                    continue
                total_obs = max(1, keep.observations + drop.observations)
                keep.x = (keep.x * keep.observations + drop.x * drop.observations) / total_obs
                keep.y = (keep.y * keep.observations + drop.y * drop.observations) / total_obs
                keep.strength = max(keep.strength, drop.strength)
                keep.confidence = max(keep.confidence, drop.confidence)
                keep.existence_probability = max(keep.existence_probability, drop.existence_probability)
                keep.observations = total_obs
                keep.consecutive_observations = max(keep.consecutive_observations, drop.consecutive_observations)
                keep.covariance_xx = min(keep.covariance_xx, drop.covariance_xx)
                keep.covariance_yy = min(keep.covariance_yy, drop.covariance_yy)
                drop.status = STATUS_SUPPRESSED
                drop.existence_probability = min(drop.existence_probability, 0.05)

    def _promote_confirmed(self) -> None:
        confirmed_positions = [
            (t.x, t.y) for t in self._tracks.values()
            if t.status == STATUS_CONFIRMED
        ]
        for track in self._tracks.values():
            if track.status not in (STATUS_CANDIDATE, STATUS_CONFIRMED):
                continue
            if track.status == STATUS_CONFIRMED:
                continue
            near_confirmed = any(
                math.hypot(track.x - x, track.y - y) < self.duplicate_radius_m
                for x, y in confirmed_positions
            )
            if near_confirmed:
                continue
            cov_ok = max(track.covariance_xx, track.covariance_yy) <= self.confirm_covariance_max
            if (
                track.existence_probability >= self.confirm_probability
                and track.observations >= self.confirm_observations
                and cov_ok
            ):
                track.status = STATUS_CONFIRMED
                confirmed_positions.append((track.x, track.y))

    def _suppress_duplicates(self) -> None:
        confirmed = [
            t for t in self._tracks.values()
            if t.status == STATUS_CONFIRMED
        ]
        for track in self._tracks.values():
            if track.status == STATUS_CONFIRMED:
                continue
            if any(math.hypot(track.x - c.x, track.y - c.y) < self.duplicate_radius_m for c in confirmed):
                track.status = STATUS_SUPPRESSED
                track.existence_probability = min(track.existence_probability, 0.05)
