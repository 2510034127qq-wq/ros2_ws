"""Lightweight thermal source candidate tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .motion_filter import MotionFilter, associate


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
    last_update_s: float
    covariance_xx: float = 2.0
    covariance_xy: float = 0.0
    covariance_yy: float = 2.0
    status: str = STATUS_CANDIDATE
    consecutive_observations: int = 1
    ever_confirmed: bool = False
    vx: float = 0.0
    vy: float = 0.0
    reacquisitions: int = 0
    last_reacquisition_s: float = 0.0
    first_confirmed_s: float = -1.0


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
        duplicate_radius_m: float = 2.5,
        confirm_probability: float = 0.75,
        confirm_observations: int = 5,
        confirm_covariance_max: float = 0.9,
        stale_after_s: float = 12.0,
        stale_decay_s: float = 20.0,
        duplicate_memory_s: float = 60.0,
        update_alpha_min: float = 0.08,
        max_detection_age_s: float = float("inf"),
        motion_model: str = "legacy",
        acceleration_std: float = 0.4,
        measurement_variance: float = 0.15,
        association_gate_chi2: float = 9.21,
        max_tracks: int = 32,
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
        self.duplicate_memory_s = float(duplicate_memory_s)
        self.update_alpha_min = max(0.0, min(0.6, float(update_alpha_min)))
        self.max_detection_age_s = float(max_detection_age_s)
        if motion_model not in ("legacy", "kalman"):
            raise ValueError("motion_model must be legacy or kalman")
        self.motion_model = motion_model
        self.acceleration_std = float(acceleration_std)
        self.measurement_variance = float(measurement_variance)
        self.association_gate_chi2 = float(association_gate_chi2)
        self.max_tracks = int(max_tracks)
        self.measurement_type = "field_direct"
        self._filters = {}
        self._last_motion_s = None
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
        last_seen_age_s: Optional[np.ndarray] = None,
    ) -> List[SourceDetection]:
        if self.motion_model=='kalman':
            return self._extract_motion_detections(temperature_mean,confidence,resolution,
                origin_x,origin_y,last_seen_age_s)
        temp = np.asarray(temperature_mean, dtype=np.float32)
        conf = np.asarray(confidence, dtype=np.float32)
        if temp.ndim != 2 or conf.shape != temp.shape or temp.size == 0:
            return []
        freshness = np.ones_like(temp, dtype=np.float32)
        hot = (temp >= self.ambient_temp + self.min_temp_rise) & (conf >= self.min_confidence)
        if last_seen_age_s is not None and math.isfinite(self.max_detection_age_s):
            age = np.asarray(last_seen_age_s, dtype=np.float32)
            if age.shape == temp.shape:
                hot &= (age >= 0.0) & (age <= self.max_detection_age_s)
                freshness = np.exp(
                    -np.clip(age, 0.0, self.max_detection_age_s)
                    / max(self.max_detection_age_s, 1e-3)
                ).astype(np.float32)
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
        scores = (temp[ys, xs] - self.ambient_temp) * conf[ys, xs] * freshness[ys, xs]
        order = np.argsort(scores)[::-1]
        detections: List[SourceDetection] = []
        for idx in order[: self.max_detections * 3]:
            x = origin_x + (float(xs[idx]) + 0.5) * resolution
            y = origin_y + (float(ys[idx]) + 0.5) * resolution
            det = SourceDetection(
                x=x,
                y=y,
                strength=float(temp[ys[idx], xs[idx]] - self.ambient_temp),
                confidence=float(conf[ys[idx], xs[idx]] * freshness[ys[idx], xs[idx]]),
                sigma=max(float(resolution) * 2.0, 0.5),
            )
            if any(math.hypot(det.x - prev.x, det.y - prev.y) < self.merge_radius_m for prev in detections):
                continue
            detections.append(det)
            if len(detections) >= self.max_detections:
                break
        return detections

    def update(self, detections: Sequence[SourceDetection], now_s: float) -> List[TrackedSource]:
        if self.motion_model == "kalman":
            return self._update_motion(detections, now_s)
        updated_ids = set()
        for det in sorted(detections, key=lambda d: d.confidence * d.strength, reverse=True):
            track = self._nearest_track(det)
            if track is not None and track.track_id in updated_ids:
                continue
            if track is None:
                if self._near_confirmed(det, now_s):
                    continue
                track = self._new_track(det, now_s)
            else:
                self._update_track(track, det, now_s)
            updated_ids.add(track.track_id)

        for track in self._tracks.values():
            if track.track_id in updated_ids:
                continue
            dt_since_seen = max(0.0, now_s - track.last_seen_s)
            step_dt = max(0.0, now_s - track.last_update_s)
            track.last_update_s = now_s
            track.consecutive_observations = 0
            if track.status != STATUS_SUPPRESSED:
                track.existence_probability *= math.exp(-step_dt / self.stale_decay_s)
                track.confidence *= math.exp(-step_dt / self.stale_decay_s)
            if dt_since_seen >= self.stale_after_s and track.status != STATUS_SUPPRESSED:
                track.status = STATUS_STALE

        self._merge_close_tracks()
        self._promote_confirmed(now_s)
        self._suppress_duplicates(now_s)
        return self.tracks

    def update_from_map(
        self,
        temperature_mean: np.ndarray,
        confidence: np.ndarray,
        resolution: float,
        origin_x: float,
        origin_y: float,
        now_s: float,
        last_seen_age_s: Optional[np.ndarray] = None,
    ) -> List[TrackedSource]:
        detections = self.extract_detections(
            temperature_mean, confidence, resolution, origin_x, origin_y, last_seen_age_s)
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

    def _near_confirmed(self, det: SourceDetection, now_s: float) -> bool:
        return any(
            self._blocks_duplicate_birth(track, now_s)
            and math.hypot(det.x - track.x, det.y - track.y) <= self.duplicate_radius_m
            for track in self._tracks.values()
        )

    def _blocks_duplicate_birth(self, track: TrackedSource, now_s: Optional[float] = None) -> bool:
        if track.status == STATUS_SUPPRESSED:
            return False
        if track.ever_confirmed:
            if now_s is None or not math.isfinite(self.duplicate_memory_s):
                return True
            return max(0.0, now_s - track.last_seen_s) <= self.duplicate_memory_s
        return track.status in (STATUS_CONFIRMED, STATUS_STALE) and track.observations >= self.confirm_observations

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
            last_update_s=now_s,
            covariance_xx=2.0,
            covariance_yy=2.0,
        )
        self._tracks[track_id] = track
        return track

    def _update_track(self, track: TrackedSource, det: SourceDetection, now_s: float) -> None:
        n = max(1, track.observations)
        alpha = max(self.update_alpha_min, min(0.6, 1.0 / (n + 1.0)))
        track.x = (1.0 - alpha) * track.x + alpha * det.x
        track.y = (1.0 - alpha) * track.y + alpha * det.y
        track.strength = max(track.strength * 0.9, det.strength)
        track.sigma = (1.0 - alpha) * track.sigma + alpha * det.sigma
        track.confidence = max(track.confidence * 0.85, det.confidence)
        track.observations += 1
        track.consecutive_observations += 1
        track.last_seen_s = now_s
        track.last_update_s = now_s
        track.existence_probability = min(
            0.99, track.existence_probability + 0.06 + 0.10 * det.confidence
        )
        cov = max(0.12, 2.0 / math.sqrt(track.observations))
        track.covariance_xx = cov
        track.covariance_yy = cov
        if track.status == STATUS_STALE:
            track.status = STATUS_CONFIRMED if track.ever_confirmed else STATUS_CANDIDATE

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
                keep.last_seen_s = max(keep.last_seen_s, drop.last_seen_s)
                keep.last_update_s = max(keep.last_update_s, drop.last_update_s)
                keep.covariance_xx = min(keep.covariance_xx, drop.covariance_xx)
                keep.covariance_yy = min(keep.covariance_yy, drop.covariance_yy)
                keep.ever_confirmed = keep.ever_confirmed or drop.ever_confirmed
                drop.status = STATUS_SUPPRESSED
                drop.existence_probability = min(drop.existence_probability, 0.05)

    def _promote_confirmed(self, now_s: float) -> None:
        confirmed_positions = [
            (t.x, t.y) for t in self._tracks.values()
            if self._blocks_duplicate_birth(t, now_s)
        ]
        for track in self._tracks.values():
            if track.status not in (STATUS_CANDIDATE, STATUS_CONFIRMED):
                continue
            if track.status == STATUS_CONFIRMED:
                continue
            near_confirmed = any(
                math.hypot(track.x - x, track.y - y) <= self.duplicate_radius_m
                for x, y in confirmed_positions
            )
            if near_confirmed:
                continue
            cov_ok = max(track.covariance_xx, track.covariance_yy) <= self.confirm_covariance_max
            if (
                track.existence_probability >= self.confirm_probability
                and track.observations >= self.confirm_observations
                and track.consecutive_observations >= self.confirm_observations
                and cov_ok
            ):
                track.status = STATUS_CONFIRMED
                track.ever_confirmed = True
                confirmed_positions.append((track.x, track.y))

    def _suppress_duplicates(self, now_s: float) -> None:
        confirmed = [
            t for t in self._tracks.values()
            if self._blocks_duplicate_birth(t, now_s)
        ]
        for track in self._tracks.values():
            if self._blocks_duplicate_birth(track, now_s):
                continue
            if any(
                c.track_id != track.track_id
                and math.hypot(track.x - c.x, track.y - c.y) <= self.duplicate_radius_m
                for c in confirmed
            ):
                track.status = STATUS_SUPPRESSED
                track.existence_probability = min(track.existence_probability, 0.05)

    def _update_motion(self, detections, now_s):
        if self._last_motion_s is not None and now_s <= self._last_motion_s:
            return self.tracks
        self._last_motion_s = float(now_s)
        detections = [d for d in detections if np.isfinite(
            [d.x, d.y, d.strength, d.confidence, d.sigma]).all() and d.confidence > 0]
        live = [t for t in self.tracks if t.status != STATUS_SUPPRESSED]
        for tr in live:
            self._filters[tr.track_id].predict(now_s, self.acceleration_std)
        matches, unmatched = associate(
            [self._filters[t.track_id] for t in live], detections,
            self.measurement_variance, self.association_gate_chi2,
            max_distance=self.gate_m, strength=[t.strength for t in live])
        updated = set()
        for i, j in matches:
            tr, det = live[i], detections[j]
            gap = now_s - tr.last_seen_s
            if tr.status == STATUS_STALE:
                tr.reacquisitions += 1
                tr.last_reacquisition_s = gap
            filt = self._filters[tr.track_id]
            filt.correct((det.x, det.y), self.measurement_variance / max(det.confidence, 0.1))
            tr.strength = 0.7*tr.strength + 0.3*det.strength
            tr.sigma = 0.7*tr.sigma + 0.3*det.sigma
            tr.confidence = det.confidence
            tr.existence_probability = min(0.999, tr.existence_probability + 0.15)
            tr.observations += 1
            tr.consecutive_observations += 1
            tr.last_seen_s = now_s
            tr.status = STATUS_CONFIRMED if tr.ever_confirmed else STATUS_CANDIDATE
            updated.add(tr.track_id)
        for j in sorted(unmatched):
            if len(live) >= self.max_tracks:
                break
            det = detections[j]
            # Suppress only duplicate peaks, not the whole neighbourhood of a
            # historical confirmation: nearby moving sources may be distinct.
            if any(np.hypot(det.x-self._filters[t.track_id].state[0],
                            det.y-self._filters[t.track_id].state[1]) < self.merge_radius_m*0.35
                   for t in live):
                continue
            tr = self._new_track(det, now_s)
            self._filters[tr.track_id] = MotionFilter(
                np.array([det.x, det.y, 0., 0.]), stamp_s=now_s)
            live.append(tr)
            updated.add(tr.track_id)
        for tr in live:
            filt = self._filters[tr.track_id]
            if tr.track_id not in updated:
                dt = max(0., now_s - tr.last_update_s)
                tr.existence_probability *= math.exp(-dt/self.stale_decay_s)
                tr.confidence *= math.exp(-dt/self.stale_decay_s)
                tr.consecutive_observations = 0
                if now_s-tr.last_seen_s >= self.stale_after_s:
                    tr.status = STATUS_STALE
            tr.x, tr.y, tr.vx, tr.vy = map(float, filt.state)
            tr.covariance_xx = float(filt.covariance[0, 0])
            tr.covariance_xy = float(filt.covariance[0, 1])
            tr.covariance_yy = float(filt.covariance[1, 1])
            tr.last_update_s = now_s
            if (tr.status == STATUS_CANDIDATE
                    and tr.existence_probability >= self.confirm_probability
                    and tr.consecutive_observations >= self.confirm_observations
                    and max(tr.covariance_xx, tr.covariance_yy) <= self.confirm_covariance_max):
                tr.status = STATUS_CONFIRMED
                tr.ever_confirmed = True
                tr.first_confirmed_s = now_s
            if now_s-tr.last_seen_s > self.duplicate_memory_s and tr.existence_probability < 0.05:
                self._tracks.pop(tr.track_id, None)
                self._filters.pop(tr.track_id, None)
        return self.tracks

    def _extract_motion_detections(self, temperature, confidence, resolution, ox, oy, age=None):
        temp=np.asarray(temperature,dtype=float);conf=np.asarray(confidence,dtype=float)
        if temp.ndim!=2 or conf.shape!=temp.shape or temp.size==0:return []
        observed=np.isfinite(temp)&np.isfinite(conf)&(conf>=self.min_confidence)
        if age is not None:
            age=np.asarray(age)
            observed &= (age>=0)&(age<=self.max_detection_age_s)
        rise=np.maximum(temp-self.ambient_temp,0)
        hot=observed&(rise>=self.min_temp_rise)
        remaining=set(map(tuple,np.argwhere(hot)))
        components=[]
        while remaining:
            start=min(remaining);remaining.remove(start);stack=[start];component=[]
            while stack:
                cell=stack.pop();component.append(cell)
                for dy in (-1,0,1):
                    for dx in (-1,0,1):
                        neighbour=(cell[0]+dy,cell[1]+dx)
                        if neighbour in remaining:
                            remaining.remove(neighbour);stack.append(neighbour)
            components.append(component)
        results=[]
        for component in components:
            cells=np.asarray(component);ys,xs=cells[:,0],cells[:,1]
            if self.measurement_type=='surface_radiance':
                # One connected hot surface -> one detection, rather than one
                # track per noisy local maximum on the same flat hot face.
                weights=rise[ys,xs]*conf[ys,xs]
                cy=float(np.average(ys,weights=weights));cx=float(np.average(xs,weights=weights))
                extent=max(resolution*.5,float(np.sqrt(np.average(
                    (ys-cy)**2+(xs-cx)**2,weights=weights)))*resolution)
                results.append(SourceDetection(ox+(cx+.5)*resolution,oy+(cy+.5)*resolution,
                    float(np.percentile(rise[ys,xs],90)),float(np.mean(conf[ys,xs])),extent))
                continue
            peaks=[]
            for y,x in component:
                if y<1 or x<1 or y>=temp.shape[0]-1 or x>=temp.shape[1]-1:continue
                # A partial footprint edge is not evidence of a source maximum.
                if not observed[y-1:y+2,x-1:x+2].all():continue
                if temp[y,x]>=np.max(temp[y-1:y+2,x-1:x+2]):peaks.append((rise[y,x],y,x))
            kept=[]
            for amplitude,y,x in sorted(peaks,reverse=True):
                duplicate=False
                for prev_amp,py,px in kept:
                    distance=np.hypot(y-py,x-px)*resolution
                    steps=max(2,int(np.ceil(np.hypot(y-py,x-px)))+1)
                    iy=np.rint(np.linspace(y,py,steps)).astype(int);ix=np.rint(np.linspace(x,px,steps)).astype(int)
                    if distance<self.merge_radius_m or np.min(rise[iy,ix])>.8*min(amplitude,prev_amp):
                        duplicate=True;break
                if duplicate:continue
                kept.append((amplitude,y,x))
                local=(np.abs(ys-y)<=3)&(np.abs(xs-x)<=3)&(rise[ys,xs]>=amplitude*.7)
                weights=rise[ys[local],xs[local]]*conf[ys[local],xs[local]]
                cy=float(np.average(ys[local],weights=weights));cx=float(np.average(xs[local],weights=weights))
                results.append(SourceDetection(ox+(cx+.5)*resolution,oy+(cy+.5)*resolution,
                    float(amplitude),float(conf[y,x]),max(resolution*2,.5)))
        return sorted(results,key=lambda d:d.strength*d.confidence,reverse=True)[:self.max_detections]
