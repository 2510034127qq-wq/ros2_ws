"""Canonical registry of stationary thermal sources and temporary candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .position_filter import PositionFilter, associate


STATUS_CANDIDATE = "candidate"
STATUS_CONFIRMED = "confirmed"


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
    """Own source IDs; confirmed entries survive absence for this session."""

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
        candidate_timeout_s: float = 12.0,
        update_alpha_min: float = 0.08,
        max_detection_age_s: float = float("inf"),
        estimator_model: str = "legacy",
        position_noise_std: float = 0.05,
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
        self.candidate_timeout_s = float(candidate_timeout_s)
        self.update_alpha_min = max(0.0, min(0.6, float(update_alpha_min)))
        self.max_detection_age_s = float(max_detection_age_s)
        if estimator_model not in ("legacy", "kalman"):
            raise ValueError("estimator_model must be legacy or kalman")
        self.estimator_model = estimator_model
        self.position_noise_std = float(position_noise_std)
        self.measurement_variance = float(measurement_variance)
        self.association_gate_chi2 = float(association_gate_chi2)
        self.max_tracks = int(max_tracks)
        self.measurement_type = "field_direct"
        self._filters = {}
        self._last_filter_s = None
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
        if self.estimator_model=='kalman':
            return self._extract_components(temperature_mean,confidence,resolution,
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
        if not math.isfinite(now_s):
            raise ValueError("nonfinite timestamp")
        if self._last_filter_s is not None and now_s <= self._last_filter_s:
            return self.tracks
        self._last_filter_s = float(now_s)
        # Only unconfirmed candidates expire. Visibility is not source identity.
        for track in self.tracks:
            if track.status == STATUS_CANDIDATE and now_s-track.last_seen_s > self.candidate_timeout_s:
                self._tracks.pop(track.track_id)
                self._filters.pop(track.track_id)
        detections = sorted((d for d in detections if np.isfinite(
            [d.x,d.y,d.strength,d.confidence,d.sigma]).all() and d.confidence > 0),
            key=lambda d:d.confidence*d.strength, reverse=True)
        live = self.tracks
        for tr in live:
            self._filters[tr.track_id].predict(now_s, self.position_noise_std)
        matches, unmatched = associate([self._filters[t.track_id] for t in live], detections,
            self.measurement_variance, self.association_gate_chi2, self.gate_m,
            [t.strength for t in live])
        updated = set()
        for i, j in matches:
            tr, det = live[i], detections[j]
            filt = self._filters[tr.track_id]
            if self.estimator_model == 'legacy':
                alpha = max(self.update_alpha_min, min(.6, 1./(tr.observations+1)))
                filt.state = (1-alpha)*filt.state + alpha*np.array([det.x,det.y])
                filt.covariance = np.eye(2)*max(.12, 2./math.sqrt(tr.observations+1))
            else:
                filt.correct((det.x,det.y), self.measurement_variance/max(det.confidence,.1))
            tr.strength = .7*tr.strength + .3*det.strength
            tr.sigma = .7*tr.sigma + .3*det.sigma
            tr.confidence = det.confidence
            tr.existence_probability = min(.999,tr.existence_probability+.15)
            tr.observations += 1
            tr.consecutive_observations += 1
            tr.last_seen_s = now_s
            updated.add(tr.track_id)
        radius = self.duplicate_radius_m if self.estimator_model == 'legacy' else self.merge_radius_m
        for j in sorted(unmatched):
            if len(live) >= self.max_tracks:
                break
            det = detections[j]
            if any(np.linalg.norm(self._filters[t.track_id].state-[det.x,det.y]) <
                   (radius if t.status == STATUS_CONFIRMED else self.merge_radius_m)
                   for t in live):
                continue
            key = f"src_{self._next_id}"
            self._next_id += 1
            tr = TrackedSource(key,det.x,det.y,det.strength,det.sigma,
                               min(.65,.3+.35*det.confidence),det.confidence,1,now_s)
            self._tracks[key] = tr
            self._filters[key] = PositionFilter(np.array([det.x,det.y]),stamp_s=now_s)
            live.append(tr)
            updated.add(key)
        for tr in live:
            filt = self._filters[tr.track_id]
            tr.x,tr.y = map(float,filt.state)
            tr.covariance_xx = float(filt.covariance[0,0])
            tr.covariance_xy = float(filt.covariance[0,1])
            tr.covariance_yy = float(filt.covariance[1,1])
            if tr.track_id not in updated:
                tr.consecutive_observations = 0
            if (tr.status == STATUS_CANDIDATE
                    and tr.consecutive_observations >= self.confirm_observations
                    and tr.existence_probability >= self.confirm_probability
                    and max(tr.covariance_xx,tr.covariance_yy) <= self.confirm_covariance_max):
                if any(other.status == STATUS_CONFIRMED and other.track_id != tr.track_id
                       and math.hypot(other.x-tr.x,other.y-tr.y) < radius for other in self.tracks):
                    self._tracks.pop(tr.track_id)
                    self._filters.pop(tr.track_id)
                else:
                    tr.status = STATUS_CONFIRMED
        return self.tracks

    def update_from_map(self, temperature_mean, confidence, resolution, origin_x, origin_y,
                        now_s, last_seen_age_s=None):
        detections = self.extract_detections(temperature_mean,confidence,resolution,
                                             origin_x,origin_y,last_seen_age_s)
        return self.update(detections,now_s)

    def _extract_components(self, temperature, confidence, resolution, ox, oy, age=None):
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
