"""Bounded labelled multi-Bernoulli source clusters (Rao-Blackwell branch).

Positions are conditional Gaussians, existence is Bernoulli, and the
cardinality PMF is their exact convolution under the cluster independence
approximation. Association is a gated maximum-score approximation, not an
exact multi-object posterior. No ground truth or ROS types enter this module.
"""
from dataclasses import dataclass, field
import copy
import math
import time
import numpy as np

from .position_filter import PositionFilter, associate


@dataclass
class BeliefParams:
    max_sources: int = 16
    position_noise_std: float = 0.05
    measurement_variance: float = 0.2
    gate_chi2: float = 9.21
    gate_m: float = 3.0
    confidence_memory_s: float = 180.0
    p_detection: float = 0.85
    false_alarm_probability: float = 0.03
    candidate_probability: float = 0.15
    prune_probability: float = 0.025
    confirm_probability: float = 0.90
    confirm_hits: int = 3
    merge_distance_m: float = 0.45
    merge_hits: int = 3
    budget_ms: float = 100.0
    field_fit_radius_m: float = 3.0
    sigma_min_m: float = 0.25
    sigma_max_m: float = 3.0


@dataclass
class SourceCluster:
    label: str
    position: PositionFilter
    probability: float
    amplitude: float
    sigma: float
    last_seen_s: float
    hits: int = 1
    confirmed: bool = False
    amplitude_variance: float = 16.0
    sigma_variance: float = 0.5


def bernoulli_entropy(p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return -p*np.log(p)-(1-p)*np.log1p(-p)


def detection_information(p, pd, pf):
    """Expected Bernoulli entropy reduction for a binary detection experiment."""
    hit = p*pd + (1-p)*pf
    ph = p*pd/max(hit, 1e-12)
    pm = p*(1-pd)/max(1-hit, 1e-12)
    return float(max(0., bernoulli_entropy(p) - hit*bernoulli_entropy(ph)
                     - (1-hit)*bernoulli_entropy(pm)))


class SourceBelief:
    def __init__(self, params=None):
        self.params = params or BeliefParams()
        self.clusters = []
        self.stamp_s = None
        self.next_label = 1
        self.merge_support = {}
        self.last_ms = 0.0
        self.health = "not_started"

    def update(self, detections, stamp_s, visibility=None, birth_detections=None,
               field_snapshot=None):
        """Transactional update: late/invalid/over-budget steps never publish.

        visibility(x,y) is current sensor detection opportunity in [0,1], not
        historical coverage. Occlusion only ages confidence; it is not a negative observation.
        birth_detections must be unexplained positive residual detections when a
        mapped field is supplied. Online caller can continue its independent fast
        loop while this computation runs in a separate ROS process.
        """
        start = time.perf_counter()
        if self.stamp_s is not None and stamp_s <= self.stamp_s:
            self.health = "stale_input"
            return False
        try:
            candidate = copy.deepcopy(self)
            candidate._step(detections, float(stamp_s), visibility,
                            birth_detections, field_snapshot)
            ms = (time.perf_counter()-start)*1000
            if ms > self.params.budget_ms:
                self.health, self.last_ms = "over_budget", ms
                return False
            self.__dict__.update(candidate.__dict__)
            self.last_ms, self.health = ms, "ready"
            return True
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            self.health = "invalid:" + type(exc).__name__
            self.last_ms = (time.perf_counter()-start)*1000
            return False

    def _step(self, detections, stamp, visibility, birth_detections, snapshot):
        p = self.params
        if not math.isfinite(stamp):
            raise ValueError("nonfinite timestamp")
        for d in detections:
            if not np.isfinite([d.x, d.y, d.strength, d.sigma, d.confidence]).all():
                raise ValueError("nonfinite detection")
        dt = 0 if self.stamp_s is None else stamp-self.stamp_s
        for c in self.clusters:
            c.position.predict(stamp, p.position_noise_std)
            c.probability *= math.exp(-dt/max(p.confidence_memory_s, 1e-6))
        matches, unmatched = associate(
            [c.position for c in self.clusters], detections,
            p.measurement_variance, p.gate_chi2, p.gate_m,
            [c.amplitude for c in self.clusters])
        matched = set()
        for i, j in matches:
            c, d = self.clusters[i], detections[j]
            c.position.correct((d.x, d.y), p.measurement_variance/max(d.confidence, .1))
            prior = c.probability
            c.probability = prior*p.p_detection / max(
                prior*p.p_detection+(1-prior)*p.false_alarm_probability, 1e-12)
            c.hits += 1
            c.last_seen_s = stamp
            # Conditional scalar Gaussian amplitude and scale updates.
            for name, value, variance in (("amplitude", d.strength, 1/max(d.confidence,.1)),
                                           ("sigma", d.sigma, .25)):
                pv = getattr(c, name+"_variance")+.02*dt
                k = pv/(pv+variance)
                setattr(c, name, max(.01, getattr(c,name)+k*(value-getattr(c,name))))
                setattr(c, name+"_variance", (1-k)*pv)
            matched.add(i)
        for i, c in enumerate(self.clusters):
            if i not in matched:
                opportunity = 0.0 if visibility is None else float(visibility(*c.position.state[:2]))
                pd = p.p_detection*np.clip(opportunity, 0, 1)
                c.probability = c.probability*(1-pd)/max(1-c.probability*pd,1e-12)
        births = detections if birth_detections is None else birth_detections
        for j in sorted(unmatched):
            d = detections[j]
            if len(self.clusters) >= p.max_sources:
                break
            if not any(math.hypot(d.x-b.x,d.y-b.y) < p.merge_distance_m for b in births):
                continue
            nearby = [c for c in self.clusters if np.linalg.norm(c.position.state[:2]-[d.x,d.y])
                      < p.merge_distance_m]
            if nearby:
                continue
            c = SourceCluster(f"belief_{self.next_label}",
                PositionFilter(np.array([d.x,d.y]), stamp_s=stamp),
                p.candidate_probability, d.strength, d.sigma, stamp)
            self.next_label += 1
            self.clusters.append(c)
        for c in self.clusters:
            if c.hits >= p.confirm_hits and c.probability >= p.confirm_probability:
                c.confirmed = True
        self._merge()
        self.clusters = [c for c in self.clusters if c.probability >= p.prune_probability]
        if snapshot is not None:
            self._fit_field(snapshot)
        self.stamp_s = stamp

    def _merge(self):
        p = self.params
        remove, active = set(), set()
        for i, a in enumerate(self.clusters):
            for j, b in enumerate(self.clusters[i+1:], i+1):
                key = (a.label,b.label)
                if (i in remove or j in remove or (a.confirmed and b.confirmed)
                    or np.linalg.norm(a.position.state[:2]-b.position.state[:2]) > p.merge_distance_m):
                    continue
                active.add(key)
                self.merge_support[key] = self.merge_support.get(key,0)+1
                if self.merge_support[key] < p.merge_hits:
                    continue
                # Covariance intersection is conservative for correlated tracks.
                ia, ib = np.linalg.inv(a.position.covariance), np.linalg.inv(b.position.covariance)
                cov = np.linalg.inv(.5*ia+.5*ib)
                a.position.state = cov @ (.5*ia@a.position.state+.5*ib@b.position.state)
                a.position.covariance = cov
                a.probability = max(a.probability,b.probability)
                a.amplitude = max(a.amplitude,b.amplitude)
                a.hits = max(a.hits,b.hits)
                a.confirmed |= b.confirmed
                remove.add(j)
        self.merge_support = {k:v for k,v in self.merge_support.items() if k in active}
        self.clusters = [c for i,c in enumerate(self.clusters) if i not in remove]

    def _fit_field(self, m):
        # Only the A-level observation has a Gaussian-field likelihood. Surface
        # observations use detection amplitude/extent updates, never a fake field.
        if m.get("measurement_type", "field_direct") != "field_direct":
            return
        yy,xx = np.indices(np.asarray(m['temperature_mean']).shape)
        x=m['origin_x']+(xx+.5)*m['resolution']; y=m['origin_y']+(yy+.5)*m['resolution']
        z=np.asarray(m['temperature_mean'])-m.get('ambient',22.)
        valid=(np.asarray(m['confidence'])>.1)&np.isfinite(z)
        if 'last_seen_age_s' in m:
            valid &= np.asarray(m['last_seen_age_s']) <= 2.0
        for c in self.clusters:
            r2=(x-c.position.state[0])**2+(y-c.position.state[1])**2
            mask=valid&(r2<self.params.field_fit_radius_m**2)
            if mask.sum()<4:
                continue
            target=z[mask].copy()
            for other in self.clusters:
                if other is not c:
                    target -= other.probability*other.amplitude*np.exp(-(
                        (x[mask]-other.position.state[0])**2+(y[mask]-other.position.state[1])**2)
                        /(2*other.sigma**2))
            best=None
            for sigma in np.linspace(self.params.sigma_min_m,self.params.sigma_max_m,12):
                basis=np.exp(-r2[mask]/(2*sigma*sigma))
                precision=1/max(c.amplitude_variance,.01)+float(basis@basis)
                amp=max(0.,(c.amplitude/max(c.amplitude_variance,.01)+float(basis@target))/precision)
                loss=float(np.mean((target-amp*basis)**2))
                if best is None or loss<best[0]:
                    best=(loss,amp,sigma,1/precision)
            _,c.amplitude,c.sigma,c.amplitude_variance=best

    def cardinality(self):
        pmf=np.array([1.])
        for c in self.clusters:
            pmf=np.convolve(pmf,[1-c.probability,c.probability])
        return pmf

    def information_gain(self, points, radius=3.0):
        points=np.asarray(points,dtype=float).reshape(-1,2)
        gain=np.zeros(len(points))
        p=self.params
        for c in self.clusters:
            footprint=np.exp(-np.sum((points-c.position.state[:2])**2,axis=1)/(2*radius**2))
            existence=detection_information(c.probability,p.p_detection,p.false_alarm_probability)
            cov=c.position.covariance[:2,:2]
            localization=.5*np.linalg.slogdet(np.eye(2)+cov/p.measurement_variance)[1]
            gain += footprint*(existence+c.probability*p.p_detection*localization)
        return gain
