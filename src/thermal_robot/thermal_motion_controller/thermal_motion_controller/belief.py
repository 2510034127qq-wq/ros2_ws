"""Bounded labelled multi-Bernoulli source clusters (Rao-Blackwell branch).

Position/velocity are conditional Gaussians, existence is Bernoulli, and the
cardinality PMF is their exact convolution under the cluster independence
approximation. Association is a gated maximum-score approximation, not an
exact multi-object posterior. No ground truth or ROS types enter this module.
"""
from dataclasses import dataclass, field
import copy
import math
import time
import numpy as np

from .motion_filter import MotionFilter, associate


@dataclass
class BeliefParams:
    max_sources: int = 16
    acceleration_std: float = 0.35
    measurement_variance: float = 0.2
    gate_chi2: float = 9.21
    gate_m: float = 3.0
    survival_time_s: float = 180.0
    p_detection: float = 0.85
    false_alarm_probability: float = 0.03
    birth_probability: float = 0.15
    death_probability: float = 0.025
    confirm_probability: float = 0.90
    birth_hits: int = 3
    merge_distance_m: float = 0.45
    merge_velocity_m_s: float = 0.2
    merge_hits: int = 3
    budget_ms: float = 100.0
    field_fit_radius_m: float = 3.0
    sigma_min_m: float = 0.25
    sigma_max_m: float = 3.0


@dataclass
class SourceCluster:
    label: str
    motion: MotionFilter
    probability: float
    amplitude: float
    sigma: float
    last_seen_s: float
    hits: int = 1
    confirmed: bool = False
    amplitude_variance: float = 16.0
    sigma_variance: float = 0.5
    parent: str = ""


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
        self.events = []
        self.merge_support = {}
        self.last_ms = 0.0
        self.health = "not_started"

    def update(self, detections, stamp_s, visibility=None, birth_detections=None,
               field_snapshot=None):
        """Transactional update: late/invalid/over-budget steps never publish.

        visibility(x,y) is current sensor detection opportunity in [0,1], not
        historical coverage. An occluded source loses only survival probability.
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
        self.events = []
        for c in self.clusters:
            c.motion.predict(stamp, p.acceleration_std)
            c.probability *= math.exp(-dt/max(p.survival_time_s, 1e-6))
        matches, unmatched = associate(
            [c.motion for c in self.clusters], detections,
            p.measurement_variance, p.gate_chi2, p.gate_m,
            [c.amplitude for c in self.clusters])
        matched = set()
        for i, j in matches:
            c, d = self.clusters[i], detections[j]
            c.motion.correct((d.x, d.y), p.measurement_variance/max(d.confidence, .1))
            prior = c.probability
            c.probability = prior*p.p_detection / max(
                prior*p.p_detection+(1-prior)*p.false_alarm_probability, 1e-12)
            c.hits += 1
            if stamp-c.last_seen_s > 3:
                self.events.append(dict(kind="reacquire", label=c.label, gap_s=stamp-c.last_seen_s))
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
                opportunity = 0.0 if visibility is None else float(visibility(*c.motion.state[:2]))
                pd = p.p_detection*np.clip(opportunity, 0, 1)
                c.probability = c.probability*(1-pd)/max(1-c.probability*pd,1e-12)
        births = detections if birth_detections is None else birth_detections
        for j in sorted(unmatched):
            d = detections[j]
            if len(self.clusters) >= p.max_sources:
                break
            if not any(math.hypot(d.x-b.x,d.y-b.y) < p.merge_distance_m for b in births):
                continue
            nearby = [c for c in self.clusters if np.linalg.norm(c.motion.state[:2]-[d.x,d.y])
                      < p.merge_distance_m]
            if nearby:
                continue
            parents = sorted(self.clusters, key=lambda c: np.linalg.norm(c.motion.state[:2]-[d.x,d.y]))
            parent = parents[0].label if parents and np.linalg.norm(
                parents[0].motion.state[:2]-[d.x,d.y]) < 2*parents[0].sigma else ""
            c = SourceCluster(f"belief_{self.next_label}",
                MotionFilter(np.array([d.x,d.y,0.,0.]), stamp_s=stamp),
                p.birth_probability, d.strength, d.sigma, stamp, parent=parent)
            self.next_label += 1
            self.clusters.append(c)
            self.events.append(dict(kind="split_birth" if parent else "birth", label=c.label, parent=parent))
        for c in self.clusters:
            if c.hits >= p.birth_hits and c.probability >= p.confirm_probability:
                c.confirmed = True
        self._merge()
        for c in self.clusters:
            if c.probability < p.death_probability:
                self.events.append(dict(kind="death",label=c.label))
        self.clusters = [c for c in self.clusters if c.probability >= p.death_probability]
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
                    or np.linalg.norm(a.motion.state[:2]-b.motion.state[:2]) > p.merge_distance_m
                    or np.linalg.norm(a.motion.state[2:]-b.motion.state[2:]) > p.merge_velocity_m_s):
                    continue
                active.add(key)
                self.merge_support[key] = self.merge_support.get(key,0)+1
                if self.merge_support[key] < p.merge_hits:
                    continue
                # Covariance intersection is conservative for correlated tracks.
                ia, ib = np.linalg.inv(a.motion.covariance), np.linalg.inv(b.motion.covariance)
                cov = np.linalg.inv(.5*ia+.5*ib)
                a.motion.state = cov @ (.5*ia@a.motion.state+.5*ib@b.motion.state)
                a.motion.covariance = cov
                a.probability = max(a.probability,b.probability)
                a.amplitude = max(a.amplitude,b.amplitude)
                a.hits = max(a.hits,b.hits)
                a.confirmed |= b.confirmed
                remove.add(j)
                self.events.append(dict(kind="merge",label=a.label,removed=b.label))
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
            r2=(x-c.motion.state[0])**2+(y-c.motion.state[1])**2
            mask=valid&(r2<self.params.field_fit_radius_m**2)
            if mask.sum()<4:
                continue
            target=z[mask].copy()
            for other in self.clusters:
                if other is not c:
                    target -= other.probability*other.amplitude*np.exp(-(
                        (x[mask]-other.motion.state[0])**2+(y[mask]-other.motion.state[1])**2)
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
            footprint=np.exp(-np.sum((points-c.motion.state[:2])**2,axis=1)/(2*radius**2))
            existence=detection_information(c.probability,p.p_detection,p.false_alarm_probability)
            cov=c.motion.covariance[:2,:2]
            localization=.5*np.linalg.slogdet(np.eye(2)+cov/p.measurement_variance)[1]
            gain += footprint*(existence+c.probability*p.p_detection*localization)
        return gain

    def clearance(self, unseen_poisson_mean):
        # Sources already confirmed are excluded from "undiscovered"; candidate
        # existence and unseen Poisson mass both remain in the calculation.
        return float(np.exp(-max(0.,unseen_poisson_mean))*np.prod(
            [1-c.probability for c in self.clusters if not c.confirmed]))
