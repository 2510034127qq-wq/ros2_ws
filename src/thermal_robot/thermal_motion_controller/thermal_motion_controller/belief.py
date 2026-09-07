"""Slow Gaussian attribute estimates on the canonical source registry.

Source IDs, confirmation and candidate removal belong to SourceTrackerCore.
The cardinality PMF describes registered hypotheses, not unseen source count.
"""
from dataclasses import dataclass
import copy
import math
import time
import numpy as np

from .position_filter import PositionFilter


@dataclass
class BeliefParams:
    position_noise_std: float = 0.05
    measurement_variance: float = 0.2
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


def source_information_gain(points, position, covariance, probability, radius,
                            p_detection, false_alarm_probability, measurement_variance):
    """Expected existence and localization gain at candidate observation points."""
    distance_sq=np.sum((np.asarray(points)-position)**2,axis=-1)
    footprint=np.exp(-distance_sq/(2*radius**2))
    existence=detection_information(probability,p_detection,false_alarm_probability)
    localization=max(0.,.5*np.linalg.slogdet(np.eye(2)+covariance/measurement_variance)[1])
    return footprint*(existence+probability*p_detection*localization)


class SourceBelief:
    def __init__(self, params=None):
        self.params = params or BeliefParams()
        self.clusters = []
        self.stamp_s = None
        self.last_ms = 0.0
        self.health = "not_started"

    def update(self, registry, stamp_s, field_snapshot=None):
        """Transactional attribute update; failed work leaves the prior unchanged."""
        start = time.perf_counter()
        if self.stamp_s is not None and stamp_s <= self.stamp_s:
            self.health = "stale_input"
            return False
        try:
            candidate = copy.deepcopy(self)
            candidate._step(registry, float(stamp_s), field_snapshot)
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

    def _step(self, registry, stamp, snapshot):
        if not math.isfinite(stamp):
            raise ValueError("nonfinite timestamp")
        previous = {c.label:c for c in self.clusters}
        clusters, labels = [], set()
        changed = False
        for tr in registry:
            if (not tr.track_id or tr.track_id in labels or tr.status not in ('candidate','confirmed')
                    or not np.isfinite([tr.x,tr.y,tr.strength,tr.sigma,tr.confidence,
                        tr.existence_probability,tr.last_seen_s]).all()):
                raise ValueError("invalid canonical source")
            labels.add(tr.track_id)
            covariance = np.array([[tr.covariance_xx,tr.covariance_xy],
                                   [tr.covariance_xy,tr.covariance_yy]])
            if not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance).min() <= 0:
                raise ValueError("invalid source covariance")
            c = previous.get(tr.track_id)
            if c is None:
                c = SourceCluster(tr.track_id,PositionFilter(np.array([tr.x,tr.y]),covariance,stamp),
                    tr.existence_probability,tr.strength,tr.sigma,tr.last_seen_s,hits=tr.observations)
                changed = True
            else:
                c.position.predict(stamp,self.params.position_noise_std)
                if tr.observations > c.hits:
                    c.position.correct((tr.x,tr.y),max(self.params.measurement_variance,
                                                      float(np.trace(covariance)/2)))
                    dt = max(0.,tr.last_seen_s-c.last_seen_s)
                    for name,value,variance in (("amplitude",tr.strength,1/max(tr.confidence,.1)),
                                               ("sigma",tr.sigma,.25)):
                        pv = getattr(c,name+"_variance")+.02*dt
                        k = pv/(pv+variance)
                        setattr(c,name,max(.01,getattr(c,name)+k*(value-getattr(c,name))))
                        setattr(c,name+"_variance",(1-k)*pv)
                    changed = True
            c.probability = tr.existence_probability
            c.confirmed = tr.status == 'confirmed'
            c.hits,c.last_seen_s = tr.observations,tr.last_seen_s
            clusters.append(c)
        self.clusters = clusters
        if changed and snapshot is not None:
            self._fit_field(snapshot)
        self.stamp_s = stamp

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
            age = np.asarray(m['last_seen_age_s'])
            valid &= (age >= 0) & (age <= 2.0)
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
