"""Stationary-source Gaussian position filtering without ROS dependencies."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PositionFilter:
    state: np.ndarray
    covariance: np.ndarray = field(default_factory=lambda: np.eye(2)*0.4)
    stamp_s: float = 0.0

    def __post_init__(self):
        self.state = np.asarray(self.state, dtype=float).reshape(2).copy()
        self.covariance = np.asarray(self.covariance, dtype=float).reshape(2, 2).copy()

    def predict(self, stamp_s, position_noise_std=0.05):
        dt = float(stamp_s) - self.stamp_s
        if dt < -1e-9:
            raise ValueError("observation time cannot move backwards")
        # A stationary source stays at its estimated position while uncertainty
        # allows for mapping/registration drift between observations.
        self.covariance += np.eye(2)*position_noise_std**2*max(0., dt)
        self.stamp_s = float(stamp_s)

    def innovation(self, xy, variance):
        delta = np.asarray(xy, dtype=float) - self.state
        s = self.covariance + np.eye(2)*variance
        return delta, s, float(delta @ np.linalg.solve(s, delta))

    def correct(self, xy, variance):
        delta, s, _ = self.innovation(xy, variance)
        k = np.linalg.solve(s, self.covariance).T
        self.state += k @ delta
        a = np.eye(2) - k
        # Joseph form retains PSD under repeated low-noise observations.
        self.covariance = a @ self.covariance @ a.T + variance * (k @ k.T)
        self.covariance = (self.covariance + self.covariance.T) / 2


def associate(filters, detections, measurement_variance=0.15, gate_chi2=9.21,
              max_distance=3.0, strength=None, allowed=None):
    """One-to-one globally ordered gated association; stable tie breaking.

    Position uncertainty and appearance distinguish nearby hotspots.
    Unassigned detections remain eligible for confirmation.
    """
    pairs = []
    for i, filt in enumerate(filters):
        for j, det in enumerate(detections):
            if allowed is not None and not allowed(i, j):
                continue
            delta, covariance, d2 = filt.innovation((det.x, det.y), measurement_variance)
            if d2 <= gate_chi2 and np.linalg.norm(delta) <= max_distance:
                appearance = 0.0 if strength is None else abs(np.log(
                    max(det.strength, 0.1)/max(strength[i], 0.1)))
                # Gaussian likelihood includes its normalizer. Mahalanobis
                # distance alone rewards a lost track's unbounded uncertainty
                # and lets it steal observations from a precise current track.
                log_volume = np.linalg.slogdet(covariance)[1]
                pairs.append((d2 + log_volume + appearance, i, j))
    used_i, used_j, assignments = set(), set(), []
    for _, i, j in sorted(pairs):
        if i not in used_i and j not in used_j:
            assignments.append((i, j))
            used_i.add(i)
            used_j.add(j)
    return assignments, set(range(len(detections))) - used_j
