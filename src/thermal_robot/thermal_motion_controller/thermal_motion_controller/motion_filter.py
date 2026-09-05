"""Constant-velocity Gaussian filtering without ROS dependencies."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MotionFilter:
    state: np.ndarray
    covariance: np.ndarray = field(default_factory=lambda: np.diag([0.4, 0.4, 1.0, 1.0]))
    stamp_s: float = 0.0

    def predict(self, stamp_s, acceleration_std=0.4):
        dt = float(stamp_s) - self.stamp_s
        if dt < -1e-9:
            raise ValueError("motion time cannot move backwards")
        f = np.eye(4)
        f[0, 2] = f[1, 3] = dt
        g = np.array([[dt*dt/2, 0], [0, dt*dt/2], [dt, 0], [0, dt]])
        self.state = f @ self.state
        self.covariance = f @ self.covariance @ f.T + acceleration_std**2 * (g @ g.T)
        self.stamp_s = float(stamp_s)

    def innovation(self, xy, variance):
        delta = np.asarray(xy, dtype=float) - self.state[:2]
        s = self.covariance[:2, :2] + np.eye(2)*variance
        return delta, s, float(delta @ np.linalg.solve(s, delta))

    def correct(self, xy, variance):
        delta, s, _ = self.innovation(xy, variance)
        k = np.linalg.solve(s, self.covariance[:2, :]).T
        self.state += k @ delta
        a = np.eye(4)
        a[:, :2] -= k
        # Joseph form retains PSD under repeated low-noise observations.
        self.covariance = a @ self.covariance @ a.T + variance * (k @ k.T)
        self.covariance = (self.covariance + self.covariance.T) / 2


def associate(filters, detections, measurement_variance=0.15, gate_chi2=9.21,
              max_distance=3.0, strength=None):
    """One-to-one globally ordered gated association; stable tie breaking.

    Motion prediction provides identity through crossings; appearance cost can
    distinguish nearby sources. Unassigned detections remain eligible for birth.
    """
    pairs = []
    for i, filt in enumerate(filters):
        for j, det in enumerate(detections):
            delta, _, d2 = filt.innovation((det.x, det.y), measurement_variance)
            if d2 <= gate_chi2 and np.linalg.norm(delta) <= max_distance:
                appearance = 0.0 if strength is None else abs(np.log(
                    max(det.strength, 0.1)/max(strength[i], 0.1)))
                pairs.append((d2 + appearance, i, j))
    used_i, used_j, assignments = set(), set(), []
    for _, i, j in sorted(pairs):
        if i not in used_i and j not in used_j:
            assignments.append((i, j))
            used_i.add(i)
            used_j.add(j)
    return assignments, set(range(len(detections))) - used_j
