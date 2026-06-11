"""Residual thermal field: measured map minus forward prediction from known sources."""

from __future__ import annotations

import numpy as np

from thermal_field_reconstructor.thermal_mapping import VIEW_CLEAR


def predict_field(width, height, resolution, origin_x, origin_y,
                  ambient_temp, sources):
    """Predict a grid from Gaussian sources: (x, y, amplitude, sigma)."""
    h, w = int(height), int(width)
    yy, xx = np.mgrid[0:h, 0:w]
    wx = origin_x + (xx.astype(np.float32) + 0.5) * resolution
    wy = origin_y + (yy.astype(np.float32) + 0.5) * resolution
    field = np.full((h, w), float(ambient_temp), dtype=np.float32)
    for sx, sy, amp, sigma in sources:
        if amp <= 0.0 or sigma <= 1e-3:
            continue
        d2 = (wx - float(sx)) ** 2 + (wy - float(sy)) ** 2
        field += float(amp) * np.exp(-0.5 * d2 / float(sigma) ** 2)
    return field


def residual_field(temperature_mean, predicted, view_state):
    residual = (np.asarray(temperature_mean, dtype=np.float32)
                - np.asarray(predicted, dtype=np.float32))
    residual = np.maximum(residual, 0.0)
    residual[np.asarray(view_state) != VIEW_CLEAR] = 0.0
    return residual
