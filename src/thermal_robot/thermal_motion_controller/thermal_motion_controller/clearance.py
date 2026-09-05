"""Generative clearance probability v1 and calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VIEW_NEVER = 0
VIEW_BLOCKED_ONLY = 1
VIEW_CLEAR = 2


@dataclass
class ClearanceParams:
    source_rate_per_m2: float = 0.01
    p_detect_per_sector: float = 0.7
    max_effective_sectors: int = 4
    residual_block_thresh: float = 1.5
    epsilon: float = 0.05
    domain_radius_m: float = 12.0
    amplitude_min_c: float = 4.0
    amplitude_prior_mean_c: float = 15.0
    detection_noise_c: float = 1.0
    detection_distance_scale_m: float = 5.0
    evidence_memory_s: float = 60.0
    source_birth_rate_m2_s: float = 0.00001


def _popcount8(arr: np.ndarray) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.uint8).reshape(-1, 1)
    return np.unpackbits(flat, axis=1).sum(axis=1).reshape(np.asarray(arr).shape)


def cell_miss_prob(view_state, view_sectors, params: ClearanceParams,
                   residual=None) -> np.ndarray:
    vs = np.asarray(view_state)
    k = np.minimum(_popcount8(view_sectors), int(params.max_effective_sectors))
    p_miss = np.ones(vs.shape, dtype=np.float64)
    clear = vs == VIEW_CLEAR
    p_miss[clear] = (1.0 - float(params.p_detect_per_sector)) ** k[clear]
    if residual is not None:
        p_miss[np.asarray(residual) >= float(params.residual_block_thresh)] = 1.0
    return p_miss


def clearance_probability(view_state, view_sectors, cell_area_m2,
                          params: ClearanceParams, residual=None,
                          free_mask=None) -> float:
    p_miss = cell_miss_prob(view_state, view_sectors, params, residual=residual)
    if free_mask is not None:
        p_miss = np.where(np.asarray(free_mask), p_miss, 0.0)
    lam = float(params.source_rate_per_m2) * float(cell_area_m2) * float(p_miss.sum())
    return float(np.exp(-lam))


def reliability_curve(claimed, outcomes, n_bins: int = 10):
    claimed = np.asarray(claimed, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (claimed >= lo) & (claimed < hi)
        else:
            mask = (claimed >= lo) & (claimed <= hi)
        if mask.any():
            rows.append((float(lo), float(hi), float(claimed[mask].mean()),
                         float(outcomes[mask].mean()), int(mask.sum())))
    return rows


def expected_calibration_error(claimed, outcomes, n_bins: int = 10) -> float:
    rows = reliability_curve(claimed, outcomes, n_bins=n_bins)
    total = sum(r[4] for r in rows)
    if total == 0:
        return 0.0
    return float(sum(abs(r[2] - r[3]) * r[4] for r in rows) / total)


def clearance_with_history(view_state, view_sectors, age_s, cell_area_m2,
                           params, residual=None, free_mask=None,
                           detection_distance_m=None, candidate_probabilities=()):
    """Poisson thinning with exponential amplitude prior conditioned on A>=Amin.

    Detectability integrates a logistic sensor response over amplitude bins;
    optional distance models attenuation. Age decays negative evidence and
    admits new births; unseen space always retains full prior mass. This is a
    model probability, not an empirical calibration claim.
    """
    vs=np.asarray(view_state);age=np.maximum(np.asarray(age_s),0.)
    distances=np.ones(vs.shape) if detection_distance_m is None else np.asarray(detection_distance_m)
    amplitudes=params.amplitude_min_c + params.amplitude_prior_mean_c*(-np.log(
        1-(np.arange(16)+.5)/16))
    attenuation=1/(1+(distances/max(params.detection_distance_scale_m,.01))**2)
    sectors=np.minimum(_popcount8(view_sectors),params.max_effective_sectors)
    miss=np.zeros(vs.shape,dtype=float)
    for amplitude in amplitudes:
        x=(amplitude*attenuation-params.amplitude_min_c)/max(params.detection_noise_c,.01)
        pd=params.p_detect_per_sector/(1+np.exp(-np.clip(x,-60,60)))
        # A source shares one unknown amplitude across sectors. Integrate the
        # joint miss likelihood, not the product of marginal detection means.
        miss += (1-pd)**sectors/len(amplitudes)
    miss=np.where(vs==VIEW_CLEAR,miss,1.)
    retention=np.exp(-age/max(params.evidence_memory_s,.01))
    miss=1-(1-miss)*retention
    if residual is not None: miss=np.where(np.asarray(residual)>=params.residual_block_thresh,1.,miss)
    mass=params.source_rate_per_m2*miss+params.source_birth_rate_m2_s*np.minimum(age,params.evidence_memory_s)
    if free_mask is not None: mass=np.where(free_mask,mass,0.)
    lam=float(mass.sum()*cell_area_m2)
    probability=float(np.exp(-lam)*np.prod(1-np.clip(candidate_probabilities,0,1)))
    return probability,lam
