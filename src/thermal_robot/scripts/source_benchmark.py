#!/usr/bin/env python3
"""Compute source-level benchmark metrics from a collector output directory."""

import argparse
import csv
import json
import math
from pathlib import Path


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def as_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def source_match_radius(truth: dict | None, estimate: dict | None) -> float:
    truth_sigma = as_float(truth or {}, 'sigma')
    estimate_sigma = as_float(estimate or {}, 'sigma')
    return max(2.0, 2.0 * truth_sigma, 2.0 * estimate_sigma)


def compute(run_dir: Path) -> dict:
    truth_rows = read_csv(run_dir / 'thermal_sources_truth.csv')
    estimate_rows = read_csv(run_dir / 'source_estimates.csv')
    traj_rows = read_csv(run_dir / 'trajectory.csv')

    truth_history = {}
    for row in truth_rows:
        if row.get('status') == 'truth_active':
            truth_history.setdefault(row['id'], []).append(row)
    confirmed = {}
    first_t = {}
    for row in estimate_rows:
        if row.get('status') != 'confirmed':
            continue
        if as_float(row, 'probability', 1.0) < 0.5:
            continue
        confirmed.setdefault(row['id'], []).append(row)
        first_t.setdefault(row['id'], as_float(row, 't'))

    matches = []
    used = set()
    for tid, truth_records in truth_history.items():
        best_id = None
        best_d = float('inf')
        best_truth = None
        best_est = None
        for eid, estimates in confirmed.items():
            if eid in used:
                continue
            for est in estimates:
                truth = min(truth_records, key=lambda r: abs(as_float(r, 't') - as_float(est, 't')))
                tx, ty = as_float(truth, 'x'), as_float(truth, 'y')
                d = math.hypot(as_float(est, 'x') - tx, as_float(est, 'y') - ty)
                if d < best_d:
                    best_id = eid
                    best_d = d
                    best_truth = truth
                    best_est = est
        match_radius = source_match_radius(best_truth, best_est)
        if best_id is not None and best_d <= match_radius:
            used.add(best_id)
            matches.append({
                'truth_id': tid,
                'estimate_id': best_id,
                'error_m': round(best_d, 3),
                'match_radius_m': round(match_radius, 3),
                'time_s': round(first_t.get(best_id, as_float(best_est, 't')), 3),
                'truth_x': round(as_float(best_truth, 'x'), 3) if best_truth else None,
                'truth_y': round(as_float(best_truth, 'y'), 3) if best_truth else None,
            })

    path_length = 0.0
    for a, b in zip(traj_rows, traj_rows[1:]):
        path_length += math.hypot(as_float(b, 'wx') - as_float(a, 'wx'),
                                  as_float(b, 'wy') - as_float(a, 'wy'))
    truth_count = len(truth_history)
    confirmed_count = len(confirmed)
    times = [m['time_s'] for m in matches]
    return {
        'source_recall': round(len(matches) / max(1, truth_count), 3),
        'source_precision': round(len(matches) / max(1, confirmed_count), 3),
        'truth_count': truth_count,
        'matched_count': len(matches),
        'confirmed_count': confirmed_count,
        'time_to_first_source': min(times) if times else None,
        'time_to_all_sources': max(times) if len(matches) == truth_count and times else None,
        'localization_errors_m': matches,
        'duplicate_confirmations': max(0, confirmed_count - len(matches)),
        'path_length_m': round(path_length, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    parser.add_argument('--out', default='source_summary.json')
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser()
    summary = compute(run_dir)
    out = run_dir / args.out
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
