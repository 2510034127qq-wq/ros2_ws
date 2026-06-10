#!/usr/bin/env python3
"""Per-truth-source failure attribution for a collector run directory."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

SCRIPTS_DIR = Path(__file__).resolve().parent

_wo_spec = importlib.util.spec_from_file_location(
    'world_occupancy', SCRIPTS_DIR / 'world_occupancy.py')
world_occupancy = importlib.util.module_from_spec(_wo_spec)
sys.modules['world_occupancy'] = world_occupancy
_wo_spec.loader.exec_module(world_occupancy)

FAILURE_CLASSES = ('not_reached', 'occluded', 'timing_missed', 'not_confirmed')
DEFAULT_MIN_VISIBLE_S = 1.0
TIME_MATCH_TOLERANCE_S = 0.6
WINDOW_GAP_S = 0.5


def _read_csv(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _f(row: Dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _nearest_record(records: Sequence[Dict], t: float) -> Optional[Dict]:
    if not records:
        return None
    lo, hi = 0, len(records) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if _f(records[mid], 't') < t:
            lo = mid + 1
        else:
            hi = mid
    best = records[lo]
    if lo > 0 and abs(_f(records[lo - 1], 't') - t) < abs(_f(best, 't') - t):
        best = records[lo - 1]
    if abs(_f(best, 't') - t) > TIME_MATCH_TOLERANCE_S:
        return None
    return best


def _group_windows(samples: List[Dict]) -> List[Dict]:
    windows: List[List[Dict]] = []
    cur: List[Dict] = []
    for sample in samples:
        if cur and sample['t'] - cur[-1]['t'] > WINDOW_GAP_S:
            windows.append(cur)
            cur = []
        cur.append(sample)
    if cur:
        windows.append(cur)
    out = []
    for group in windows:
        out.append({
            't_start': round(group[0]['t'], 2),
            't_end': round(group[-1]['t'], 2),
            'duration_s': round(group[-1]['t'] - group[0]['t'], 2),
            'min_dist_m': round(min(g['d'] for g in group), 2),
            'max_strength': round(max(g['strength'] for g in group), 2),
        })
    return out


def _count_events(flags: List[Dict]) -> int:
    return len(_group_windows(flags))


def compute_attribution(
    traj_rows: Sequence[Dict],
    truth_rows: Sequence[Dict],
    matched_truth_ids: Set[str],
    estimate_rows: Sequence[Dict],
    fov_x: float = 4.0,
    fov_y: float = 3.0,
    occupancy=None,
    min_visible_s: float = DEFAULT_MIN_VISIBLE_S,
) -> Dict:
    truth_by_id: Dict[str, List[Dict]] = {}
    for row in truth_rows:
        truth_by_id.setdefault(str(row['id']), []).append(row)
    for records in truth_by_id.values():
        records.sort(key=lambda r: _f(r, 't'))

    traj = sorted(traj_rows, key=lambda r: _f(r, 't'))
    per_source: Dict[str, Dict] = {}
    failure_counts = {key: 0 for key in FAILURE_CLASSES}

    for source_id, records in truth_by_id.items():
        visible: List[Dict] = []
        occluded: List[Dict] = []
        fov_inactive: List[Dict] = []
        min_dist = float('inf')
        ever_in_fov = False

        for pose in traj:
            t = _f(pose, 't')
            rec = _nearest_record(records, t)
            if rec is None:
                continue
            active = rec.get('status') == 'truth_active'
            sx, sy = _f(rec, 'x'), _f(rec, 'y')
            wx, wy, yaw = _f(pose, 'wx'), _f(pose, 'wy'), _f(pose, 'yaw')
            dx, dy = sx - wx, sy - wy
            d = math.hypot(dx, dy)
            cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
            fx = cos_y * dx - sin_y * dy
            fy = sin_y * dx + cos_y * dy
            in_fov = abs(fx) <= fov_x / 2.0 and abs(fy) <= fov_y / 2.0
            if active:
                min_dist = min(min_dist, d)
            if not in_fov:
                continue
            ever_in_fov = True
            if not active:
                fov_inactive.append({'t': t, 'd': d, 'strength': 0.0})
                continue
            los = (occupancy is None
                   or world_occupancy.line_of_sight(occupancy, wx, wy, sx, sy))
            sample = {'t': t, 'd': d, 'strength': _f(rec, 'strength')}
            if los:
                visible.append(sample)
            else:
                occluded.append(sample)

        windows = _group_windows(visible)
        visible_total = sum(w['duration_s'] for w in windows)
        dt_truth = 0.1
        if len(records) >= 2:
            span = _f(records[-1], 't') - _f(records[0], 't')
            dt_truth = max(1e-3, span / max(1, len(records) - 1))
        active_total = sum(
            1 for r in records if r.get('status') == 'truth_active') * dt_truth

        first_detect_t = None
        statuses_seen: Set[str] = set()
        for est in estimate_rows:
            rec = _nearest_record(records, _f(est, 't'))
            if rec is None:
                continue
            d = math.hypot(_f(est, 'x') - _f(rec, 'x'), _f(est, 'y') - _f(rec, 'y'))
            if d <= 2.5:
                statuses_seen.add(str(est.get('status')))
                if first_detect_t is None or _f(est, 't') < first_detect_t:
                    first_detect_t = _f(est, 't')

        matched = source_id in matched_truth_ids
        if matched:
            failure = None
        elif visible_total >= min_visible_s:
            failure = 'not_confirmed'
        elif occluded:
            failure = 'occluded'
        elif fov_inactive:
            failure = 'timing_missed'
        else:
            failure = 'not_reached'
        if failure is not None:
            failure_counts[failure] += 1

        per_source[source_id] = {
            'matched': matched,
            'failure_class': failure,
            'ever_in_fov': ever_in_fov,
            'min_center_dist_m': round(min_dist, 3) if math.isfinite(min_dist) else None,
            'visible_total_s': round(visible_total, 2),
            'visible_windows': windows,
            'occluded_events': _count_events(occluded),
            'fov_inactive_events': _count_events(fov_inactive),
            'active_total_s': round(active_total, 2),
            'visible_over_active_ratio': round(
                visible_total / active_total, 3) if active_total > 0 else None,
            'tracker_first_detect_t': first_detect_t,
            'tracker_statuses_seen': sorted(statuses_seen),
        }

    return {
        'per_source': per_source,
        'failure_counts': failure_counts,
        'n_truth_sources': len(per_source),
        'n_matched': sum(1 for r in per_source.values() if r['matched']),
    }


def compute_for_run_dir(run_dir: Path, occupancy_path: Optional[Path],
                        fov_x: float, fov_y: float) -> Dict:
    summary = {}
    summary_path = run_dir / 'source_summary.json'
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    matched_ids = {
        str(m['truth_id']) for m in summary.get('localization_errors_m', [])
    }
    occupancy = None
    if occupancy_path and Path(occupancy_path).exists():
        occupancy = world_occupancy.load_grid(Path(occupancy_path))
    return compute_attribution(
        traj_rows=_read_csv(run_dir / 'trajectory.csv'),
        truth_rows=_read_csv(run_dir / 'thermal_sources_truth.csv'),
        matched_truth_ids=matched_ids,
        estimate_rows=_read_csv(run_dir / 'source_estimates.csv'),
        fov_x=fov_x, fov_y=fov_y, occupancy=occupancy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('--occupancy', default='', help='truth occupancy npz')
    parser.add_argument('--fov-x', type=float, default=4.0)
    parser.add_argument('--fov-y', type=float, default=3.0)
    parser.add_argument('--out', default='attribution.json')
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    occ = Path(args.occupancy).expanduser() if args.occupancy else None
    result = compute_for_run_dir(run_dir, occ, args.fov_x, args.fov_y)
    out = run_dir / args.out
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result['failure_counts'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
