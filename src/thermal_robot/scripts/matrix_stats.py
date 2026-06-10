#!/usr/bin/env python3
"""Pure aggregation, rank statistics, and report rendering for the scenario matrix."""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional, Sequence

METRIC_KEYS = (
    'source_recall',
    'source_precision',
    'duplicate_confirmations',
    'time_to_first_source',
    'path_length_m',
)

FAILURE_CLASSES = ('not_reached', 'occluded', 'timing_missed', 'not_confirmed')

EXACT_TEST_MAX_COMBINATIONS = 20000


def _rank(values: Sequence[float]) -> List[float]:
    """Average ranks (1-based) with tie handling."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def aggregate_case_runs(runs: Sequence[Dict]) -> Dict:
    out = {
        'n_runs': len(runs),
        'n_passed': sum(1 for r in runs if r.get('passed')),
        'metrics': {},
    }
    for key in METRIC_KEYS:
        values = [float(r[key]) for r in runs
                  if r.get(key) is not None and math.isfinite(float(r[key]))]
        if not values:
            out['metrics'][key] = None
            continue
        mean = sum(values) / len(values)
        if len(values) > 1:
            var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        else:
            var = 0.0
        out['metrics'][key] = {
            'mean': round(mean, 4),
            'std': round(math.sqrt(var), 4),
            'min': round(min(values), 4),
            'max': round(max(values), 4),
            'n': len(values),
            'values': [round(v, 4) for v in values],
        }
    return out


def mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> Dict:
    """Two-sided Mann-Whitney U.

    Uses exact rank permutations when feasible, otherwise a normal
    approximation with tie correction and continuity correction.
    """
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return {'u': None, 'p_value': None, 'method': 'undefined', 'n1': n1, 'n2': n2}
    pooled = list(a) + list(b)
    ranks = _rank(pooled)
    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u = min(u1, n1 * n2 - u1)

    if math.comb(n1 + n2, n1) <= EXACT_TEST_MAX_COMBINATIONS:
        count = 0
        total = 0
        for combo in combinations(range(n1 + n2), n1):
            rr = sum(ranks[i] for i in combo)
            uu1 = rr - n1 * (n1 + 1) / 2.0
            uu = min(uu1, n1 * n2 - uu1)
            total += 1
            if uu <= u + 1e-9:
                count += 1
        p = min(1.0, count / total)
        method = 'exact'
    else:
        mu = n1 * n2 / 2.0
        n = n1 + n2
        tie_term = sum(t ** 3 - t for t in Counter(pooled).values())
        sigma_sq = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))
        if sigma_sq <= 0.0:
            p = 1.0
        else:
            z = (u - mu + 0.5) / math.sqrt(sigma_sq)
            p = min(1.0, math.erfc(abs(z) / math.sqrt(2.0)))
        method = 'normal_approx'
    return {'u': round(u, 4), 'p_value': round(p, 5), 'method': method, 'n1': n1, 'n2': n2}


def _fmt_metric(metric: Optional[Dict]) -> str:
    if not metric:
        return '-'
    return f"{metric['mean']:.3f}+/-{metric['std']:.3f} (n={metric['n']})"


def render_markdown_report(summary: Dict) -> str:
    cfg = summary.get('config', {})
    lines = [
        '# Scenario Matrix Evaluation Report',
        '',
        f"- preset: `{cfg.get('preset')}`  strategy: `{cfg.get('strategy')}`",
        f"- seeds: {cfg.get('seeds')}",
        f"- duration: {cfg.get('duration_s')}s  warmup: {cfg.get('warmup_s')}s",
        '',
        '## Per-Case Multi-Seed Statistics',
        '',
        '| case | runs | passed | recall | precision | duplicates | t_first(s) |',
        '|---|---|---|---|---|---|---|',
    ]
    for name in sorted(summary.get('cases', {})):
        case = summary['cases'][name]
        m = case.get('metrics', {})
        lines.append(
            f"| {name} | {case.get('n_runs')} | {case.get('n_passed')} "
            f"| {_fmt_metric(m.get('source_recall'))} "
            f"| {_fmt_metric(m.get('source_precision'))} "
            f"| {_fmt_metric(m.get('duplicate_confirmations'))} "
            f"| {_fmt_metric(m.get('time_to_first_source'))} |"
        )
    lines += ['', '## Missed-Source Failure Attribution', '',
              '| case | ' + ' | '.join(FAILURE_CLASSES) + ' |',
              '|---|' + '---|' * len(FAILURE_CLASSES)]
    for name in sorted(summary.get('cases', {})):
        counts = summary['cases'][name].get('failure_counts') or {}
        row = ' | '.join(str(counts.get(k, 0)) for k in FAILURE_CLASSES)
        lines.append(f'| {name} | {row} |')
    lines.append('')
    return '\n'.join(lines)
