#!/usr/bin/env python3
"""Offline spatial truth matching for reduced runs; never imported by control nodes.

These are window-level detection diagnostics, not a calibrated accuracy claim.
Ambiguous nearest matches are excluded; switches require repeated evidence.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


def evaluate(source_rows, truth_rows, gate_m=1., max_age_s=1.5, min_hits=3):
    truth_rows = sorted(truth_rows, key=lambda row: row['t'])
    active = {s['id'] for row in truth_rows for s in row['sources'] if s['status']=='truth_active'}
    counts, identities = Counter(), defaultdict(Counter)
    errors, switches = [], []
    stable, pending, pending_hits = {}, {}, Counter()
    fresh_count = unmatched = inactive = ambiguous = 0
    duplicate_bins = defaultdict(lambda: defaultdict(set))
    index = 0
    for row in sorted(source_rows, key=lambda row: row['t']):
        if row['status'] != 'confirmed' or row.get('age', 0.) > max_age_s:
            continue
        fresh_count += 1
        if not truth_rows:
            unmatched += 1
            continue
        while index+1 < len(truth_rows) and truth_rows[index+1]['t'] <= row['t']:
            index += 1
        truth = truth_rows[index]
        if abs(truth['t']-row['t']) > 1.:
            unmatched += 1
            continue
        distances = sorted((math.hypot(row['x']-s['x'], row['y']-s['y']), s['id'], s['status'])
                           for s in truth['sources'])
        if not distances or distances[0][0] > gate_m:
            unmatched += 1
            continue
        distance, physical_id, status = distances[0]
        if len(distances)>1 and distances[1][0]-distance < .2:
            ambiguous += 1
            continue
        key = row['id']
        identities[key][physical_id] += 1
        if status != 'truth_active':
            inactive += 1
        else:
            counts[physical_id] += 1
            errors.append(distance)
        duplicate_bins[int((row['t']-truth_rows[0]['t'])/.75)][physical_id].add(key)
        if pending.get(key) != physical_id:
            pending[key], pending_hits[key] = physical_id, 0
        pending_hits[key] += 1
        if pending_hits[key] >= min_hits and stable.get(key) != physical_id:
            if key in stable:
                switches.append(dict(track_id=key, before=stable[key], after=physical_id, t=row['t']))
            stable[key] = physical_id
    found = sorted(key for key, count in counts.items() if count >= min_hits)
    errors.sort()
    return dict(
        definition='fresh confirmed tracks; nearest active truth within gate; >= min_hits over probe window',
        identity_definition='identity continuity matches physical bodies including temporarily inactive heaters',
        gate_m=gate_m, max_age_s=max_age_s, min_hits=min_hits,
        active_truth_ids=sorted(active), detected_truth_ids=found, missed_truth_ids=sorted(active-set(found)),
        window_recall=len(found)/len(active) if active else None,
        confirmed_track_to_truth={key:dict(value) for key,value in identities.items()},
        identity_switches=switches, identity_switch_count=len(switches),
        duplicate_truth_bins=sum(len(ids)>1 for frames in duplicate_bins.values() for ids in frames.values()),
        fresh_confirmed_rows=fresh_count, unmatched_rows=unmatched, inactive_match_rows=inactive,
        ambiguous_rows=ambiguous,
        active_location_p95_m=errors[min(len(errors)-1, math.ceil(.95*len(errors))-1)] if errors else None)


def evaluate_directory(directory):
    directory = Path(directory)
    return evaluate(json.loads((directory/'sources.json').read_text()),
                    json.loads((directory/'truth.json').read_text()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = json.dumps(evaluate_directory(args.directory), indent=2)
    if args.out:
        args.out.write_text(result+'\n')
    print(result)
