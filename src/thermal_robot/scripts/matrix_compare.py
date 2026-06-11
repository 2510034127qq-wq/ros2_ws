#!/usr/bin/env python3
"""Compare two matrix output roots with paired tests and event-pooled CIs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def _load_sibling(name: str):
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ms = _load_sibling("matrix_stats")

PAIRED_METRICS = ("source_recall", "time_to_first_source", "path_length_m")


def _load_runs(root: Path):
    with open(root / "matrix_summary.json") as f:
        summary = json.load(f)
    return {(r["name"], int(r["seed"])): r for r in summary["runs"]}


def _run_counts(root: Path, run):
    run_dir = root / run["name"] / f"seed{run['seed']}"
    n_correct = n_confirmed = n_truth = 0
    source_summary = run_dir / "source_summary.json"
    if source_summary.exists():
        with open(source_summary) as f:
            s = json.load(f)
        n_confirmed = int(s.get("confirmed_count", 0))
        n_correct = len(s.get("localization_errors_m", []))
    attribution = run_dir / "attribution.json"
    if attribution.exists():
        with open(attribution) as f:
            a = json.load(f)
        n_truth = int(a.get("n_truth_sources", 0))
    n_dup = int(run.get("duplicate_confirmations", 0) or 0)
    return n_correct, n_confirmed, n_truth, n_dup


def compare_roots(root_a: Path, root_b: Path) -> dict:
    root_a, root_b = Path(root_a), Path(root_b)
    runs_a = _load_runs(root_a)
    runs_b = _load_runs(root_b)
    keys = sorted(set(runs_a) & set(runs_b))
    if not keys:
        raise SystemExit("no paired (case, seed) runs between the two roots")

    metrics = {}
    for metric in PAIRED_METRICS:
        xs, ys = [], []
        for key in keys:
            va, vb = runs_a[key].get(metric), runs_b[key].get(metric)
            if va is None or vb is None:
                continue
            xs.append(float(vb))
            ys.append(float(va))
        if xs:
            metrics[metric] = {
                "n": len(xs),
                "mean_a": sum(ys) / len(ys),
                "mean_b": sum(xs) / len(xs),
                "p_value": _ms.paired_permutation_test(xs, ys),
            }

    sides = {}
    for label, root, runs in (("a", root_a, runs_a), ("b", root_b, runs_b)):
        corr, conf, truth, dup = [], [], [], []
        for key in keys:
            c, n, t, d = _run_counts(root, runs[key])
            corr.append(c)
            conf.append(n)
            truth.append(t)
            dup.append(d)
        p_point, p_lo, p_hi = _ms.cluster_bootstrap_ratio_ci(corr, conf, seed=11)
        d_point, d_lo, d_hi = _ms.cluster_bootstrap_ratio_ci(dup, truth, seed=13)
        sides[label] = {
            "precision": {"point": p_point, "lo": p_lo, "hi": p_hi},
            "duplicate_rate": {"point": d_point, "lo": d_lo, "hi": d_hi},
        }

    per_case = {}
    for case in sorted({k[0] for k in keys}):
        xs = [float(runs_b[k]["source_recall"]) for k in keys if k[0] == case]
        ys = [float(runs_a[k]["source_recall"]) for k in keys if k[0] == case]
        per_case[case] = {
            "n": len(xs),
            "recall_a": sum(ys) / len(ys),
            "recall_b": sum(xs) / len(xs),
            "p_value": _ms.paired_permutation_test(xs, ys),
        }

    return {"n_pairs": len(keys), "metrics": metrics,
            "sides": sides, "per_case": per_case}


def render_report(report: dict, label_a: str, label_b: str) -> str:
    lines = [f"# Matrix compare: {label_a} (A) vs {label_b} (B)", "",
             f"paired runs: {report['n_pairs']}", "",
             "## Pooled paired tests (sign-flip permutation)", "",
             "| metric | n | mean A | mean B | p |", "|---|---|---|---|---|"]
    for metric, row in report["metrics"].items():
        lines.append(f"| {metric} | {row['n']} | {row['mean_a']:.3f} "
                     f"| {row['mean_b']:.3f} | {row['p_value']:.5f} |")
    lines += ["", "## Event-pooled precision / duplicate rate "
              "(run-level cluster bootstrap 95% CI)", "",
              "| side | precision [lo, hi] | duplicate rate [lo, hi] |",
              "|---|---|---|"]
    for label, key in ((label_a, "a"), (label_b, "b")):
        side = report["sides"][key]
        p, d = side["precision"], side["duplicate_rate"]
        lines.append(f"| {label} | {p['point']:.3f} [{p['lo']:.3f}, {p['hi']:.3f}] "
                     f"| {d['point']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}] |")
    lines += ["", "## Per-case recall (paired permutation)", "",
              "| case | n | recall A | recall B | p |", "|---|---|---|---|---|"]
    for case, row in report["per_case"].items():
        lines.append(f"| {case} | {row['n']} | {row['recall_a']:.3f} "
                     f"| {row['recall_b']:.3f} | {row['p_value']:.5f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_root")
    parser.add_argument("candidate_root")
    parser.add_argument("--label-a", default="baseline")
    parser.add_argument("--label-b", default="candidate")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    report = compare_roots(Path(args.baseline_root), Path(args.candidate_root))
    md = render_report(report, args.label_a, args.label_b)
    out = Path(args.out) if args.out else Path(args.candidate_root) / "compare_report.md"
    out.write_text(md)
    print(md)
    print(f"[compare] written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
