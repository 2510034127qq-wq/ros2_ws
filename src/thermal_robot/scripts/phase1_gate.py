#!/usr/bin/env python3
"""Evaluate the phase-1 gate from the program spec against two matrix roots."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple


SCRIPTS_DIR = Path(__file__).resolve().parent
FUSE_RE = re.compile(r"\[FUSE\].*?\b([0-9]+(?:\.[0-9]+)?)ms\b")
CLEARANCE_RE = re.compile(r"\[CLEARANCE\].*?\beval_ms=([0-9]+(?:\.[0-9]+)?)")
RESIDUAL_PLAN_RE = re.compile(r"\[RESIDUAL_PLAN\].*?\beval_ms=([0-9]+(?:\.[0-9]+)?)")
ARRIVAL_RE = re.compile(r"\[COARSE_WP#\d+\]\s+arrival\b")
TIMEOUT_RE = re.compile(r"\[COARSE_WP#\d+\]\s+timeout\b")
REQUIRED_CANDIDATE_ARTIFACTS = (
    "metadata.json",
    "attribution.json",
    "source_summary.json",
    "clearance.csv",
    "launch.log",
    "run_result.json",
    "trajectory.csv",
    "thermal_stats.csv",
    "field_stats.csv",
    "thermal_map_stats.csv",
    "gradient_stats.csv",
    "thermal_sources_truth.csv",
    "cmd_vel.csv",
    "scan_stats.csv",
)
EXPECTED_WORLDS = ("open", "boxes", "walls", "mixed")
EXPECTED_SCENARIOS = ("static2", "static3", "static5", "dyn4", "dyn5", "birthdeath")
EXPECTED_SEEDS = (101, 102, 103, 104, 105)
EXPECTED_KEYS = frozenset(
    (f"{world}__{scenario}", seed)
    for world in EXPECTED_WORLDS
    for scenario in EXPECTED_SCENARIOS
    for seed in EXPECTED_SEEDS
)


def _load_matrix_stats():
    name = "matrix_stats"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / "matrix_stats.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_matrix_runner():
    name = "phase1_gate_matrix_runner"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(
        name, SCRIPTS_DIR / "run_multiscenario_matrix.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ms = _load_matrix_stats()
_runner = _load_matrix_runner()
CANONICAL_PHASE0_CASES = {
    case.name: case for case in _runner.phase0_cases()
}


def _safe_json_dict(path: Path) -> Tuple[dict, str]:
    try:
        with open(path) as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {}, str(exc)
    if not isinstance(value, dict):
        return {}, f"expected JSON object, got {type(value).__name__}"
    return value, ""


def _is_finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_nonnegative_int(value) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def _run_schema_issues(value: dict, expected_strategy: str) -> List[str]:
    """Validate fields consumed by pairing, statistics, and provenance checks."""
    issues: List[str] = []
    if not isinstance(value.get("name"), str) or not value.get("name"):
        issues.append("name must be a nonempty string")
    if not _is_nonnegative_int(value.get("seed")):
        issues.append("seed must be a nonnegative integer")
    if value.get("strategy") != expected_strategy:
        issues.append(f"strategy must be {expected_strategy!r}")
    if not isinstance(value.get("passed"), bool):
        issues.append("passed must be boolean")
    if not isinstance(value.get("missing_counts"), list):
        issues.append("missing_counts must be a list")
    duration_s = value.get("duration_s")
    if not _is_finite_number(duration_s) or float(duration_s) <= 0.0:
        issues.append("duration_s must be a positive finite number")
    for key in ("source_recall", "source_precision"):
        metric = value.get(key)
        if not _is_finite_number(metric):
            issues.append(f"{key} must be a finite number")
        elif not 0.0 <= float(metric) <= 1.0:
            issues.append(f"{key} must be in [0, 1]")
    for key in (
            "truth_count", "matched_count", "confirmed_count",
            "duplicate_confirmations"):
        if not _is_nonnegative_int(value.get(key)):
            issues.append(f"{key} must be a nonnegative integer")
    failure_counts = value.get("failure_counts")
    if not isinstance(failure_counts, dict):
        issues.append("failure_counts must be an object")
    else:
        if "not_reached" not in failure_counts:
            issues.append("failure_counts.not_reached is required")
        if any(not _is_nonnegative_int(count)
               for count in failure_counts.values()):
            issues.append("failure_counts values must be nonnegative integers")
    for key in ("world", "scenario"):
        if not isinstance(value.get(key), str) or not value.get(key):
            issues.append(f"{key} must be a nonempty path string")
    if expected_strategy == "residual":
        if not _is_nonnegative_int(value.get("collector_returncode")):
            issues.append("collector_returncode must be a nonnegative integer")
        counts = value.get("counts")
        if not isinstance(counts, dict):
            issues.append("counts must be an object")
        else:
            for key in (*_runner.REQUIRED_PIPELINE_COUNTS, "clearance"):
                if not _is_nonnegative_int(counts.get(key)):
                    issues.append(f"counts.{key} must be a nonnegative integer")
        fingerprint = value.get("run_fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            issues.append("run_fingerprint must be a 64-character string")
    return issues


def _artifact_schema_issues(filename: str, value: dict) -> List[str]:
    """Return fail-closed schema errors for gate-authoritative JSON files."""
    issues: List[str] = []
    if filename == "metadata.json":
        counts = value.get("counts")
        if not isinstance(counts, dict):
            return ["counts must be an object"]
        required_counts = (*_runner.REQUIRED_PIPELINE_COUNTS, "clearance")
        for key in required_counts:
            if not _is_nonnegative_int(counts.get(key)):
                issues.append(f"counts.{key} must be a nonnegative integer")
    elif filename == "source_summary.json":
        for key in ("source_recall", "source_precision"):
            if not _is_finite_number(value.get(key)):
                issues.append(f"{key} must be a finite number")
        for key in (
                "truth_count", "matched_count", "confirmed_count",
                "duplicate_confirmations"):
            if not _is_nonnegative_int(value.get(key)):
                issues.append(f"{key} must be a nonnegative integer")
        localization_errors = value.get("localization_errors_m")
        if not isinstance(localization_errors, list):
            issues.append("localization_errors_m must be a list")
        elif any(not isinstance(item, dict) for item in localization_errors):
            issues.append("localization_errors_m entries must be objects")
    elif filename == "attribution.json":
        failure_counts = value.get("failure_counts")
        if not isinstance(failure_counts, dict):
            issues.append("failure_counts must be an object")
        elif any(not _is_nonnegative_int(count) for count in failure_counts.values()):
            issues.append("failure_counts values must be nonnegative integers")
        for key in ("n_truth_sources", "n_matched"):
            if not _is_nonnegative_int(value.get(key)):
                issues.append(f"{key} must be a nonnegative integer")
        if not isinstance(value.get("per_source"), dict):
            issues.append("per_source must be an object")
    elif filename == "run_result.json":
        issues.extend(_run_schema_issues(value, "residual"))
    return issues


def _canonical_case_issues(matrix: dict) -> List[str]:
    issues: List[str] = []
    for (name, seed), run in sorted(matrix["runs"].items()):
        canonical = CANONICAL_PHASE0_CASES.get(name)
        if canonical is None:
            continue
        for field, expected_path in (
                ("world", canonical.world), ("scenario", canonical.scenario)):
            raw_path = run.get(field)
            if not isinstance(raw_path, str) or not raw_path:
                issues.append(f"{name}/seed{seed}: {field} path missing/invalid")
                continue
            try:
                actual = Path(raw_path).expanduser().resolve()
                expected = Path(expected_path).expanduser().resolve()
            except (OSError, RuntimeError, ValueError) as exc:
                issues.append(f"{name}/seed{seed}: invalid {field} path: {exc}")
                continue
            if actual != expected:
                issues.append(
                    f"{name}/seed{seed}: {field}={actual} expected={expected}")
    return issues


def _load_matrix(root: Path) -> dict:
    summary, error = _safe_json_dict(root / "matrix_summary.json")
    load_issues: List[str] = []
    if error:
        load_issues.append(f"matrix_summary.json invalid: {error}")
    raw_config = summary.get("config")
    if not isinstance(raw_config, dict):
        load_issues.append("matrix_summary.config must be an object")
        config = {}
    else:
        config = raw_config
    try:
        matrix_duration_s = float(config.get("duration_s", 0.0) or 0.0)
    except (TypeError, ValueError, OverflowError):
        matrix_duration_s = 0.0
    runs = {}
    duplicate_keys = []
    raw_runs = summary.get("runs", [])
    if not isinstance(raw_runs, list):
        load_issues.append("matrix_summary.runs must be a list")
        raw_runs = []
    for index, raw_run in enumerate(raw_runs):
        if not isinstance(raw_run, dict):
            load_issues.append(f"runs[{index}] must be an object")
            continue
        run = dict(raw_run)
        run.setdefault("duration_s", matrix_duration_s)
        name = run.get("name")
        seed = run.get("seed")
        if not isinstance(name, str) or not name:
            load_issues.append(f"runs[{index}].name must be a nonempty string")
            continue
        if not _is_nonnegative_int(seed):
            load_issues.append(f"runs[{index}].seed must be a nonnegative integer")
            continue
        key = (name, seed)
        if key in runs:
            duplicate_keys.append(key)
        runs[key] = run
    return {
        "summary": summary,
        "runs": runs,
        "raw_run_count": len(raw_runs),
        "duplicate_keys": duplicate_keys,
        "config": config,
        "load_issues": load_issues,
    }


def _matrix_protocol_issues(matrix: dict, expected_strategy: str) -> List[str]:
    summary = matrix["summary"]
    config = matrix["config"]
    issues: List[str] = list(matrix["load_issues"])
    if matrix["raw_run_count"] != len(EXPECTED_KEYS):
        issues.append(
            f"runs length={matrix['raw_run_count']} expected={len(EXPECTED_KEYS)}")
    try:
        declared_n_runs = int(summary.get("n_runs", -1))
    except (TypeError, ValueError):
        declared_n_runs = -1
    if declared_n_runs != len(EXPECTED_KEYS):
        issues.append(
            f"n_runs={summary.get('n_runs')!r} expected={len(EXPECTED_KEYS)}")
    if matrix["duplicate_keys"]:
        issues.append(f"duplicate_keys={len(matrix['duplicate_keys'])}")
    keys = set(matrix["runs"])
    if keys != EXPECTED_KEYS:
        issues.append(
            f"key_set missing={len(EXPECTED_KEYS - keys)} extra={len(keys - EXPECTED_KEYS)}")
    if config.get("preset") != "phase0":
        issues.append(f"preset={config.get('preset')!r} expected='phase0'")
    if config.get("strategy") != expected_strategy:
        issues.append(
            f"strategy={config.get('strategy')!r} expected={expected_strategy!r}")
    try:
        seeds = tuple(int(seed) for seed in config.get("seeds", []))
    except (TypeError, ValueError):
        seeds = ()
    if seeds != EXPECTED_SEEDS:
        issues.append(f"seeds={seeds!r} expected={EXPECTED_SEEDS!r}")
    for key, expected in (("duration_s", 120.0), ("warmup_s", 36.0),
                          ("jitter_std_m", 0.0)):
        try:
            actual = float(config.get(key))
        except (TypeError, ValueError):
            actual = float("nan")
        if not math.isfinite(actual) or abs(actual - expected) > 1e-9:
            issues.append(f"{key}={config.get(key)!r} expected={expected}")
    if config.get("health_only") is not False:
        issues.append(f"health_only={config.get('health_only')!r} expected=False")
    for (name, seed), run in sorted(matrix["runs"].items()):
        issues.extend(
            f"{name}/seed{seed}: {issue}"
            for issue in _run_schema_issues(run, expected_strategy)
        )
    issues.extend(_canonical_case_issues(matrix))
    return issues


def _run_dir(root: Path, run: dict) -> Path:
    return root / run["name"] / f"seed{int(run['seed'])}"


def _paired_stat(
    runs_a: Dict[Tuple[str, int], dict],
    runs_b: Dict[Tuple[str, int], dict],
    keys: Sequence[Tuple[str, int]],
    getter: Callable[[dict], float],
) -> dict:
    values_a = [float(getter(runs_a[key])) for key in keys]
    values_b = [float(getter(runs_b[key])) for key in keys]
    return {
        "n": len(keys),
        "mean_a": sum(values_a) / len(values_a) if values_a else 0.0,
        "mean_b": sum(values_b) / len(values_b) if values_b else 0.0,
        "p_value": (
            _ms.paired_permutation_test(values_b, values_a)
            if values_a else 1.0
        ),
    }


def _event_counts(root: Path, run: dict) -> Tuple[int, int, int, int]:
    def safe_int(value, default=0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return int(default)

    run_dir = _run_dir(root, run)
    source_summary_path = run_dir / "source_summary.json"
    attribution_path = run_dir / "attribution.json"
    if source_summary_path.exists():
        source_summary, _ = _safe_json_dict(source_summary_path)
        if source_summary:
            n_confirmed = safe_int(
                source_summary.get("confirmed_count"),
                safe_int(run.get("confirmed_count")),
            )
            localization_errors = source_summary.get("localization_errors_m")
            n_correct = (
                len(localization_errors)
                if isinstance(localization_errors, list)
                else safe_int(run.get("matched_count"))
            )
        else:
            n_confirmed = safe_int(run.get("confirmed_count"))
            n_correct = safe_int(run.get("matched_count"))
    else:
        n_confirmed = safe_int(run.get("confirmed_count"))
        n_correct = safe_int(run.get("matched_count"))
    if attribution_path.exists():
        attribution, _ = _safe_json_dict(attribution_path)
        n_truth = safe_int(
            attribution.get("n_truth_sources"),
            safe_int(run.get("truth_count")),
        )
    else:
        n_truth = safe_int(run.get("truth_count"))
    n_duplicates = safe_int(run.get("duplicate_confirmations"))
    return n_correct, n_confirmed, n_truth, n_duplicates


def _side_event_stats(root: Path, runs: Dict[Tuple[str, int], dict], keys) -> dict:
    correct: List[int] = []
    confirmed: List[int] = []
    truth: List[int] = []
    duplicates: List[int] = []
    for key in keys:
        c, n, t, d = _event_counts(root, runs[key])
        correct.append(c)
        confirmed.append(n)
        truth.append(t)
        duplicates.append(d)
    p_point, p_lo, p_hi = _ms.cluster_bootstrap_ratio_ci(
        correct, confirmed, n_boot=10000, seed=11)
    d_point, d_lo, d_hi = _ms.cluster_bootstrap_ratio_ci(
        duplicates, truth, n_boot=10000, seed=13)
    return {
        "precision": {"point": p_point, "lo": p_lo, "hi": p_hi},
        "duplicate_rate": {"point": d_point, "lo": d_lo, "hi": d_hi},
    }


def _percentile(values: Iterable[float], percentile: float):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * float(percentile) / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    fraction = pos - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def _runtime_and_behavior(root: Path, runs, keys, planner_required_keys=()) -> dict:
    fuse_ms: List[float] = []
    clearance_ms: List[float] = []
    planner_ms: List[float] = []
    missing_runtime: List[str] = []
    missing_planner: List[str] = []
    planner_required = set(planner_required_keys)
    per_run: List[dict] = []
    for key in keys:
        run = runs[key]
        run_path = _run_dir(root, run)
        log_path = run_path / "launch.log"
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
        run_fuse = [float(value) for value in FUSE_RE.findall(text)]
        run_clearance = [float(value) for value in CLEARANCE_RE.findall(text)]
        run_planner = [float(value) for value in RESIDUAL_PLAN_RE.findall(text)]
        fuse_ms.extend(run_fuse)
        clearance_ms.extend(run_clearance)
        planner_ms.extend(run_planner)
        if not run_fuse or not run_clearance:
            missing_runtime.append(f"{key[0]}/seed{key[1]}")
        if key in planner_required and not run_planner:
            missing_planner.append(f"{key[0]}/seed{key[1]}")
        arrivals = len(ARRIVAL_RE.findall(text))
        timeouts = len(TIMEOUT_RE.findall(text))
        duration_s = max(1e-6, float(run.get("duration_s", 0.0) or 0.0))
        arrival_rate = arrivals / (duration_s / 60.0)
        per_run.append({
            "case": key[0],
            "seed": key[1],
            "arrivals": arrivals,
            "timeouts": timeouts,
            "duration_s": duration_s,
            "arrival_rate_per_min": arrival_rate,
        })
    return {
        "fuse": {
            "n": len(fuse_ms),
            "p99_ms": _percentile(fuse_ms, 99.0),
            "max_ms": max(fuse_ms) if fuse_ms else None,
        },
        "clearance": {
            "n": len(clearance_ms),
            "p99_ms": _percentile(clearance_ms, 99.0),
            "max_ms": max(clearance_ms) if clearance_ms else None,
        },
        "planner": {
            "n": len(planner_ms),
            "p99_ms": _percentile(planner_ms, 99.0),
            "max_ms": max(planner_ms) if planner_ms else None,
        },
        "missing_runtime_runs": missing_runtime,
        "missing_planner_runs": missing_planner,
        "per_run_behavior": per_run,
        "max_arrival_rate_per_min": max(
            (row["arrival_rate_per_min"] for row in per_run), default=0.0),
        "total_arrivals": sum(row["arrivals"] for row in per_run),
        "total_timeouts": sum(row["timeouts"] for row in per_run),
    }


def _check(passed: bool, actual, threshold: str) -> dict:
    return {"passed": bool(passed), "actual": actual, "threshold": threshold}


def evaluate_roots(baseline_root: Path, candidate_root: Path) -> dict:
    baseline_root = Path(baseline_root)
    candidate_root = Path(candidate_root)
    matrix_a = _load_matrix(baseline_root)
    matrix_b = _load_matrix(candidate_root)
    runs_a = matrix_a["runs"]
    runs_b = matrix_b["runs"]
    keys_a = set(runs_a)
    keys_b = set(runs_b)
    keys = sorted(keys_a & keys_b)
    protocol_issues_a = _matrix_protocol_issues(matrix_a, "full")
    protocol_issues_b = _matrix_protocol_issues(matrix_b, "residual")
    run_schema_issues_a = {
        key: _run_schema_issues(run, "full") for key, run in runs_a.items()
    }
    run_schema_issues_b = {
        key: _run_schema_issues(run, "residual") for key, run in runs_b.items()
    }
    statistic_keys = [
        key for key in keys
        if not run_schema_issues_a[key] and not run_schema_issues_b[key]
    ]
    candidate_runtime_keys = [
        key for key in keys if not run_schema_issues_b[key]
    ]

    missing_artifacts: List[str] = []
    candidate_health_issues: List[str] = []
    candidate_config = matrix_b["config"]
    runtime_fingerprint = candidate_config.get("runtime_bundle_sha256")
    if candidate_config.get("run_fingerprint_version") != _runner.RUN_FINGERPRINT_VERSION:
        candidate_health_issues.append(
            "matrix config has missing/wrong run_fingerprint_version")
    if not isinstance(runtime_fingerprint, str) or len(runtime_fingerprint) != 64:
        candidate_health_issues.append(
            "matrix config has missing/invalid runtime_bundle_sha256")
    else:
        current_runtime = _runner.runtime_bundle_fingerprint()
        if current_runtime != runtime_fingerprint:
            candidate_health_issues.append(
                f"runtime bundle mismatch: matrix={runtime_fingerprint} current={current_runtime}")
    try:
        candidate_min_recall = float(candidate_config["min_recall"])
    except (KeyError, TypeError, ValueError):
        candidate_min_recall = float("nan")
        candidate_health_issues.append("matrix config has missing/invalid min_recall")

    json_artifacts = {
        "metadata.json", "source_summary.json", "attribution.json", "run_result.json"}
    for key in keys:
        run = runs_b[key]
        run_path = _run_dir(candidate_root, run)
        run_label = f"{key[0]}/seed{key[1]}"
        for filename in REQUIRED_CANDIDATE_ARTIFACTS:
            artifact_path = run_path / filename
            if not artifact_path.exists():
                missing_artifacts.append(f"{run_label}/{filename}")
                continue
            if filename in json_artifacts:
                artifact, error = _safe_json_dict(artifact_path)
                if error:
                    missing_artifacts.append(
                        f"{run_label}/{filename}(invalid: {error})")
                else:
                    for issue in _artifact_schema_issues(filename, artifact):
                        missing_artifacts.append(
                            f"{run_label}/{filename}(invalid: {issue})")
            elif filename.endswith(".csv"):
                rows = _runner._csv_data_row_count(artifact_path)
                if rows is None or rows <= 0:
                    missing_artifacts.append(
                        f"{run_label}/{filename}(no-valid-data)")
            elif filename == "launch.log" and artifact_path.stat().st_size == 0:
                missing_artifacts.append(
                    f"{run_label}/{filename}(empty)")

        if run.get("collector_returncode") != 0:
            candidate_health_issues.append(
                f"{run_label}: collector_returncode={run.get('collector_returncode')!r}")
        if run.get("missing_counts") != []:
            candidate_health_issues.append(
                f"{run_label}: missing_counts={run.get('missing_counts')!r}")
        if not math.isfinite(candidate_min_recall):
            continue
        try:
            case = _runner.MatrixCase(
                name=run["name"],
                world=Path(run["world"]),
                scenario=Path(run["scenario"]),
            )
            expected_fingerprint = _runner.build_run_fingerprint(
                case=case,
                seed=int(run["seed"]),
                strategy="residual",
                duration_s=float(candidate_config["duration_s"]),
                warmup_s=float(candidate_config["warmup_s"]),
                jitter_std_m=float(candidate_config["jitter_std_m"]),
                min_recall=candidate_min_recall,
                health_only=False,
                runtime_fingerprint=runtime_fingerprint,
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            candidate_health_issues.append(
                f"{run_label}: cannot reconstruct fingerprint: {exc}")
            continue
        if run.get("run_fingerprint") != expected_fingerprint:
            candidate_health_issues.append(
                f"{run_label}: summary run_fingerprint mismatch")
        try:
            checkpoint = _runner.load_resumable_result(
                run_path,
                case,
                seed=int(run["seed"]),
                strategy="residual",
                expected_fingerprint=expected_fingerprint,
            )
        except (KeyError, OSError, TypeError, ValueError, OverflowError) as exc:
            checkpoint = None
            candidate_health_issues.append(
                f"{run_label}: checkpoint validation error: {exc}")
        if checkpoint is None:
            candidate_health_issues.append(
                f"{run_label}: checkpoint/artifact validation failed")
        else:
            normalized_checkpoint = dict(checkpoint)
            normalized_checkpoint.setdefault(
                "duration_s", float(candidate_config["duration_s"]))
            if normalized_checkpoint != run:
                candidate_health_issues.append(
                    f"{run_label}: matrix_summary run differs from run_result.json")

    truth_mismatches = [
        f"{key[0]}/seed{key[1]}"
        for key in statistic_keys
        if runs_a[key]["truth_count"] != runs_b[key]["truth_count"]
    ]
    high_source_keys = [
        key for key in statistic_keys
        if runs_a[key]["truth_count"] >= 4
    ]
    high_source_recall = _paired_stat(
        runs_a, runs_b, high_source_keys, lambda run: run["source_recall"])
    high_source_not_reached = _paired_stat(
        runs_a,
        runs_b,
        high_source_keys,
        lambda run: (run.get("failure_counts") or {}).get("not_reached", 0),
    )

    sides = {
        "a": _side_event_stats(baseline_root, runs_a, high_source_keys)
        if high_source_keys else {},
        "b": _side_event_stats(candidate_root, runs_b, high_source_keys)
        if high_source_keys else {},
    }
    sides_all = {
        "a": _side_event_stats(baseline_root, runs_a, statistic_keys)
        if statistic_keys else {},
        "b": _side_event_stats(candidate_root, runs_b, statistic_keys)
        if statistic_keys else {},
    }
    runtime = _runtime_and_behavior(
        candidate_root,
        runs_b,
        candidate_runtime_keys,
        planner_required_keys=high_source_keys,
    )
    precision_b = sides.get("b", {}).get("precision", {})
    duplicate_a = sides.get("a", {}).get("duplicate_rate", {})
    duplicate_b = sides.get("b", {}).get("duplicate_rate", {})
    fuse_p99 = runtime["fuse"]["p99_ms"]
    clearance_p99 = runtime["clearance"]["p99_ms"]
    planner_p99 = runtime["planner"]["p99_ms"]

    def _within_budget(value) -> bool:
        return value is not None and math.isfinite(float(value)) and float(value) < 100.0

    checks = {
        "pairing": _check(
            keys_a == EXPECTED_KEYS
            and keys_b == EXPECTED_KEYS
            and not protocol_issues_a
            and not protocol_issues_b
            and not truth_mismatches,
            {
                "pairs": len(keys),
                "expected_pairs": len(EXPECTED_KEYS),
                "baseline_only": len(keys_a - keys_b),
                "candidate_only": len(keys_b - keys_a),
                "truth_mismatches": truth_mismatches,
                "baseline_protocol_issues": protocol_issues_a,
                "candidate_protocol_issues": protocol_issues_b,
            },
            "exact phase0 24 cases x 5 seeds (120 unique pairs), fixed protocol, identical truth counts",
        ),
        "artifacts": _check(
            not missing_artifacts,
            {"missing": missing_artifacts, "candidate_runs": len(keys)},
            "every candidate run has parseable metadata/attribution/source_summary/checkpoint and complete raw CSV/runtime logs",
        ),
        "candidate_health": _check(
            not candidate_health_issues,
            {"issues": candidate_health_issues},
            "collector=0, missing_counts=[], current runtime fingerprint, matching resumable checkpoint for every run",
        ),
        "high_source_recall": _check(
            high_source_recall["n"] > 0
            and high_source_recall["mean_b"] > high_source_recall["mean_a"]
            and high_source_recall["p_value"] < 0.05,
            high_source_recall,
            "truth_count >= 4: mean B > A and paired permutation p < 0.05",
        ),
        "precision": _check(
            precision_b.get("lo", 0.0) >= 0.90,
            precision_b,
            "candidate event-pooled precision 95% lower bound >= 0.90",
        ),
        "duplicate_rate": _check(
            duplicate_b.get("point", float("inf"))
            <= duplicate_a.get("point", -float("inf")) + 1e-12,
            {"baseline": duplicate_a, "candidate": duplicate_b},
            "candidate event-pooled duplicate rate <= baseline",
        ),
        "not_reached": _check(
            high_source_not_reached["n"] > 0
            and high_source_not_reached["mean_b"] < high_source_not_reached["mean_a"]
            and high_source_not_reached["p_value"] < 0.05,
            high_source_not_reached,
            "truth_count >= 4: mean B < A and paired permutation p < 0.05",
        ),
        "runtime_budget": _check(
            not runtime["missing_runtime_runs"]
            and not runtime["missing_planner_runs"]
            and _within_budget(fuse_p99)
            and _within_budget(clearance_p99)
            and _within_budget(planner_p99),
            {
                "fuse": runtime["fuse"],
                "clearance": runtime["clearance"],
                "planner": runtime["planner"],
                "missing_runs": runtime["missing_runtime_runs"],
                "missing_planner_runs": runtime["missing_planner_runs"],
            },
            "FUSE/clearance evidence in every run, planner evidence in 4/5-source runs, p99 < 100 ms",
        ),
        "waypoint_churn": _check(
            runtime["max_arrival_rate_per_min"] <= 12.0,
            {
                "max_arrival_rate_per_min": runtime["max_arrival_rate_per_min"],
                "total_arrivals": runtime["total_arrivals"],
                "total_timeouts": runtime["total_timeouts"],
            },
            "every candidate run <= 12 COARSE arrival events/min",
        ),
    }
    passed = all(check["passed"] for check in checks.values())
    return {
        "baseline_root": str(baseline_root),
        "candidate_root": str(candidate_root),
        "n_pairs": len(keys),
        "groups": {
            "high_source": {
                "n": len(high_source_keys),
                "recall": high_source_recall,
                "not_reached": high_source_not_reached,
            },
        },
        "sides": sides,
        "sides_all": sides_all,
        "runtime": runtime,
        "checks": checks,
        "passed": passed,
    }


def _fmt(value) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.5f}"
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def render_report(report: dict, label_a: str = "v31", label_b: str = "candidate") -> str:
    result = "PASS" if report["passed"] else "FAIL"
    lines = [
        "# Phase 1 gate report",
        "",
        f"- result: **{result}**",
        f"- baseline: `{label_a}` ({report['baseline_root']})",
        f"- candidate: `{label_b}` ({report['candidate_root']})",
        f"- paired runs: {report['n_pairs']}",
        "",
        "## Gate checks",
        "",
        "| check | result | threshold | actual |",
        "|---|---|---|---|",
    ]
    for name, check in report["checks"].items():
        status = "PASS" if check["passed"] else "FAIL"
        actual = _fmt(check["actual"]).replace("|", "\\|")
        threshold = str(check["threshold"]).replace("|", "\\|")
        lines.append(f"| {name} | {status} | {threshold} | {actual} |")

    high = report["groups"]["high_source"]
    lines += [
        "",
        "## Four/five-source paired statistics",
        "",
        "| metric | n | mean A | mean B | p |",
        "|---|---|---|---|---|",
        f"| recall | {high['recall']['n']} | {high['recall']['mean_a']:.4f} "
        f"| {high['recall']['mean_b']:.4f} | {high['recall']['p_value']:.5f} |",
        f"| not_reached/run | {high['not_reached']['n']} "
        f"| {high['not_reached']['mean_a']:.4f} "
        f"| {high['not_reached']['mean_b']:.4f} "
        f"| {high['not_reached']['p_value']:.5f} |",
        "",
        "## Event-pooled confirmation quality (four/five-source cohort)",
        "",
        "| side | precision [lo, hi] | duplicate rate [lo, hi] |",
        "|---|---|---|",
    ]
    for label, key in ((label_a, "a"), (label_b, "b")):
        side = report.get("sides", {}).get(key, {})
        precision = side.get("precision")
        duplicate = side.get("duplicate_rate")
        if not precision or not duplicate:
            lines.append(f"| {label} | n/a | n/a |")
            continue
        lines.append(
            f"| {label} | {precision['point']:.3f} "
            f"[{precision['lo']:.3f}, {precision['hi']:.3f}] | "
            f"{duplicate['point']:.3f} [{duplicate['lo']:.3f}, {duplicate['hi']:.3f}] |")

    worst = sorted(
        report["runtime"]["per_run_behavior"],
        key=lambda row: row["arrival_rate_per_min"],
        reverse=True,
    )[:10]
    lines += [
        "",
        "## Worst waypoint arrival rates",
        "",
        "| case | seed | arrivals | timeouts | arrivals/min |",
        "|---|---|---|---|---|",
    ]
    for row in worst:
        lines.append(
            f"| {row['case']} | {row['seed']} | {row['arrivals']} "
            f"| {row['timeouts']} | {row['arrival_rate_per_min']:.2f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_root")
    parser.add_argument("candidate_root")
    parser.add_argument("--label-a", default="v31")
    parser.add_argument("--label-b", default="phase1")
    parser.add_argument("--out", default="")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    report = evaluate_roots(Path(args.baseline_root), Path(args.candidate_root))
    markdown = render_report(report, args.label_a, args.label_b)
    candidate_root = Path(args.candidate_root)
    out = Path(args.out) if args.out else candidate_root / "phase1_gate_report.md"
    json_out = (Path(args.json_out) if args.json_out
                else candidate_root / "phase1_gate_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown)
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(markdown)
    print(f"[phase1-gate] markdown={out}")
    print(f"[phase1-gate] json={json_out}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
