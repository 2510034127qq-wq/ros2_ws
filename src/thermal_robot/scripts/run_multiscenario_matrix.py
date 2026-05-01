#!/usr/bin/env python3
"""Run multi-world, multi-thermal-scenario closed-loop simulation checks."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence


WORKSPACE = Path(__file__).resolve().parents[3]
BRINGUP = WORKSPACE / "src/thermal_robot/thermal_bringup"
WORLDS = BRINGUP / "worlds"
SCENARIOS = BRINGUP / "config/scenarios"
CONFIG_B = BRINGUP / "config/config_b_sources.yaml"
COLLECTOR = WORKSPACE / "src/thermal_robot/scripts/collect_sim_data.py"


@dataclass(frozen=True)
class MatrixCase:
    name: str
    world: Path
    scenario: Path


REPRESENTATIVE_CASES = [
    MatrixCase("open_config_b", WORLDS / "thermal_scene_nav.world", CONFIG_B),
    MatrixCase("obstacle_linear", WORLDS / "thermal_scene_obstacle_field.world", SCENARIOS / "dynamic_linear_sources.yaml"),
    MatrixCase("corridor_appear", WORLDS / "thermal_scene_corridor_rooms.world", SCENARIOS / "dynamic_appear_disappear_sources.yaml"),
    MatrixCase("mixed_waypoint", WORLDS / "thermal_scene_mixed_rooms.world", SCENARIOS / "dynamic_waypoint_random_sources.yaml"),
]

FULL_WORLDS = [
    WORLDS / "thermal_scene_nav.world",
    WORLDS / "thermal_scene_obstacle_field.world",
    WORLDS / "thermal_scene_corridor_rooms.world",
    WORLDS / "thermal_scene_mixed_rooms.world",
]
FULL_SCENARIOS = [
    CONFIG_B,
    SCENARIOS / "static_offset_sources.yaml",
    SCENARIOS / "dynamic_linear_sources.yaml",
    SCENARIOS / "dynamic_circular_sources.yaml",
    SCENARIOS / "dynamic_appear_disappear_sources.yaml",
    SCENARIOS / "dynamic_waypoint_random_sources.yaml",
]


def _sourced_command(command: str) -> List[str]:
    return [
        "bash",
        "-lc",
        "source /opt/ros/humble/setup.bash && "
        f"source {shlex.quote(str(WORKSPACE / 'install/setup.bash'))} && "
        f"{command}",
    ]


def _case_name_from_paths(world: Path, scenario: Path) -> str:
    world_name = world.stem.replace("thermal_scene_", "")
    scenario_name = scenario.stem.replace("_sources", "")
    return f"{world_name}__{scenario_name}"


def _full_cases() -> List[MatrixCase]:
    return [
        MatrixCase(_case_name_from_paths(world, scenario), world, scenario)
        for world in FULL_WORLDS
        for scenario in FULL_SCENARIOS
    ]


def _parse_case(raw: str) -> MatrixCase:
    parts = raw.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--case must be name:world_file:scenario_file")
    name, world, scenario = parts
    return MatrixCase(name=name, world=Path(world).expanduser(), scenario=Path(scenario).expanduser())


def _ensure_files(cases: Sequence[MatrixCase]) -> None:
    required = [COLLECTOR, WORKSPACE / "install/setup.bash"]
    for case in cases:
        required.extend([case.world, case.scenario])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n  " + "\n  ".join(missing))


def _stop_process_group(proc: subprocess.Popen, grace_s: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    pgid = os.getpgid(proc.pid)
    os.killpg(pgid, signal.SIGINT)
    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    os.killpg(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    os.killpg(pgid, signal.SIGKILL)
    proc.wait(timeout=5.0)


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def run_case(
    case: MatrixCase,
    out_root: Path,
    duration_s: float,
    warmup_s: float,
    domain_id: int,
    min_recall: float,
) -> Dict:
    run_dir = out_root / case.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "ros_logs"
    log_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["ROS_LOG_DIR"] = str(log_dir)
    env["ROS_DOMAIN_ID"] = str(domain_id)
    env["PYTHONUNBUFFERED"] = "1"

    launch_cmd = " ".join([
        "ros2", "launch", "thermal_bringup", "sim_nav_slam_launch.py",
        "use_rviz:=false",
        "use_gzclient:=false",
        f"world_file:={shlex.quote(str(case.world))}",
        f"scenario_file:={shlex.quote(str(case.scenario))}",
    ])
    collector_cmd = " ".join([
        "python3",
        shlex.quote(str(COLLECTOR)),
        "--out-dir", shlex.quote(str(run_dir)),
        "--duration", str(duration_s),
    ])

    launch_log_path = run_dir / "launch.log"
    collector_log_path = run_dir / "collector.log"
    started_at = time.time()
    with open(launch_log_path, "w") as launch_log:
        launch_proc = subprocess.Popen(
            _sourced_command(launch_cmd),
            cwd=str(WORKSPACE),
            env=env,
            stdout=launch_log,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    collector_returncode = None
    try:
        time.sleep(warmup_s)
        with open(collector_log_path, "w") as collector_log:
            collector_run = subprocess.run(
                _sourced_command(collector_cmd),
                cwd=str(WORKSPACE),
                env=env,
                stdout=collector_log,
                stderr=subprocess.STDOUT,
                timeout=duration_s + 45.0,
                check=False,
            )
            collector_returncode = collector_run.returncode
    finally:
        _stop_process_group(launch_proc)
        time.sleep(2.0)

    metadata = _load_json(run_dir / "metadata.json")
    summary = _load_json(run_dir / "source_summary.json")
    counts = metadata.get("counts", {})
    required_counts = ["thermal_stats", "field_stats", "map_stats", "grad_stats", "truth_sources", "cmd_vel"]
    missing_counts = [key for key in required_counts if int(counts.get(key, 0) or 0) <= 0]
    recall = float(summary.get("source_recall", 0.0) or 0.0)
    duplicates = int(summary.get("duplicate_confirmations", 0) or 0)
    passed = (
        collector_returncode == 0
        and not missing_counts
        and bool(summary)
        and recall >= min_recall
        and duplicates == 0
    )
    return {
        "name": case.name,
        "world": str(case.world),
        "scenario": str(case.scenario),
        "run_dir": str(run_dir),
        "ros_domain_id": domain_id,
        "collector_returncode": collector_returncode,
        "elapsed_wall_s": round(time.time() - started_at, 3),
        "passed": passed,
        "missing_counts": missing_counts,
        "source_recall": summary.get("source_recall"),
        "source_precision": summary.get("source_precision"),
        "duplicate_confirmations": summary.get("duplicate_confirmations"),
        "path_length_m": summary.get("path_length_m"),
        "plans_observed": summary.get("nav2_goal_proxy", {}).get("plans_observed"),
        "counts": counts,
    }


def select_cases(args: argparse.Namespace) -> List[MatrixCase]:
    if args.case:
        return args.case
    if args.preset == "full":
        return _full_cases()
    return REPRESENTATIVE_CASES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["representative", "full"], default="representative")
    parser.add_argument("--case", type=_parse_case, action="append", help="name:world_file:scenario_file")
    parser.add_argument("--out-root", default="", help="default: /tmp/thermal_world_scenario_matrix_<timestamp>")
    parser.add_argument("--duration", type=float, default=90.0, help="collector duration per case")
    parser.add_argument("--warmup", type=float, default=36.0, help="launch warmup before collector starts")
    parser.add_argument("--domain-start", type=int, default=71)
    parser.add_argument("--min-recall", type=float, default=0.0)
    args = parser.parse_args()

    cases = select_cases(args)
    _ensure_files(cases)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root).expanduser() if args.out_root else Path(f"/tmp/thermal_world_scenario_matrix_{ts}")
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, case in enumerate(cases):
        print(f"[matrix] case {idx + 1}/{len(cases)}: {case.name}")
        result = run_case(
            case=case,
            out_root=out_root,
            duration_s=args.duration,
            warmup_s=args.warmup,
            domain_id=args.domain_start + idx,
            min_recall=args.min_recall,
        )
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"[matrix] {status} {case.name}: "
            f"recall={result['source_recall']} precision={result['source_precision']} "
            f"dup={result['duplicate_confirmations']} dir={result['run_dir']}"
        )

    summary = {
        "preset": args.preset,
        "duration_s": args.duration,
        "warmup_s": args.warmup,
        "n_cases": len(results),
        "n_passed": sum(1 for item in results if item["passed"]),
        "all_passed": all(item["passed"] for item in results),
        "results": results,
    }
    out_path = out_root / "matrix_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[matrix] wrote {out_path}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
