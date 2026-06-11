#!/usr/bin/env python3
"""Run multi-world, multi-thermal-scenario closed-loop simulation matrices."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


WORKSPACE = Path(__file__).resolve().parents[3]
SCRIPTS = WORKSPACE / "src/thermal_robot/scripts"
BRINGUP = WORKSPACE / "src/thermal_robot/thermal_bringup"
WORLDS = BRINGUP / "worlds"
OCCUPANCY_DIR = WORLDS / "occupancy"
SCENARIOS = BRINGUP / "config/scenarios"
CONFIG_B = BRINGUP / "config/config_b_sources.yaml"
COLLECTOR = SCRIPTS / "collect_sim_data.py"
ATTRIBUTION = SCRIPTS / "attribution.py"
HEALTH_CHECK = SCRIPTS / "nav2_health_check.py"


def _load_matrix_stats():
    spec = importlib.util.spec_from_file_location("matrix_stats", SCRIPTS / "matrix_stats.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class MatrixCase:
    name: str
    world: Path
    scenario: Path


PHASE0_WORLD_CLASSES = {
    "open": WORLDS / "thermal_scene_nav.world",
    "boxes": WORLDS / "thermal_scene_obstacle_field.world",
    "walls": WORLDS / "thermal_scene_corridor_rooms.world",
    "mixed": WORLDS / "thermal_scene_mixed_rooms.world",
}
PHASE0_SCENARIO_CLASSES = {
    "static2": SCENARIOS / "static_two_sources.yaml",
    "static3": CONFIG_B,
    "static5": SCENARIOS / "static_five_sources.yaml",
    "dyn4": SCENARIOS / "dynamic_four_sources.yaml",
    "dyn5": SCENARIOS / "dynamic_five_sources.yaml",
    "birthdeath": SCENARIOS / "dynamic_appear_disappear_sources.yaml",
}


def phase0_cases() -> List[MatrixCase]:
    return [
        MatrixCase(f"{wkey}__{skey}", world, scenario)
        for wkey, world in PHASE0_WORLD_CLASSES.items()
        for skey, scenario in PHASE0_SCENARIO_CLASSES.items()
    ]


REPRESENTATIVE_CASES = [
    MatrixCase("open_config_b", WORLDS / "thermal_scene_nav.world", CONFIG_B),
    MatrixCase("obstacle_linear", WORLDS / "thermal_scene_obstacle_field.world", SCENARIOS / "dynamic_linear_sources.yaml"),
    MatrixCase("corridor_appear", WORLDS / "thermal_scene_corridor_rooms.world", SCENARIOS / "dynamic_appear_disappear_sources.yaml"),
    MatrixCase("mixed_waypoint", WORLDS / "thermal_scene_mixed_rooms.world", SCENARIOS / "dynamic_waypoint_random_sources.yaml"),
]

EXTENDED_CASES = [
    *REPRESENTATIVE_CASES,
    MatrixCase("zigzag_circular", WORLDS / "thermal_scene_zigzag_corridors.world", SCENARIOS / "dynamic_circular_sources.yaml"),
    MatrixCase("islands_static_offset", WORLDS / "thermal_scene_sparse_islands.world", SCENARIOS / "static_offset_sources.yaml"),
]

VARIABLE_SOURCE_CASES = [
    MatrixCase("open_static_2src", WORLDS / "thermal_scene_nav.world", SCENARIOS / "static_two_sources.yaml"),
    MatrixCase("mixed_dynamic_4src", WORLDS / "thermal_scene_mixed_rooms.world", SCENARIOS / "dynamic_four_sources.yaml"),
    MatrixCase("zigzag_dynamic_5src", WORLDS / "thermal_scene_zigzag_corridors.world", SCENARIOS / "dynamic_five_sources.yaml"),
]

FULL_WORLDS = [
    WORLDS / "thermal_scene_nav.world",
    WORLDS / "thermal_scene_obstacle_field.world",
    WORLDS / "thermal_scene_corridor_rooms.world",
    WORLDS / "thermal_scene_mixed_rooms.world",
    WORLDS / "thermal_scene_zigzag_corridors.world",
    WORLDS / "thermal_scene_sparse_islands.world",
]
FULL_SCENARIOS = [
    CONFIG_B,
    SCENARIOS / "static_offset_sources.yaml",
    SCENARIOS / "dynamic_linear_sources.yaml",
    SCENARIOS / "dynamic_circular_sources.yaml",
    SCENARIOS / "dynamic_appear_disappear_sources.yaml",
    SCENARIOS / "dynamic_waypoint_random_sources.yaml",
    SCENARIOS / "static_two_sources.yaml",
    SCENARIOS / "dynamic_four_sources.yaml",
    SCENARIOS / "dynamic_five_sources.yaml",
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


def parse_seeds(raw: str) -> List[int]:
    return [int(v.strip()) for v in raw.split(",") if v.strip()]


def filter_cases(cases: Sequence[MatrixCase], worlds: str, case_names: str) -> List[MatrixCase]:
    out = list(cases)
    if worlds:
        keys = {w.strip() for w in worlds.split(",") if w.strip()}
        out = [c for c in out if c.name.split("__")[0] in keys]
    if case_names:
        keys = {n.strip() for n in case_names.split(",") if n.strip()}
        out = [c for c in out if c.name in keys]
    return out


def occupancy_path_for_world(world: Path) -> Path:
    return OCCUPANCY_DIR / f"{world.stem}.npz"


def scenario_fov(scenario_path: Path) -> Tuple[float, float]:
    """Read the sensor FOV from a scenario YAML; fall back to the 4.0x3.0 default."""
    try:
        import yaml
        with open(scenario_path) as f:
            raw = yaml.safe_load(f) or {}
    except Exception:
        return 4.0, 3.0
    sensor = raw.get("sensor", {}) or {}
    fov_x = float(sensor.get("fov_x_m", sensor.get("fov_x", 4.0)))
    fov_y = float(sensor.get("fov_y_m", sensor.get("fov_y", 3.0)))
    return fov_x, fov_y


def _sourced_command(command: str) -> List[str]:
    return [
        "bash",
        "-lc",
        "source /opt/ros/humble/setup.bash && "
        f"source {shlex.quote(str(WORKSPACE / 'install/setup.bash'))} && "
        f"{command}",
    ]


def _ensure_files(cases: Sequence[MatrixCase]) -> None:
    required = [COLLECTOR, ATTRIBUTION, HEALTH_CHECK, WORKSPACE / "install/setup.bash"]
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
    run_dir: Path,
    duration_s: float,
    warmup_s: float,
    domain_id: int,
    min_recall: float,
    seed: int,
    strategy: str,
    jitter_std_m: float,
    health_only: bool,
) -> Dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "ros_logs"
    log_dir.mkdir(exist_ok=True)
    case_home = run_dir / "home"
    gazebo_log_dir = run_dir / "gazebo_logs"
    case_home.mkdir(exist_ok=True)
    gazebo_log_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["ROS_LOG_DIR"] = str(log_dir)
    env["ROS_DOMAIN_ID"] = str(domain_id)
    gazebo_port = 11345 + int(domain_id)
    env["GAZEBO_MASTER_URI"] = f"http://127.0.0.1:{gazebo_port}"
    env["GAZEBO_LOG_PATH"] = str(gazebo_log_dir)
    model_paths = [
        "/home/hanchen/.gazebo/models",
        "/usr/share/gazebo-11/models",
        env.get("GAZEBO_MODEL_PATH", ""),
    ]
    env["GAZEBO_MODEL_PATH"] = ":".join(path for path in model_paths if path)
    env["GAZEBO_MODEL_DATABASE_URI"] = ""
    env["HOME"] = str(case_home)
    env["PYTHONUNBUFFERED"] = "1"

    launch_cmd = " ".join([
        "ros2", "launch", "thermal_bringup", "sim_nav_slam_launch.py",
        "use_rviz:=false",
        "use_gzclient:=false",
        f"world_file:={shlex.quote(str(case.world))}",
        f"scenario_file:={shlex.quote(str(case.scenario))}",
        f"run_seed:={int(seed)}",
        f"strategy:={strategy}",
        f"scenario_jitter_std_m:={jitter_std_m}",
    ])
    if health_only:
        payload_cmd = " ".join([
            "python3", shlex.quote(str(HEALTH_CHECK)),
            "--timeout", str(max(20.0, duration_s)), "--json",
        ])
    else:
        payload_cmd = " ".join([
            "python3",
            shlex.quote(str(COLLECTOR)),
            "--out-dir", shlex.quote(str(run_dir)),
            "--duration", str(duration_s),
        ])

    launch_log_path = run_dir / "launch.log"
    payload_log_path = run_dir / ("health.log" if health_only else "collector.log")
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

    payload_returncode = None
    payload_stdout = ""
    try:
        time.sleep(warmup_s)
        with open(payload_log_path, "w") as payload_log:
            payload_run = subprocess.run(
                _sourced_command(payload_cmd),
                cwd=str(WORKSPACE),
                env=env,
                stdout=subprocess.PIPE if health_only else payload_log,
                stderr=subprocess.STDOUT,
                timeout=duration_s + 60.0,
                check=False,
            )
            payload_returncode = payload_run.returncode
            if health_only:
                payload_stdout = (payload_run.stdout or b"").decode("utf-8", "replace")
                payload_log.write(payload_stdout)
    finally:
        _stop_process_group(launch_proc)
        time.sleep(2.0)

    result: Dict = {
        "name": case.name,
        "seed": int(seed),
        "strategy": strategy,
        "world": str(case.world),
        "scenario": str(case.scenario),
        "run_dir": str(run_dir),
        "ros_domain_id": domain_id,
        "elapsed_wall_s": round(time.time() - started_at, 3),
    }

    if health_only:
        health: Dict = {}
        try:
            start = payload_stdout.find("{")
            if start >= 0:
                health = json.loads(payload_stdout[start:])
        except json.JSONDecodeError:
            health = {}
        (run_dir / "nav2_health.json").write_text(json.dumps(health, indent=2))
        result["passed"] = bool(health.get("ok")) and payload_returncode == 0
        result["nav2_health"] = health
        return result

    occupancy = occupancy_path_for_world(case.world)
    fov_x, fov_y = scenario_fov(case.scenario)
    attr_cmd = [sys.executable, str(ATTRIBUTION), str(run_dir),
                "--fov-x", str(fov_x), "--fov-y", str(fov_y)]
    if occupancy.exists():
        attr_cmd += ["--occupancy", str(occupancy)]
    subprocess.run(attr_cmd, cwd=str(WORKSPACE), check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    metadata = _load_json(run_dir / "metadata.json")
    summary = _load_json(run_dir / "source_summary.json")
    attribution = _load_json(run_dir / "attribution.json")
    counts = metadata.get("counts", {})
    required_counts = ["thermal_stats", "field_stats", "map_stats", "grad_stats", "truth_sources", "cmd_vel"]
    missing_counts = [key for key in required_counts if int(counts.get(key, 0) or 0) <= 0]
    recall = float(summary.get("source_recall", 0.0) or 0.0)
    duplicates = int(summary.get("duplicate_confirmations", 0) or 0)
    passed = (
        payload_returncode == 0
        and not missing_counts
        and bool(summary)
        and recall >= min_recall
        and duplicates == 0
    )
    result.update({
        "collector_returncode": payload_returncode,
        "passed": passed,
        "missing_counts": missing_counts,
        "source_recall": summary.get("source_recall"),
        "source_precision": summary.get("source_precision"),
        "truth_count": summary.get("truth_count"),
        "matched_count": summary.get("matched_count"),
        "confirmed_count": summary.get("confirmed_count"),
        "duplicate_confirmations": summary.get("duplicate_confirmations"),
        "time_to_first_source": summary.get("time_to_first_source"),
        "path_length_m": summary.get("path_length_m"),
        "nav2_available": metadata.get("nav2_available"),
        "nav2_plan_count": metadata.get("nav2_plan_count"),
        "failure_counts": attribution.get("failure_counts"),
        "counts": counts,
    })
    return result


def select_cases(args: argparse.Namespace) -> List[MatrixCase]:
    if args.case:
        return args.case
    if args.preset == "phase0":
        return filter_cases(phase0_cases(), args.worlds, args.cases)
    if args.preset == "full":
        return _full_cases()
    if args.preset == "variable":
        return VARIABLE_SOURCE_CASES
    if args.preset == "extended":
        return EXTENDED_CASES
    return REPRESENTATIVE_CASES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["phase0", "representative", "extended", "variable", "full"], default="phase0")
    parser.add_argument("--case", type=_parse_case, action="append", help="name:world_file:scenario_file")
    parser.add_argument("--worlds", default="", help="phase0 world class filter, e.g. open,boxes")
    parser.add_argument("--cases", default="", help="phase0 case filter, e.g. open__static2")
    parser.add_argument("--seeds", default="101,102,103,104,105", help="comma-separated run seeds")
    parser.add_argument("--strategy", choices=["full", "frontier", "levy"], default="full")
    parser.add_argument("--jitter", type=float, default=0.0, help="scenario_jitter_std_m")
    parser.add_argument("--out-root", default="", help="default: /tmp/thermal_matrix_<timestamp>")
    parser.add_argument("--duration", type=float, default=120.0, help="collector duration per run")
    parser.add_argument("--warmup", type=float, default=36.0, help="launch warmup before payload starts")
    parser.add_argument("--domain-start", type=int, default=71)
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--health-only", action="store_true", help="run Nav2 health check instead of collecting data")
    args = parser.parse_args()

    matrix_stats = _load_matrix_stats()
    cases = select_cases(args)
    if not cases:
        raise ValueError("case selection is empty; check --worlds/--cases filters")
    _ensure_files(cases)
    seeds = parse_seeds(args.seeds)
    if args.health_only:
        seeds = seeds[:1]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root).expanduser() if args.out_root else Path(f"/tmp/thermal_matrix_{ts}")
    out_root.mkdir(parents=True, exist_ok=True)
    n_runs = len(cases) * len(seeds)
    if args.domain_start > 232:
        raise ValueError("ROS_DOMAIN_ID must stay <= 232; lower --domain-start")
    domain_span = max(1, 233 - args.domain_start)

    runs: List[Dict] = []
    run_idx = 0
    for case in cases:
        for seed in seeds:
            run_idx += 1
            domain_id = args.domain_start + ((run_idx - 1) % domain_span)
            print(f"[matrix] run {run_idx}/{n_runs}: {case.name} seed={seed} strategy={args.strategy}")
            result = run_case(
                case=case,
                run_dir=out_root / case.name / f"seed{seed}",
                duration_s=args.duration,
                warmup_s=args.warmup,
                domain_id=domain_id,
                min_recall=args.min_recall,
                seed=seed,
                strategy=args.strategy,
                jitter_std_m=args.jitter,
                health_only=args.health_only,
            )
            runs.append(result)
            status = "PASS" if result.get("passed") else "FAIL"
            if args.health_only:
                print(f"[matrix] {status} {case.name} seed={seed} (nav2 health)")
            else:
                print(
                    f"[matrix] {status} {case.name} seed={seed}: "
                    f"recall={result.get('source_recall')} precision={result.get('source_precision')} "
                    f"dup={result.get('duplicate_confirmations')}"
                )

    cases_agg: Dict[str, Dict] = {}
    for case in cases:
        case_runs = [r for r in runs if r["name"] == case.name]
        agg = matrix_stats.aggregate_case_runs(case_runs)
        fc_total = {k: 0 for k in matrix_stats.FAILURE_CLASSES}
        for run in case_runs:
            for key, value in (run.get("failure_counts") or {}).items():
                fc_total[key] = fc_total.get(key, 0) + int(value)
        agg["failure_counts"] = fc_total
        cases_agg[case.name] = agg

    summary = {
        "config": {
            "preset": args.preset,
            "strategy": args.strategy,
            "seeds": seeds,
            "duration_s": args.duration,
            "warmup_s": args.warmup,
            "jitter_std_m": args.jitter,
            "health_only": args.health_only,
        },
        "n_runs": len(runs),
        "n_passed": sum(1 for item in runs if item.get("passed")),
        "all_passed": all(item.get("passed") for item in runs),
        "cases": cases_agg,
        "runs": runs,
    }
    out_path = out_root / "matrix_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    report_path = out_root / "matrix_report.md"
    report_path.write_text(matrix_stats.render_markdown_report(summary))
    print(f"[matrix] wrote {out_path}")
    print(f"[matrix] wrote {report_path}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
