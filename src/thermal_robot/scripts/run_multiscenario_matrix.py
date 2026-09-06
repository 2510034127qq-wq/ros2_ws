#!/usr/bin/env python3
"""Run multi-world, multi-thermal-scenario closed-loop simulation matrices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


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

RUN_FINGERPRINT_VERSION = "phase1-matrix-v1"
REQUIRED_PIPELINE_COUNTS = (
    "trajectory",
    "thermal_stats",
    "field_stats",
    "map_stats",
    "grad_stats",
    "truth_sources",
    "cmd_vel",
    "scan_stats",
)
COUNT_CSV_ARTIFACTS = {
    "trajectory": "trajectory.csv",
    "thermal_stats": "thermal_stats.csv",
    "field_stats": "field_stats.csv",
    "map_stats": "thermal_map_stats.csv",
    "grad_stats": "gradient_stats.csv",
    "truth_sources": "thermal_sources_truth.csv",
    "cmd_vel": "cmd_vel.csv",
    "scan_stats": "scan_stats.csv",
}


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
}


def phase0_cases() -> List[MatrixCase]:
    return [
        MatrixCase(f"{wkey}__{skey}", world, scenario)
        for wkey, world in PHASE0_WORLD_CLASSES.items()
        for skey, scenario in PHASE0_SCENARIO_CLASSES.items()
    ]


REPRESENTATIVE_CASES = [
    MatrixCase("obstacle_static2", WORLDS / "thermal_scene_obstacle_field.world", SCENARIOS / "static_two_sources.yaml"),
    MatrixCase("corridor_static5", WORLDS / "thermal_scene_corridor_rooms.world", SCENARIOS / "static_five_sources.yaml"),
    MatrixCase("mixed_offset", WORLDS / "thermal_scene_mixed_rooms.world", SCENARIOS / "static_offset_sources.yaml"),
    MatrixCase("open_config_b", WORLDS / "thermal_scene_nav.world", CONFIG_B),
]

EXTENDED_CASES = [
    *REPRESENTATIVE_CASES,
    MatrixCase("zigzag_static2", WORLDS / "thermal_scene_zigzag_corridors.world", SCENARIOS / "static_two_sources.yaml"),
    MatrixCase("islands_static_offset", WORLDS / "thermal_scene_sparse_islands.world", SCENARIOS / "static_offset_sources.yaml"),
]

VARIABLE_SOURCE_CASES = [
    MatrixCase("corridor_static5", WORLDS / "thermal_scene_corridor_rooms.world", SCENARIOS / "static_five_sources.yaml"),
    MatrixCase("mixed_static3", WORLDS / "thermal_scene_mixed_rooms.world", CONFIG_B),
    MatrixCase("open_static_2src", WORLDS / "thermal_scene_nav.world", SCENARIOS / "static_two_sources.yaml"),
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
    SCENARIOS / "static_two_sources.yaml",
    SCENARIOS / "static_five_sources.yaml",
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
    mismatches = runtime_sync_mismatches(runtime_sync_pairs())
    if mismatches:
        shown = "\n  ".join(
            f"{source} != {installed}" for source, installed in mismatches[:8])
        more = (f"\n  ... and {len(mismatches) - 8} more"
                if len(mismatches) > 8 else "")
        raise RuntimeError(
            "Installed ROS runtime is missing or stale. Rebuild the workspace "
            "before collecting a matrix:\n  " + shown + more)


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


def _csv_data_row_count(path: Path) -> Optional[int]:
    try:
        with open(path, newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                return None
            return sum(1 for _ in reader)
    except (OSError, csv.Error, UnicodeDecodeError):
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_sync_pairs() -> List[Tuple[Path, Path]]:
    """Map active source files to the installed files launched by ROS 2."""
    pairs: List[Tuple[Path, Path]] = []
    for package in (
            "thermal_sensor_sim", "signal_preprocessor",
            "thermal_field_reconstructor", "thermal_gradient_processor",
            "thermal_motion_controller"):
        source_root = WORKSPACE / "src/thermal_robot" / package / package
        install_candidates = sorted(
            (WORKSPACE / "install" / package / "lib").glob(
                f"python*/site-packages/{package}"))
        install_root = (install_candidates[0] if install_candidates else
                        WORKSPACE / "install" / package / "lib" /
                        f"python{sys.version_info.major}.{sys.version_info.minor}" /
                        "site-packages" / package)
        for source in sorted(source_root.rglob("*.py")):
            pairs.append((source, install_root / source.relative_to(source_root)))

    data_roots = (
        (WORKSPACE / "src/thermal_robot/thermal_bringup/config",
         WORKSPACE / "install/thermal_bringup/share/thermal_bringup/config"),
        (WORKSPACE / "src/thermal_robot/thermal_bringup/launch",
         WORKSPACE / "install/thermal_bringup/share/thermal_bringup/launch"),
        (WORKSPACE / "src/thermal_robot/g1_description/urdf",
         WORKSPACE / "install/g1_description/share/g1_description/urdf"),
    )
    runtime_data_suffixes = {".py", ".yaml", ".xml", ".urdf", ".xacro"}
    for source_root, install_root in data_roots:
        for source in sorted(
                path for path in source_root.rglob("*")
                if path.is_file() and path.suffix in runtime_data_suffixes):
            pairs.append((source, install_root / source.relative_to(source_root)))
    return pairs


def runtime_sync_mismatches(
        pairs: Sequence[Tuple[Path, Path]]) -> List[Tuple[Path, Path]]:
    """Return source/install pairs whose content is missing or different."""
    mismatches = []
    for source, installed in pairs:
        if (not source.is_file() or not installed.is_file()
                or _sha256_file(source) != _sha256_file(installed)):
            mismatches.append((source, installed))
    return mismatches


def _runtime_fingerprint_paths() -> List[Path]:
    """Return files that define matrix collection and installed runtime behavior."""
    paths = set()
    source_roots = [
        WORKSPACE / "src/thermal_robot/thermal_sensor_sim",
        WORKSPACE / "src/thermal_robot/signal_preprocessor",
        WORKSPACE / "src/thermal_robot/thermal_field_reconstructor",
        WORKSPACE / "src/thermal_robot/thermal_gradient_processor",
        WORKSPACE / "src/thermal_robot/thermal_motion_controller",
        WORKSPACE / "src/thermal_robot/thermal_bringup/config",
        WORKSPACE / "src/thermal_robot/thermal_bringup/launch",
        WORKSPACE / "src/thermal_robot/thermal_bringup/worlds",
        WORKSPACE / "src/thermal_robot/g1_description/urdf",
    ]
    allowed_suffixes = {".py", ".yaml", ".world", ".urdf", ".xml", ".npz"}
    for root in source_roots:
        if root.exists():
            paths.update(
                path for path in root.rglob("*")
                if path.is_file() and path.suffix in allowed_suffixes
                and "__pycache__" not in path.parts
            )
    for script in (
            Path(__file__).resolve(), COLLECTOR, ATTRIBUTION,
            SCRIPTS / "matrix_stats.py", SCRIPTS / "world_occupancy.py"):
        if script.exists():
            paths.add(script)

    install_roots = [
        WORKSPACE / "install/thermal_sensor_sim",
        WORKSPACE / "install/signal_preprocessor",
        WORKSPACE / "install/thermal_field_reconstructor",
        WORKSPACE / "install/thermal_gradient_processor",
        WORKSPACE / "install/thermal_motion_controller",
        WORKSPACE / "install/thermal_bringup/share/thermal_bringup/config",
        WORKSPACE / "install/thermal_bringup/share/thermal_bringup/launch",
        WORKSPACE / "install/g1_description/share/g1_description/urdf",
    ]
    for root in install_roots:
        if root.exists():
            paths.update(
                path for path in root.rglob("*")
                if path.is_file() and path.suffix in allowed_suffixes
                and "__pycache__" not in path.parts
            )
    return sorted(paths, key=lambda path: str(path.resolve()))


def runtime_bundle_fingerprint() -> str:
    """Hash source, installed runtime, configs, worlds, and collection code."""
    digest = hashlib.sha256()
    digest.update((RUN_FINGERPRINT_VERSION + "\n").encode())
    for path in _runtime_fingerprint_paths():
        try:
            label = str(path.resolve().relative_to(WORKSPACE.resolve()))
        except ValueError:
            label = str(path.resolve())
        digest.update(label.encode())
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def require_unchanged_runtime(expected: str, actual: str) -> None:
    """Abort rather than mix runs produced by different runtime bundles."""
    if actual != expected:
        raise RuntimeError(
            "runtime bundle changed during matrix collection; stop editing/building, "
            "then rerun the same command with --resume")


def build_run_fingerprint(
    case: MatrixCase,
    seed: int,
    strategy: str,
    duration_s: float,
    warmup_s: float,
    jitter_std_m: float,
    min_recall: float,
    health_only: bool,
    runtime_fingerprint: Optional[str] = None,
) -> str:
    """Hash all invocation and runtime inputs that make a checkpoint reusable."""
    occupancy = occupancy_path_for_world(case.world)
    payload = {
        "version": RUN_FINGERPRINT_VERSION,
        "case": {
            "name": case.name,
            "world": str(case.world.resolve()),
            "world_sha256": _sha256_file(case.world),
            "scenario": str(case.scenario.resolve()),
            "scenario_sha256": _sha256_file(case.scenario),
            "occupancy": str(occupancy.resolve()) if occupancy.exists() else None,
            "occupancy_sha256": _sha256_file(occupancy) if occupancy.exists() else None,
        },
        "invocation": {
            "seed": int(seed),
            "strategy": str(strategy),
            "duration_s": float(duration_s),
            "warmup_s": float(warmup_s),
            "jitter_std_m": float(jitter_std_m),
            "min_recall": float(min_recall),
            "health_only": bool(health_only),
        },
        "runtime_bundle_sha256": (
            runtime_fingerprint
            if runtime_fingerprint is not None
            else runtime_bundle_fingerprint()),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_run_result(run_dir: Path, result: Dict) -> None:
    """Atomically checkpoint one completed run for safe matrix resume."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_result.json"
    tmp_path = run_dir / "run_result.json.tmp"
    tmp_path.write_text(json.dumps(result, indent=2) + "\n")
    tmp_path.replace(path)


def prepare_run_attempt(
    run_dir: Path,
    resume: bool,
    expected_fingerprint: str,
) -> Optional[Path]:
    """Protect existing evidence before starting a new run attempt.

    An incompatible completed checkpoint is never overwritten.  A partial
    directory from an interrupted attempt is moved intact under
    ``.incomplete/`` so stale CSV files cannot leak into the next attribution
    pass and no evidence is deleted.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists() or not any(run_dir.iterdir()):
        return None
    if not resume:
        raise FileExistsError(
            f"run directory already contains data: {run_dir}; use a new "
            "--out-root or rerun the identical command with --resume")

    checkpoint_path = run_dir / "run_result.json"
    if checkpoint_path.exists():
        try:
            checkpoint = _load_json(checkpoint_path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            checkpoint = {}
        found_fingerprint = checkpoint.get("run_fingerprint")
        if found_fingerprint != expected_fingerprint:
            raise RuntimeError(
                f"incompatible checkpoint in {run_dir}: expected "
                f"{expected_fingerprint}, found {found_fingerprint!r}; "
                "use a new --out-root so different experiments cannot mix")

    out_root = run_dir.parents[1]
    archive_parent = out_root / ".incomplete" / run_dir.parent.name
    archive_parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archived = archive_parent / f"{run_dir.name}-{timestamp}"
    run_dir.rename(archived)
    return archived


def load_resumable_result(
    run_dir: Path,
    case: MatrixCase,
    seed: int,
    strategy: str,
    expected_fingerprint: str,
) -> Optional[Dict]:
    """Load a complete, provenance-matching checkpoint or return None."""
    path = Path(run_dir) / "run_result.json"
    try:
        result = _load_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    if not result:
        return None
    if result.get("name") != case.name:
        return None
    if int(result.get("seed", -1)) != int(seed):
        return None
    if result.get("strategy") != strategy:
        return None
    if result.get("run_fingerprint") != expected_fingerprint:
        return None
    required = ("collector_returncode", "passed", "missing_counts", "counts")
    if any(key not in result for key in required):
        return None
    if result.get("collector_returncode") != 0:
        return None
    if result.get("missing_counts") != []:
        return None
    counts = result.get("counts")
    if not isinstance(counts, dict):
        return None
    required_counts = list(REQUIRED_PIPELINE_COUNTS)
    try:
        result_count_values = {
            key: int(counts.get(key, 0) or 0) for key in required_counts}
    except (TypeError, ValueError):
        return None
    if any(value <= 0 for value in result_count_values.values()):
        return None
    required_artifacts = [
        "metadata.json",
        "source_summary.json",
        "attribution.json",
        "launch.log",
    ]
    required_artifacts.extend(COUNT_CSV_ARTIFACTS.values())
    if any(not (Path(run_dir) / filename).exists()
           for filename in required_artifacts):
        return None
    try:
        metadata = _load_json(Path(run_dir) / "metadata.json")
        source_summary = _load_json(Path(run_dir) / "source_summary.json")
        attribution = _load_json(Path(run_dir) / "attribution.json")
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    metadata_counts = metadata.get("counts") if isinstance(metadata, dict) else None
    if not isinstance(metadata_counts, dict):
        return None
    try:
        metadata_count_values = {
            key: int(metadata_counts.get(key, 0) or 0) for key in required_counts}
    except (TypeError, ValueError):
        return None
    if any(value <= 0 for value in metadata_count_values.values()):
        return None
    if metadata_count_values != result_count_values:
        return None
    for count_key, filename in COUNT_CSV_ARTIFACTS.items():
        if _csv_data_row_count(Path(run_dir) / filename) != result_count_values[count_key]:
            return None
    summary_keys = {
        "source_recall", "source_precision", "truth_count", "matched_count",
        "confirmed_count", "duplicate_confirmations",
    }
    if not isinstance(source_summary, dict) or not summary_keys.issubset(source_summary):
        return None
    if (not isinstance(attribution, dict)
            or not isinstance(attribution.get("failure_counts"), dict)):
        return None
    for key in summary_keys:
        if key not in result:
            continue
        artifact_value = source_summary.get(key)
        result_value = result.get(key)
        if isinstance(artifact_value, (int, float)) and isinstance(
                result_value, (int, float)):
            if not math.isclose(
                    float(artifact_value), float(result_value),
                    rel_tol=0.0, abs_tol=1e-12):
                return None
        elif artifact_value != result_value:
            return None
    if ("failure_counts" in result
            and attribution.get("failure_counts") != result.get("failure_counts")):
        return None
    if ("truth_count" in result and "n_truth_sources" in attribution
            and int(attribution["n_truth_sources"]) != int(result["truth_count"])):
        return None
    if ("matched_count" in result and "n_matched" in attribution
            and int(attribution["n_matched"]) != int(result["matched_count"])):
        return None
    localization_errors = source_summary.get("localization_errors_m")
    if ("matched_count" in result and localization_errors is not None
            and (not isinstance(localization_errors, list)
                 or len(localization_errors) != int(result["matched_count"]))):
        return None
    if (Path(run_dir) / "launch.log").stat().st_size <= 0:
        return None
    return result


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
        f"use_sim_time:={'true' if strategy in ('fast', 'dual', 'gp_ucb') else 'false'}",
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
    required_counts = list(REQUIRED_PIPELINE_COUNTS)
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
    parser.add_argument("--strategy", choices=["full", "frontier", "levy", "residual", "fast", "dual", "gp_ucb"], default="full")
    parser.add_argument("--jitter", type=float, default=0.0, help="scenario_jitter_std_m")
    parser.add_argument("--out-root", default="", help="default: /tmp/thermal_matrix_<timestamp>")
    parser.add_argument("--duration", type=float, default=120.0, help="collector duration per run")
    parser.add_argument("--warmup", type=float, default=36.0, help="launch warmup before payload starts")
    parser.add_argument("--domain-start", type=int, default=71)
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--health-only", action="store_true", help="run Nav2 health check instead of collecting data")
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse provenance-matching, artifact-complete run_result.json checkpoints")
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
    runtime_fingerprint = runtime_bundle_fingerprint()

    runs: List[Dict] = []
    run_idx = 0
    for case in cases:
        for seed in seeds:
            run_idx += 1
            require_unchanged_runtime(
                runtime_fingerprint, runtime_bundle_fingerprint())
            domain_id = args.domain_start + ((run_idx - 1) % domain_span)
            run_dir = out_root / case.name / f"seed{seed}"
            run_fingerprint = build_run_fingerprint(
                case=case,
                seed=seed,
                strategy=args.strategy,
                duration_s=args.duration,
                warmup_s=args.warmup,
                jitter_std_m=args.jitter,
                min_recall=args.min_recall,
                health_only=args.health_only,
                runtime_fingerprint=runtime_fingerprint,
            )
            if args.resume and not args.health_only:
                cached = load_resumable_result(
                    run_dir, case, seed=seed, strategy=args.strategy,
                    expected_fingerprint=run_fingerprint)
                if cached is not None:
                    runs.append(cached)
                    print(
                        f"[matrix] resume {run_idx}/{n_runs}: {case.name} "
                        f"seed={seed} strategy={args.strategy}")
                    continue
            archived = prepare_run_attempt(
                run_dir,
                resume=bool(args.resume and not args.health_only),
                expected_fingerprint=run_fingerprint,
            )
            if archived is not None:
                print(f"[matrix] archived incomplete attempt: {archived}")
            print(f"[matrix] run {run_idx}/{n_runs}: {case.name} seed={seed} strategy={args.strategy}")
            result = run_case(
                case=case,
                run_dir=run_dir,
                duration_s=args.duration,
                warmup_s=args.warmup,
                domain_id=domain_id,
                min_recall=args.min_recall,
                seed=seed,
                strategy=args.strategy,
                jitter_std_m=args.jitter,
                health_only=args.health_only,
            )
            require_unchanged_runtime(
                runtime_fingerprint, runtime_bundle_fingerprint())
            result["run_fingerprint"] = run_fingerprint
            if not args.health_only:
                write_run_result(run_dir, result)
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
            "min_recall": args.min_recall,
            "health_only": args.health_only,
            "resume": args.resume,
            "run_fingerprint_version": RUN_FINGERPRINT_VERSION,
            "runtime_bundle_sha256": runtime_fingerprint,
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
