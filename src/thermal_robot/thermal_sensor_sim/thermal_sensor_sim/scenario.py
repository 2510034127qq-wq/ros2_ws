"""Scenario loading and stationary heat-source models for thermal simulation."""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - package dependency is declared for ROS runtime.
    yaml = None


SPAWN_X = -6.0
SPAWN_Y = 0.0
SENSOR_FOV_X = 4.0
SENSOR_FOV_Y = 3.0


@dataclass
class SourceState:
    source_id: str
    x: float
    y: float
    amplitude: float
    sigma_m: float
    active: bool = True


@dataclass
class HeatSource:
    source_id: str
    world_x: float
    world_y: float
    amplitude: float
    sigma_m: float
    seed: int = 0

    def state(self, t: float) -> SourceState:
        return SourceState(self.source_id, self.world_x, self.world_y,
                           self.amplitude, self.sigma_m)


@dataclass
class ThermalScenario:
    spawn_x: float = SPAWN_X
    spawn_y: float = SPAWN_Y
    fov_x: float = SENSOR_FOV_X
    fov_y: float = SENSOR_FOV_Y
    sources: List[HeatSource] = field(default_factory=list)

    def active_states(self, t: float) -> List[SourceState]:
        states = [src.state(t) for src in self.sources]
        return [state for state in states if state.active]

    def all_states(self, t: float) -> List[SourceState]:
        return [src.state(t) for src in self.sources]


def apply_run_seed(scenario: "ThermalScenario", run_seed: int,
                   jitter_std_m: float = 0.0,
                   amplitude_jitter_frac: float = 0.0) -> "ThermalScenario":
    """Apply a deterministic run-level seed to scenario-local randomness."""
    if run_seed is None or int(run_seed) <= 0:
        return scenario
    run_seed = int(run_seed)
    for src in scenario.sources:
        rng = random.Random(run_seed * 1000003
                            + zlib.crc32(src.source_id.encode('utf-8')))
        src.seed = src.seed + run_seed * 1000
        if jitter_std_m > 0.0:
            src.world_x += rng.gauss(0.0, jitter_std_m)
            src.world_y += rng.gauss(0.0, jitter_std_m)
        if amplitude_jitter_frac > 0.0:
            src.amplitude = max(1.0, src.amplitude
                                * (1.0 + rng.gauss(0.0, amplitude_jitter_frac)))
    return scenario


def default_config_b_sources() -> List[HeatSource]:
    return [
        HeatSource("SA_left", -1.0, 3.5, 35.0, 1.1),
        HeatSource("SB_far", 6.0, -3.0, 22.0, 0.9),
        HeatSource("SC_weak", -5.0, -5.5, 16.0, 0.8),
    ]


def default_config_b_scenario(num_sources: int = 3) -> ThermalScenario:
    sources = default_config_b_sources()
    requested = int(num_sources)
    n_src = len(sources) if requested <= 0 else max(1, min(requested, len(sources)))
    return ThermalScenario(sources=sources[:n_src])


def load_scenario_file(path: str, num_sources: int = 0) -> ThermalScenario:
    if not path:
        return default_config_b_scenario(num_sources)
    if yaml is None:
        raise RuntimeError("python3-yaml is required for scenario_file support")
    with open(Path(path).expanduser(), "r") as f:
        raw = yaml.safe_load(f) or {}

    robot = raw.get("robot", {})
    spawn = robot.get("spawn", robot)
    sensor = raw.get("sensor", {})
    scenario = ThermalScenario(
        spawn_x=float(spawn.get("x", spawn.get("spawn_x", SPAWN_X))),
        spawn_y=float(spawn.get("y", spawn.get("spawn_y", SPAWN_Y))),
        fov_x=float(sensor.get("fov_x", sensor.get("fov_x_m", SENSOR_FOV_X))),
        fov_y=float(sensor.get("fov_y", sensor.get("fov_y_m", SENSOR_FOV_Y))),
        sources=[],
    )
    for idx, item in enumerate(raw.get("sources", [])):
        scenario.sources.append(_source_from_mapping(idx, item))
    if not scenario.sources:
        scenario.sources = default_config_b_sources()
    requested = int(num_sources)
    if requested > 0:
        scenario.sources = scenario.sources[: max(1, min(requested, len(scenario.sources)))]
    return scenario


def _source_from_mapping(idx: int, item: Dict[str, Any]) -> HeatSource:
    xy = item.get("xy", item.get("position", item.get("world", {})))
    if isinstance(xy, dict):
        x = float(xy.get("x", item.get("world_x", 0.0)))
        y = float(xy.get("y", item.get("world_y", 0.0)))
    else:
        x, y = _xy_pair(xy)
    if (item.get("motion", item.get("motion_type", "static")) != "static"
            or item.get("motion_params") or item.get("active_schedule")
            or item.get("strength_drift") or item.get("drift")):
        raise ValueError("Only stationary, continuously emitting sources are supported")
    return HeatSource(
        source_id=str(item.get("id", item.get("name", f"source_{idx}"))),
        world_x=x,
        world_y=y,
        amplitude=float(item.get("amplitude", item.get("strength", item.get("amp", 20.0)))),
        sigma_m=float(item.get("sigma_m", item.get("sigma", 1.0))),
        seed=int(item.get("seed", idx + 1)),
    )




def _xy_pair(value: Any) -> Tuple[float, float]:
    if isinstance(value, dict):
        return float(value.get("x", 0.0)), float(value.get("y", 0.0))
    if isinstance(value, Sequence) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return 0.0, 0.0
