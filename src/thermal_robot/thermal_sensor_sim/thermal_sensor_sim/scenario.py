"""Scenario loading and dynamic heat-source models for thermal simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
class DynamicHeatSource:
    source_id: str
    world_x: float
    world_y: float
    amplitude: float
    sigma_m: float
    motion: str = "static"
    motion_params: Dict[str, Any] = field(default_factory=dict)
    active_schedule: List[Tuple[float, float]] = field(default_factory=list)
    strength_drift_per_s: float = 0.0
    strength_sin_amplitude: float = 0.0
    strength_sin_period_s: float = 1.0
    seed: int = 0

    _rw_x: Optional[float] = field(default=None, init=False, repr=False)
    _rw_y: Optional[float] = field(default=None, init=False, repr=False)
    _rw_t: Optional[float] = field(default=None, init=False, repr=False)
    _rw_heading: float = field(default=0.0, init=False, repr=False)

    def is_active(self, t: float) -> bool:
        if self.active_schedule:
            return any(start <= t <= end for start, end in self.active_schedule)
        if self.motion == "appear_disappear":
            period = float(self.motion_params.get("period_s", 30.0))
            duty = float(self.motion_params.get("duty_cycle", 0.5))
            phase = float(self.motion_params.get("phase_s", 0.0))
            if period <= 0.0:
                return True
            return ((t + phase) % period) <= period * max(0.0, min(1.0, duty))
        return True

    def strength_at(self, t: float) -> float:
        amp = self.amplitude + self.strength_drift_per_s * t
        if self.strength_sin_amplitude:
            period = max(1e-6, self.strength_sin_period_s)
            amp += self.strength_sin_amplitude * math.sin(2.0 * math.pi * t / period)
        return max(0.0, float(amp))

    def position(self, t: float) -> Tuple[float, float]:
        motion = self.motion
        p = self.motion_params
        if motion in ("static", "appear_disappear"):
            return self.world_x, self.world_y
        if motion == "linear":
            vx = float(p.get("vx", p.get("velocity_x", 0.0)))
            vy = float(p.get("vy", p.get("velocity_y", 0.0)))
            return self.world_x + vx * t, self.world_y + vy * t
        if motion == "circular":
            cx = float(p.get("center_x", self.world_x))
            cy = float(p.get("center_y", self.world_y))
            radius = float(p.get("radius", p.get("radius_m", 1.0)))
            omega = float(p.get("angular_speed", p.get("angular_speed_rad_s", 0.2)))
            phase = float(p.get("phase", 0.0))
            return cx + radius * math.cos(omega * t + phase), cy + radius * math.sin(omega * t + phase)
        if motion == "sinusoidal":
            ax = float(p.get("amplitude_x", p.get("amplitude_m", 1.0)))
            ay = float(p.get("amplitude_y", 0.0))
            period = max(1e-6, float(p.get("period_s", 20.0)))
            phase = float(p.get("phase", 0.0))
            s = math.sin(2.0 * math.pi * t / period + phase)
            return self.world_x + ax * s, self.world_y + ay * s
        if motion == "waypoint_loop":
            return self._waypoint_loop_position(t)
        if motion == "random_walk":
            return self._random_walk_position(t)
        if motion == "random_waypoint":
            return self._random_walk_position(t)
        return self.world_x, self.world_y

    def state(self, t: float) -> SourceState:
        x, y = self.position(t)
        return SourceState(
            source_id=self.source_id,
            x=x,
            y=y,
            amplitude=self.strength_at(t),
            sigma_m=self.sigma_m,
            active=self.is_active(t),
        )

    def _waypoint_loop_position(self, t: float) -> Tuple[float, float]:
        waypoints = self.motion_params.get("waypoints") or []
        pts = [_xy_pair(p) for p in waypoints]
        if len(pts) < 2:
            return self.world_x, self.world_y
        speed = max(1e-6, float(self.motion_params.get("speed_m_s", self.motion_params.get("speed", 0.5))))
        segments: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = []
        total = 0.0
        for i, a in enumerate(pts):
            b = pts[(i + 1) % len(pts)]
            d = math.hypot(b[0] - a[0], b[1] - a[1])
            if d <= 1e-9:
                continue
            segments.append((a, b, d))
            total += d
        if total <= 1e-9:
            return pts[0]
        s = (t * speed) % total
        for a, b, d in segments:
            if s <= d:
                u = s / d
                return a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u
            s -= d
        return segments[-1][1]

    def _random_walk_position(self, t: float) -> Tuple[float, float]:
        if self._rw_x is None or self._rw_y is None:
            self._rw_x = self.world_x
            self._rw_y = self.world_y
            self._rw_t = t
            self._rw_heading = _stable_angle(self.seed)
            return self._rw_x, self._rw_y
        prev_t = float(self._rw_t) if self._rw_t is not None else t
        dt = max(0.0, min(1.0, t - prev_t))
        self._rw_t = t
        speed = float(self.motion_params.get("speed_m_s", self.motion_params.get("speed", 0.2)))
        turn_rate = float(self.motion_params.get("turn_rate_rad_s", 0.6))
        self._rw_heading += turn_rate * math.sin(0.37 * t + self.seed)
        self._rw_x += speed * dt * math.cos(self._rw_heading)
        self._rw_y += speed * dt * math.sin(self._rw_heading)
        radius = self.motion_params.get("bounds_radius_m")
        if radius is not None:
            r = float(radius)
            dx = self._rw_x - self.world_x
            dy = self._rw_y - self.world_y
            d = math.hypot(dx, dy)
            if d > r > 0.0:
                self._rw_x = self.world_x + dx / d * r
                self._rw_y = self.world_y + dy / d * r
                self._rw_heading += math.pi * 0.7
        return self._rw_x, self._rw_y


@dataclass
class ThermalScenario:
    spawn_x: float = SPAWN_X
    spawn_y: float = SPAWN_Y
    fov_x: float = SENSOR_FOV_X
    fov_y: float = SENSOR_FOV_Y
    sources: List[DynamicHeatSource] = field(default_factory=list)

    def active_states(self, t: float) -> List[SourceState]:
        states = [src.state(t) for src in self.sources]
        return [state for state in states if state.active]

    def all_states(self, t: float) -> List[SourceState]:
        return [src.state(t) for src in self.sources]


def default_config_b_sources() -> List[DynamicHeatSource]:
    return [
        DynamicHeatSource("SA_left", -1.0, 3.5, 35.0, 1.1),
        DynamicHeatSource("SB_far", 6.0, -3.0, 22.0, 0.9),
        DynamicHeatSource("SC_weak", -5.0, -5.5, 16.0, 0.8),
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


def _source_from_mapping(idx: int, item: Dict[str, Any]) -> DynamicHeatSource:
    xy = item.get("xy", item.get("position", item.get("world", {})))
    if isinstance(xy, dict):
        x = float(xy.get("x", item.get("world_x", 0.0)))
        y = float(xy.get("y", item.get("world_y", 0.0)))
    else:
        x, y = _xy_pair(xy)
    drift = item.get("strength_drift", item.get("drift", {}))
    if isinstance(drift, dict):
        drift_rate = float(drift.get("rate_per_s", drift.get("linear_per_s", 0.0)))
        sin_amp = float(drift.get("sin_amplitude", 0.0))
        sin_period = float(drift.get("sin_period_s", 1.0))
    else:
        drift_rate = float(drift or 0.0)
        sin_amp = 0.0
        sin_period = 1.0
    schedule = item.get("active_schedule", [])
    return DynamicHeatSource(
        source_id=str(item.get("id", item.get("name", f"source_{idx}"))),
        world_x=x,
        world_y=y,
        amplitude=float(item.get("amplitude", item.get("strength", item.get("amp", 20.0)))),
        sigma_m=float(item.get("sigma_m", item.get("sigma", 1.0))),
        motion=str(item.get("motion", item.get("motion_type", "static"))),
        motion_params=dict(item.get("motion_params", {})),
        active_schedule=_parse_schedule(schedule),
        strength_drift_per_s=drift_rate,
        strength_sin_amplitude=sin_amp,
        strength_sin_period_s=sin_period,
        seed=int(item.get("seed", idx + 1)),
    )


def _parse_schedule(raw: Iterable[Any]) -> List[Tuple[float, float]]:
    out = []
    for item in raw:
        if isinstance(item, dict):
            out.append((float(item.get("start", 0.0)), float(item.get("end", float("inf")))))
        else:
            start, end = item
            out.append((float(start), float(end)))
    return out


def _xy_pair(value: Any) -> Tuple[float, float]:
    if isinstance(value, dict):
        return float(value.get("x", 0.0)), float(value.get("y", 0.0))
    if isinstance(value, Sequence) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return 0.0, 0.0


def _stable_angle(seed: int) -> float:
    return ((seed * 1103515245 + 12345) % 6283) / 1000.0 - math.pi
