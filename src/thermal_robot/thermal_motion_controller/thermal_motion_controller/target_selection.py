"""Source-seeking target selection for the thermal controller.

The controller state machine owns mission phases and motion execution.  This
module owns the source-seek objective policy: given the current robot pose,
thermal map, tracker sources, confirmed sources, and phase context, choose the
next world-frame target and an execution preference.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from thermal_motion_controller.planning import (
    PlannerSource,
    PlannerWeights,
    select_coverage_ring_target,
    select_exploration_sector_yaw,
    select_information_gain_target,
)


SOURCE_SEEK_STRATEGY = "source_seek"
EXECUTION_NAV2_PREFERRED = "nav2_preferred"
EXECUTION_DIRECT_FIRST = "direct_first"


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class SourceSeekConfig:
    source_repulsion_k: float
    source_repulsion_min_dist: float
    source_exclusion_radius: float
    frontier_safe_buf: float
    survey_safe_dist: float
    pc_min_d: float
    pc_dist_sigma: float
    coverage_directional_weight: float
    departure_directional_weight: float
    coverage_ring_min_d: float
    coverage_ring_max_d: float
    coverage_ring_fov_radius: float
    coverage_ring_angles: int
    coverage_ring_rings: int
    coverage_recent_yaw_penalty: float
    coverage_recent_yaw_window: int
    source_set_expansion_min_sources: int
    source_set_expansion_max_d: float
    source_set_expansion_directional_weight: float
    source_set_outward_directional_weight: float
    source_set_outward_bonus: float
    source_set_lateral_directional_weight: float
    source_set_lateral_bonus: float
    source_set_lateral_max_d: float
    departure_dist: float
    planner_weights: PlannerWeights
    planner_map_stale_s: float


@dataclass
class SourceSeekContext:
    now: float
    robot_wx: float
    robot_wy: float
    odom_yaw: float
    spawn_x: float
    spawn_y: float
    search_rounds: int
    coarse_wp_count: int
    found_sources: Sequence[Tuple[float, float, float]]
    tracker_sources: Sequence[Dict]
    tracker_sources_t: float
    thermal_map: Optional[Dict]
    thermal_map_t: float
    belief_map: object


@dataclass
class StrategyTarget:
    x: float
    y: float
    score: float = 0.0
    reason: str = "unknown"
    execution_hint: str = EXECUTION_NAV2_PREFERRED
    strategy: str = SOURCE_SEEK_STRATEGY
    metadata: Dict = field(default_factory=dict)

    @property
    def xy(self) -> Tuple[float, float]:
        return self.x, self.y


class SourceSeekTargetSelector:
    """Target selector for the current source-seeking behavior.

    State retained here is target-policy state only: recent coverage directions
    and sweep indices.  Mission state, Nav2 state, and direct command fallback
    stay in controller_node.py.
    """

    def __init__(self, config: SourceSeekConfig):
        self.config = config
        self._coverage_recent_yaws: List[float] = []
        self._single_source_sweep_idx = 0
        self._source_set_sweep_idx = 0

    def reset_source_sweeps(self) -> None:
        self._single_source_sweep_idx = 0
        self._source_set_sweep_idx = 0

    def select_frontier(
        self,
        ctx: SourceSeekContext,
        min_d: float = 3.0,
        max_d: float = 16.0,
        dist_sigma: float = 8.0,
    ) -> Optional[StrategyTarget]:
        planner_sources = self._planner_sources(ctx)
        preferred_yaw = None
        directional_weight = 0.0
        if not planner_sources:
            sector = self._map_sector_yaw(
                ctx,
                min_d=min_d,
                max_d=max_d,
                safe_dist=self._safe_dist(),
                phase_yaw=self._phase_yaw_for_coverage(ctx),
            )
            preferred_yaw = sector.yaw if sector is not None else self._phase_yaw_for_coverage(ctx)
            directional_weight = self.config.coverage_directional_weight

        target = None
        if not planner_sources:
            target = self._map_coverage_ring(
                ctx,
                min_radius=max(min_d, self.config.coverage_ring_min_d),
                max_radius=min(max_d, self.config.coverage_ring_max_d),
                safe_dist=self._safe_dist(),
                preferred_yaw=preferred_yaw,
                directional_weight=directional_weight,
                min_travel_d=min_d,
                max_travel_d=max_d,
            )
        if target is None:
            target = self._map_best_frontier(
                ctx,
                min_d=min_d,
                max_d=max_d,
                safe_dist=self._safe_dist(),
                preferred_yaw=preferred_yaw,
                directional_weight=directional_weight,
            )
        if target is None and ctx.belief_map is not None:
            fallback = ctx.belief_map.best_frontier(
                ctx.robot_wx,
                ctx.robot_wy,
                min_d=min_d,
                max_d=max_d,
                dist_sigma=dist_sigma,
                known_sources=self._known_source_positions(ctx),
                safe_dist=self._safe_dist(),
            )
            target = self._from_tuple(fallback, "belief_fallback")
        if target is not None:
            self._record_if_coverage(ctx, target)
        return target

    def select_levy_jump(
        self,
        ctx: SourceSeekContext,
        step: float,
    ) -> StrategyTarget:
        target = self._map_best_frontier(
            ctx,
            min_d=step * 0.4,
            max_d=step * 1.6,
            safe_dist=self._safe_dist(),
        )
        if target is None and ctx.belief_map is not None:
            fallback = ctx.belief_map.best_frontier(
                ctx.robot_wx,
                ctx.robot_wy,
                min_d=step * 0.4,
                max_d=step * 1.6,
                known_sources=self._known_source_positions(ctx),
                safe_dist=self._safe_dist(),
            )
            target = self._from_tuple(fallback, "belief_fallback")
        direction = (
            math.atan2(target.y - ctx.robot_wy, target.x - ctx.robot_wx)
            if target is not None else random.uniform(-math.pi, math.pi)
        )
        return StrategyTarget(
            x=ctx.robot_wx + step * math.cos(direction),
            y=ctx.robot_wy + step * math.sin(direction),
            reason="levy_jump",
            execution_hint=EXECUTION_NAV2_PREFERRED,
            metadata={"direction": direction, "step": step},
        )

    def select_coarse_waypoint(
        self,
        ctx: SourceSeekContext,
        min_d: float,
        max_d: float,
    ) -> Optional[StrategyTarget]:
        single_expansion = self._single_source_expansion_target(ctx, min_d)
        if single_expansion is not None:
            target = single_expansion
            target.execution_hint = EXECUTION_NAV2_PREFERRED
            return target

        if len(ctx.found_sources) >= self.config.source_set_expansion_min_sources:
            expansion = self._source_set_expansion_target(ctx, min_d)
            if expansion is not None:
                expansion.execution_hint = EXECUTION_DIRECT_FIRST
                return expansion

        planner_sources = self._planner_sources(ctx)
        preferred_yaw = None
        directional_weight = 0.0
        if not planner_sources:
            sector = self._map_sector_yaw(
                ctx,
                min_d=min_d,
                max_d=max_d,
                safe_dist=self.config.survey_safe_dist,
                phase_yaw=self._phase_yaw_for_coverage(ctx),
            )
            preferred_yaw = sector.yaw if sector is not None else self._phase_yaw_for_coverage(ctx)
            directional_weight = self.config.coverage_directional_weight

        target = None
        if not planner_sources:
            target = self._map_coverage_ring(
                ctx,
                min_radius=max(min_d, self.config.coverage_ring_min_d),
                max_radius=min(max_d, self.config.coverage_ring_max_d),
                safe_dist=self.config.survey_safe_dist,
                preferred_yaw=preferred_yaw,
                directional_weight=directional_weight,
                min_travel_d=min_d,
                max_travel_d=max_d,
            )
        if target is None:
            target = self._map_best_frontier(
                ctx,
                min_d=min_d,
                max_d=max_d,
                safe_dist=self.config.survey_safe_dist,
                preferred_yaw=preferred_yaw,
                directional_weight=directional_weight,
            )
        if target is None and ctx.belief_map is not None:
            fallback = ctx.belief_map.best_frontier(
                ctx.robot_wx,
                ctx.robot_wy,
                min_d=min_d,
                max_d=max_d,
                dist_sigma=10.0,
                heat_prior=0.0,
                known_sources=self._known_source_positions(ctx),
                safe_dist=self.config.survey_safe_dist,
            )
            target = self._from_tuple(fallback, "belief_fallback")
        if target is not None:
            self._record_if_coverage(ctx, target)
        return target

    def select_departure(
        self,
        ctx: SourceSeekContext,
        min_travel_d: float,
    ) -> StrategyTarget:
        cx, cy = self._sources_centroid(ctx)

        single_expansion = self._single_source_expansion_target(ctx, min_travel_d)
        if single_expansion is not None:
            single_expansion.execution_hint = EXECUTION_DIRECT_FIRST
            single_expansion.metadata.setdefault("centroid", (cx, cy))
            return single_expansion

        expansion = self._source_set_expansion_target(ctx, min_travel_d)
        if expansion is not None:
            expansion.execution_hint = EXECUTION_DIRECT_FIRST
            expansion.metadata.setdefault("centroid", (cx, cy))
            return expansion

        best_ring = self._best_departure_ring(ctx, cx, cy, min_travel_d)
        if best_ring is not None:
            best_ring.execution_hint = EXECUTION_DIRECT_FIRST
            return best_ring

        max_d = max(self.config.pc_min_d + 2.0, self.config.departure_dist)
        sector = self._map_sector_yaw(
            ctx,
            min_d=min_travel_d,
            max_d=max_d,
            safe_dist=max(self.config.survey_safe_dist, self._safe_dist()),
            phase_yaw=self._phase_yaw_away_from_sources(ctx),
        )
        preferred_yaw = sector.yaw if sector is not None else self._phase_yaw_away_from_sources(ctx)

        ring = self._map_coverage_ring(
            ctx,
            min_radius=max(self._safe_dist() + 1.0, self.config.coverage_ring_min_d),
            max_radius=max(self.config.coverage_ring_max_d, self.config.departure_dist),
            safe_dist=max(self.config.survey_safe_dist, self._safe_dist()),
            preferred_yaw=preferred_yaw,
            directional_weight=self.config.departure_directional_weight,
            anchor=(cx, cy),
            min_travel_d=min_travel_d,
            max_travel_d=max_d,
        )
        if ring is not None:
            ring.execution_hint = EXECUTION_DIRECT_FIRST
            ring.metadata.update({"centroid": (cx, cy), "yaw": preferred_yaw, "label": "ring"})
            self._record_if_coverage(ctx, ring)
            return ring

        frontier = self._map_best_frontier(
            ctx,
            min_d=min_travel_d,
            max_d=max_d,
            safe_dist=max(self.config.survey_safe_dist, self._safe_dist()),
            preferred_yaw=preferred_yaw,
            directional_weight=self.config.departure_directional_weight,
        )
        if frontier is not None:
            frontier.execution_hint = EXECUTION_DIRECT_FIRST
            frontier.metadata.update({"centroid": (cx, cy), "yaw": preferred_yaw, "label": "map"})
            return frontier

        return self._fallback_departure(ctx, cx, cy, preferred_yaw)

    def select_escape_yaw(self, ctx: SourceSeekContext, frontier_bias: float) -> float:
        if ctx.found_sources:
            vx, vy = self._repulsion_vec(ctx)
            rep_yaw = math.atan2(vy, vx)
        else:
            rep_yaw = math.atan2(-math.sin(ctx.odom_yaw), -math.cos(ctx.odom_yaw))
        if frontier_bias <= 0.0:
            return rep_yaw
        frontier = self._map_best_frontier(
            ctx,
            min_d=4.0,
            max_d=14.0,
            safe_dist=self._safe_dist(),
        )
        if frontier is None and ctx.belief_map is not None:
            fallback = ctx.belief_map.best_frontier(
                ctx.robot_wx,
                ctx.robot_wy,
                min_d=4.0,
                max_d=14.0,
                known_sources=self._known_source_positions(ctx),
                safe_dist=self._safe_dist(),
            )
            frontier = self._from_tuple(fallback, "belief_fallback")
        if frontier is None:
            return rep_yaw
        fr_yaw = math.atan2(frontier.y - ctx.robot_wy, frontier.x - ctx.robot_wx)
        rx = (1.0 - frontier_bias) * math.cos(rep_yaw) + frontier_bias * math.cos(fr_yaw)
        ry = (1.0 - frontier_bias) * math.sin(rep_yaw) + frontier_bias * math.sin(fr_yaw)
        return math.atan2(ry, rx)

    def _from_tuple(self, target, default_reason: str) -> Optional[StrategyTarget]:
        if target is None:
            return None
        reason = target[3] if len(target) > 3 else default_reason
        return StrategyTarget(
            x=float(target[0]),
            y=float(target[1]),
            score=float(target[2]) if len(target) > 2 else 0.0,
            reason=str(reason),
        )

    def _record_if_coverage(self, ctx: SourceSeekContext, target: StrategyTarget) -> None:
        if target.reason in ("coverage_ring", "annular_coverage", "source_verify_ring"):
            self._remember_coverage_yaw(ctx, target.x, target.y)

    def _remember_coverage_yaw(self, ctx: SourceSeekContext, tx: float, ty: float) -> None:
        self._coverage_recent_yaws.append(math.atan2(ty - ctx.robot_wy, tx - ctx.robot_wx))
        max_len = max(1, int(self.config.coverage_recent_yaw_window))
        if len(self._coverage_recent_yaws) > max_len:
            del self._coverage_recent_yaws[:-max_len]

    def _map_available(self, ctx: SourceSeekContext) -> bool:
        return (
            ctx.thermal_map is not None
            and (ctx.now - ctx.thermal_map_t) <= self.config.planner_map_stale_s
        )

    def _known_source_positions(self, ctx: SourceSeekContext) -> List[Tuple[float, float]]:
        return [(s[0], s[1]) for s in ctx.found_sources]

    def _sources_centroid(self, ctx: SourceSeekContext) -> Tuple[float, float]:
        if not ctx.found_sources:
            return ctx.spawn_x, ctx.spawn_y
        cx = sum(s[0] for s in ctx.found_sources) / len(ctx.found_sources)
        cy = sum(s[1] for s in ctx.found_sources) / len(ctx.found_sources)
        return cx, cy

    def _safe_dist(self) -> float:
        return self.config.source_exclusion_radius + self.config.frontier_safe_buf

    def _phase_yaw_for_coverage(self, ctx: SourceSeekContext) -> float:
        golden = math.pi * (3.0 - math.sqrt(5.0))
        return ctx.odom_yaw + golden * (ctx.search_rounds + ctx.coarse_wp_count)

    def _phase_yaw_away_from_sources(self, ctx: SourceSeekContext) -> float:
        vx, vy = self._repulsion_vec(ctx)
        if math.hypot(vx, vy) > 1e-6:
            return math.atan2(vy, vx)
        return self._phase_yaw_for_coverage(ctx)

    def _repulsion_vec(self, ctx: SourceSeekContext) -> Tuple[float, float]:
        if not ctx.found_sources:
            return 0.0, 0.0
        rx = ry = 0.0
        for sx, sy, _ in ctx.found_sources:
            dx = ctx.robot_wx - sx
            dy = ctx.robot_wy - sy
            dist = max(math.hypot(dx, dy), self.config.source_repulsion_min_dist)
            scale = self.config.source_repulsion_k / (dist * dist)
            rx += scale * dx / dist
            ry += scale * dy / dist
        mag = math.hypot(rx, ry)
        return (rx / mag, ry / mag) if mag > 1e-6 else (0.0, 0.0)

    def _is_near_known_pos(
        self,
        ctx: SourceSeekContext,
        x: float,
        y: float,
        radius: Optional[float] = None,
    ) -> bool:
        r = self.config.source_exclusion_radius if radius is None else radius
        return any(math.hypot(x - sx, y - sy) < r for sx, sy, _ in ctx.found_sources)

    def _planner_sources(self, ctx: SourceSeekContext) -> List[PlannerSource]:
        if (ctx.now - ctx.tracker_sources_t) > self.config.planner_map_stale_s * 2.0:
            return []
        return [
            PlannerSource(
                x=s["x"],
                y=s["y"],
                probability=s["probability"],
                status=s["status"],
                confidence=s["confidence"],
            )
            for s in ctx.tracker_sources
            if s["status"] in ("candidate", "confirmed")
            and not self._is_near_known_pos(ctx, s["x"], s["y"], radius=self.config.source_exclusion_radius)
        ]

    def _map_sector_yaw(
        self,
        ctx: SourceSeekContext,
        min_d: float,
        max_d: float,
        safe_dist: float,
        phase_yaw: float,
    ):
        if not self._map_available(ctx):
            return None
        m = ctx.thermal_map
        return select_exploration_sector_yaw(
            robot_wx=ctx.robot_wx,
            robot_wy=ctx.robot_wy,
            width=m["width"],
            height=m["height"],
            resolution=m["resolution"],
            origin_x=m["origin_x"],
            origin_y=m["origin_y"],
            temperature_variance=m["temperature_variance"],
            confidence=m["confidence"],
            visit_count=m["visit_count"],
            last_seen_age_s=m["last_seen_age_s"],
            known_sources=self._known_source_positions(ctx),
            min_d=min_d,
            max_d=max_d,
            safe_dist=safe_dist,
            phase_yaw=phase_yaw,
        )

    def _map_best_frontier(
        self,
        ctx: SourceSeekContext,
        min_d: float,
        max_d: float,
        safe_dist: float,
        preferred_yaw: Optional[float] = None,
        directional_weight: float = 0.0,
    ) -> Optional[StrategyTarget]:
        if not self._map_available(ctx):
            return None
        m = ctx.thermal_map
        target = select_information_gain_target(
            robot_wx=ctx.robot_wx,
            robot_wy=ctx.robot_wy,
            width=m["width"],
            height=m["height"],
            resolution=m["resolution"],
            origin_x=m["origin_x"],
            origin_y=m["origin_y"],
            temperature_variance=m["temperature_variance"],
            confidence=m["confidence"],
            visit_count=m["visit_count"],
            last_seen_age_s=m["last_seen_age_s"],
            source_estimates=self._planner_sources(ctx),
            known_sources=self._known_source_positions(ctx),
            min_d=min_d,
            max_d=max_d,
            safe_dist=safe_dist,
            weights=self.config.planner_weights,
            preferred_yaw=preferred_yaw,
            directional_weight=directional_weight,
        )
        if target is None:
            return None
        return StrategyTarget(target.x, target.y, target.score, target.reason)

    def _map_coverage_ring(
        self,
        ctx: SourceSeekContext,
        min_radius: float,
        max_radius: float,
        safe_dist: float,
        preferred_yaw: Optional[float] = None,
        directional_weight: float = 0.0,
        anchor: Optional[Tuple[float, float]] = None,
        min_travel_d: Optional[float] = None,
        max_travel_d: Optional[float] = None,
        angle_span_rad: Optional[float] = None,
    ) -> Optional[StrategyTarget]:
        if not self._map_available(ctx):
            return None
        m = ctx.thermal_map
        target = select_coverage_ring_target(
            robot_wx=ctx.robot_wx,
            robot_wy=ctx.robot_wy,
            width=m["width"],
            height=m["height"],
            resolution=m["resolution"],
            origin_x=m["origin_x"],
            origin_y=m["origin_y"],
            temperature_variance=m["temperature_variance"],
            confidence=m["confidence"],
            visit_count=m["visit_count"],
            last_seen_age_s=m["last_seen_age_s"],
            source_estimates=self._planner_sources(ctx),
            known_sources=self._known_source_positions(ctx),
            min_radius=min_radius,
            max_radius=max_radius,
            safe_dist=safe_dist,
            weights=self.config.planner_weights,
            preferred_yaw=preferred_yaw,
            directional_weight=directional_weight,
            footprint_radius=self.config.coverage_ring_fov_radius,
            num_angles=max(8, self.config.coverage_ring_angles),
            num_rings=max(1, self.config.coverage_ring_rings),
            recent_yaws=tuple(self._coverage_recent_yaws),
            recent_yaw_penalty=self.config.coverage_recent_yaw_penalty,
            anchor_x=None if anchor is None else anchor[0],
            anchor_y=None if anchor is None else anchor[1],
            min_travel_d=min_travel_d,
            max_travel_d=max_travel_d,
            angle_span_rad=angle_span_rad,
        )
        if target is None:
            return None
        return StrategyTarget(target.x, target.y, target.score, target.reason)

    def _source_set_gap_yaws(self, ctx: SourceSeekContext, cx: float, cy: float) -> List[float]:
        if len(ctx.found_sources) < 2:
            return []
        bearings = sorted(wrap_angle(math.atan2(sy - cy, sx - cx)) for sx, sy, _ in ctx.found_sources)
        gaps = []
        for idx, yaw in enumerate(bearings):
            nxt = bearings[(idx + 1) % len(bearings)]
            gap = (nxt - yaw) % (2.0 * math.pi)
            if gap <= 1e-3:
                continue
            gaps.append((gap, wrap_angle(yaw + 0.5 * gap)))
        gaps.sort(key=lambda item: item[0], reverse=True)
        return [mid for _, mid in gaps]

    def _source_pair_lateral_yaws(self, ctx: SourceSeekContext) -> List[float]:
        if len(ctx.found_sources) < 2:
            return []
        best_pair = None
        best_d = -1.0
        for i, (ax, ay, _) in enumerate(ctx.found_sources):
            for bx, by, _ in ctx.found_sources[i + 1:]:
                dist = math.hypot(bx - ax, by - ay)
                if dist > best_d:
                    best_d = dist
                    best_pair = (ax, ay, bx, by)
        if best_pair is None or best_d < 1e-3:
            return []
        ax, ay, bx, by = best_pair
        axis_yaw = math.atan2(by - ay, bx - ax)
        return [wrap_angle(axis_yaw + math.pi * 0.5), wrap_angle(axis_yaw - math.pi * 0.5)]

    def _single_source_expansion_target(
        self,
        ctx: SourceSeekContext,
        min_travel_d: float,
    ) -> Optional[StrategyTarget]:
        if len(ctx.found_sources) != 1 or not self._map_available(ctx):
            return None
        cx, cy = self._sources_centroid(ctx)
        base_yaw = math.atan2(cy - ctx.spawn_y, cx - ctx.spawn_x)
        if math.hypot(cx - ctx.spawn_x, cy - ctx.spawn_y) <= 1.0:
            base_yaw = self._phase_yaw_for_coverage(ctx)
        fan_offsets = (
            -math.pi / 3.0,
            math.pi / 3.0,
            0.0,
            -2.0 * math.pi / 3.0,
            2.0 * math.pi / 3.0,
            math.pi,
        )
        fan = [
            (
                wrap_angle(base_yaw + offset),
                self.config.source_set_outward_directional_weight,
                0.0,
                "single_fan",
            )
            for offset in fan_offsets
        ]
        start = self._single_source_sweep_idx % len(fan)
        yaw_candidates = fan[start:] + fan[:start]
        sector = self._map_sector_yaw(
            ctx,
            min_d=min_travel_d,
            max_d=self.config.source_set_expansion_max_d,
            safe_dist=max(self.config.survey_safe_dist, self._safe_dist()),
            phase_yaw=base_yaw,
        )
        if sector is not None:
            yaw_candidates.append((
                sector.yaw,
                self.config.source_set_expansion_directional_weight,
                0.0,
                "single_map_sector",
            ))
        yaw_candidates.append((
            self._phase_yaw_for_coverage(ctx),
            self.config.source_set_expansion_directional_weight,
            0.0,
            "coverage_phase",
        ))

        best = None
        best_yaw = None
        best_label = "single_source"
        best_eval_score = -float("inf")
        seen = set()
        for order_idx, (yaw, directional_weight, bonus, label) in enumerate(yaw_candidates):
            key = round(wrap_angle(yaw), 2)
            if key in seen:
                continue
            seen.add(key)
            target = self._map_coverage_ring(
                ctx,
                min_radius=max(self._safe_dist() + 1.0, self.config.coverage_ring_min_d),
                max_radius=max(self.config.coverage_ring_max_d, self.config.source_set_expansion_max_d),
                safe_dist=max(self.config.survey_safe_dist, self._safe_dist()),
                preferred_yaw=yaw,
                directional_weight=directional_weight,
                anchor=(cx, cy),
                min_travel_d=min_travel_d,
                max_travel_d=max(min_travel_d, self.config.source_set_expansion_max_d),
            )
            if target is None:
                continue
            sequence_bonus = max(0.0, 0.75 - 0.15 * order_idx)
            eval_score = target.score + bonus + sequence_bonus
            if best is None or eval_score > best_eval_score:
                best = target
                best_yaw = yaw
                best_label = label
                best_eval_score = eval_score
        if best is None:
            return None
        self._single_source_sweep_idx += 1
        best.metadata.update({"centroid": (cx, cy), "yaw": best_yaw, "label": best_label})
        self._record_if_coverage(ctx, best)
        return best

    def _source_set_expansion_target(
        self,
        ctx: SourceSeekContext,
        min_travel_d: float,
    ) -> Optional[StrategyTarget]:
        if len(ctx.found_sources) < self.config.source_set_expansion_min_sources or not self._map_available(ctx):
            return None
        cx, cy = self._sources_centroid(ctx)
        yaw_candidates: List[Tuple[float, float, float, str, float, float]] = []
        outward_yaw = math.atan2(cy - ctx.spawn_y, cx - ctx.spawn_x)
        for yaw in self._source_pair_lateral_yaws(ctx):
            yaw_candidates.append((
                yaw,
                self.config.source_set_lateral_directional_weight,
                self.config.source_set_lateral_bonus,
                "source_lateral",
                self.config.source_set_lateral_max_d,
                self.config.source_set_lateral_max_d,
            ))
        for yaw in self._source_set_gap_yaws(ctx, cx, cy):
            yaw_candidates.append((
                yaw,
                self.config.source_set_expansion_directional_weight,
                0.10,
                "source_gap",
                self.config.source_set_lateral_max_d,
                self.config.source_set_lateral_max_d,
            ))
        if math.hypot(cx - ctx.spawn_x, cy - ctx.spawn_y) > 1.0:
            yaw_candidates.append((
                outward_yaw,
                self.config.source_set_outward_directional_weight,
                self.config.source_set_outward_bonus,
                "outward",
                self.config.source_set_expansion_max_d,
                self.config.source_set_expansion_max_d,
            ))
        sector = self._map_sector_yaw(
            ctx,
            min_d=min_travel_d,
            max_d=self.config.source_set_expansion_max_d,
            safe_dist=max(self.config.survey_safe_dist, self._safe_dist()),
            phase_yaw=self._phase_yaw_for_coverage(ctx),
        )
        if sector is not None:
            yaw_candidates.append((
                sector.yaw,
                self.config.source_set_expansion_directional_weight,
                0.0,
                "map_sector",
                self.config.source_set_expansion_max_d,
                self.config.source_set_expansion_max_d,
            ))
        yaw_candidates.append((
            self._phase_yaw_for_coverage(ctx),
            self.config.source_set_expansion_directional_weight,
            0.0,
            "coverage_phase",
            self.config.source_set_expansion_max_d,
            self.config.source_set_expansion_max_d,
        ))
        if not yaw_candidates:
            return None
        start = self._source_set_sweep_idx % len(yaw_candidates)
        yaw_candidates = yaw_candidates[start:] + yaw_candidates[:start]

        best = None
        best_yaw = None
        best_label = "source_set"
        best_eval_score = -float("inf")
        priority = None
        seen = set()
        for order_idx, (yaw, directional_weight, bonus, label, max_travel, max_radius) in enumerate(yaw_candidates):
            key = round(wrap_angle(yaw), 2)
            if key in seen:
                continue
            seen.add(key)
            angle_span = math.radians(70.0) if label in ("source_lateral", "source_gap") else None
            target = self._map_coverage_ring(
                ctx,
                min_radius=max(self._safe_dist() + 1.0, self.config.coverage_ring_min_d),
                max_radius=max(self.config.coverage_ring_min_d, max_radius),
                safe_dist=max(self.config.survey_safe_dist, self._safe_dist()),
                preferred_yaw=yaw,
                directional_weight=directional_weight,
                anchor=(cx, cy),
                min_travel_d=min_travel_d,
                max_travel_d=max(min_travel_d, max_travel),
                angle_span_rad=angle_span,
            )
            if target is None:
                continue
            sequence_bonus = max(0.0, 0.65 - 0.13 * order_idx)
            eval_score = target.score + bonus + sequence_bonus
            candidate = target
            candidate.metadata.update({"centroid": (cx, cy), "yaw": yaw, "label": label})
            if priority is None and label in ("source_lateral", "source_gap"):
                priority = candidate
            if best is None or eval_score > best_eval_score:
                best = candidate
                best_yaw = yaw
                best_label = label
                best_eval_score = eval_score
        if priority is not None:
            result = priority
        else:
            result = best
            if result is not None:
                result.metadata.update({"centroid": (cx, cy), "yaw": best_yaw, "label": best_label})
        if result is None:
            return None
        self._source_set_sweep_idx += 1
        self._record_if_coverage(ctx, result)
        return result

    def _best_departure_ring(
        self,
        ctx: SourceSeekContext,
        cx: float,
        cy: float,
        min_travel_d: float,
    ) -> Optional[StrategyTarget]:
        if not self._map_available(ctx):
            return None
        max_d = max(self.config.pc_min_d + 2.0, self.config.departure_dist, self.config.coverage_ring_max_d)
        yaw_candidates: List[Tuple[float, float, float, str]] = []
        for phase, label in (
            (self._phase_yaw_for_coverage(ctx), "coverage_phase"),
            (self._phase_yaw_away_from_sources(ctx), "source_away"),
        ):
            sector = self._map_sector_yaw(
                ctx,
                min_d=min_travel_d,
                max_d=max_d,
                safe_dist=max(self.config.survey_safe_dist, self._safe_dist()),
                phase_yaw=phase,
            )
            if sector is not None:
                yaw_candidates.append((sector.yaw, self.config.departure_directional_weight, 0.0, f"{label}_sector"))
            yaw_candidates.append((phase, self.config.departure_directional_weight, 0.0, label))

        best = None
        best_yaw = None
        best_label = "departure"
        best_eval_score = -float("inf")
        seen = set()
        for yaw, directional_weight, bonus, label in yaw_candidates:
            key = round(wrap_angle(yaw), 2)
            if key in seen:
                continue
            seen.add(key)
            target = self._map_coverage_ring(
                ctx,
                min_radius=max(self._safe_dist() + 1.0, self.config.coverage_ring_min_d),
                max_radius=max_d,
                safe_dist=max(self.config.survey_safe_dist, self._safe_dist()),
                preferred_yaw=yaw,
                directional_weight=directional_weight,
                anchor=(cx, cy),
                min_travel_d=min_travel_d,
                max_travel_d=max_d,
            )
            if target is None:
                continue
            eval_score = target.score + bonus
            if best is None or eval_score > best_eval_score:
                best = target
                best_yaw = yaw
                best_label = label
                best_eval_score = eval_score
        if best is None:
            return None
        best.metadata.update({"centroid": (cx, cy), "yaw": best_yaw, "label": best_label})
        self._record_if_coverage(ctx, best)
        return best

    def _fallback_departure(
        self,
        ctx: SourceSeekContext,
        cx: float,
        cy: float,
        preferred_yaw: float,
    ) -> StrategyTarget:
        best_yaw = random.uniform(-math.pi, math.pi)
        best_score = -1.0
        for idx in range(24):
            yaw = -math.pi + (2.0 * math.pi / 24.0) * idx
            score = 0.0
            for frac in (0.4, 0.65, 0.85, 1.0):
                px = cx + self.config.departure_dist * frac * math.cos(yaw)
                py = cy + self.config.departure_dist * frac * math.sin(yaw)
                novelty = self._novelty_at(ctx, px, py)
                source_ok = 1.0
                if ctx.found_sources:
                    min_src_d = min(math.hypot(px - sx, py - sy) for sx, sy, _ in ctx.found_sources)
                    source_ok = 1.0 if min_src_d > self.config.departure_dist * 0.3 else 0.0
                align = 0.5 + 0.5 * math.cos(yaw - preferred_yaw)
                score += novelty * source_ok + 0.15 * align
            if score > best_score:
                best_score = score
                best_yaw = yaw
        return StrategyTarget(
            x=cx + self.config.departure_dist * math.cos(best_yaw),
            y=cy + self.config.departure_dist * math.sin(best_yaw),
            score=best_score,
            reason="fallback_departure",
            execution_hint=EXECUTION_DIRECT_FIRST,
            metadata={"centroid": (cx, cy), "yaw": best_yaw, "label": "fallback"},
        )

    def _novelty_at(self, ctx: SourceSeekContext, wx: float, wy: float) -> float:
        if self._map_available(ctx):
            m = ctx.thermal_map
            ix = int((wx - m["origin_x"]) / m["resolution"])
            iy = int((wy - m["origin_y"]) / m["resolution"])
            if 0 <= ix < m["width"] and 0 <= iy < m["height"]:
                visits = float(m["visit_count"][iy, ix])
                conf = float(m["confidence"][iy, ix])
                return 0.6 / (1.0 + visits) + 0.4 * (1.0 - max(0.0, min(1.0, conf)))
            return 0.0
        if ctx.belief_map is None:
            return 0.0
        ci, cj = ctx.belief_map._ci(wx, wy)
        return 1.0 / (1.0 + float(ctx.belief_map.visit[ci, cj]))
