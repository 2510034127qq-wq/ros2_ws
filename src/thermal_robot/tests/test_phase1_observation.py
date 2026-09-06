#!/usr/bin/env python3
"""Phase 1 pure-module tests: visibility, residual planning, stats."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parents[3]
ROBOT = WORKSPACE / "src" / "thermal_robot"
for rel in ("thermal_field_reconstructor", "thermal_motion_controller"):
    p = str(ROBOT / rel)
    if p not in sys.path:
        sys.path.insert(0, p)

from thermal_field_reconstructor import grid_geometry  # noqa: E402


def _load_script(name: str):
    path = ROBOT / "scripts" / f"{name}.py"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _make_occ(width=40, height=40, resolution=0.25, origin_x=-5.0, origin_y=-5.0):
    from thermal_field_reconstructor import visibility
    data = np.zeros((height, width), dtype=np.int16)
    return visibility.OccupancyView(origin_x, origin_y, resolution, data)


class TestGridGeometry:
    def test_constants(self):
        assert grid_geometry.GRID_CENTER_X == -6.0
        assert grid_geometry.GRID_CENTER_Y == 0.0
        assert grid_geometry.GRID_SIZE_M == 50.0

    def test_world_occupancy_uses_shared_constants(self):
        wo = _load_script("world_occupancy")
        assert wo.GRID_CENTER_X == grid_geometry.GRID_CENTER_X
        assert wo.GRID_CENTER_Y == grid_geometry.GRID_CENTER_Y
        assert wo.GRID_SIZE_M == grid_geometry.GRID_SIZE_M
        assert wo.GRID_RESOLUTION == 0.1

    def test_world_thermal_grid_uses_shared_constants(self):
        from thermal_field_reconstructor.thermal_mapping import WorldThermalGrid
        g = WorldThermalGrid()
        assert g.origin_x == grid_geometry.GRID_CENTER_X - grid_geometry.GRID_SIZE_M / 2.0
        assert g.origin_y == grid_geometry.GRID_CENTER_Y - grid_geometry.GRID_SIZE_M / 2.0

    def test_existing_npz_origin_matches(self):
        sample = ROBOT / "thermal_bringup" / "worlds" / "occupancy" / "thermal_scene_nav.npz"
        assert sample.exists()
        with np.load(sample) as f:
            assert float(f["origin_x"]) == pytest.approx(
                grid_geometry.GRID_CENTER_X - grid_geometry.GRID_SIZE_M / 2.0)


class TestObservationContract:
    def test_contract_fields_and_legacy_reexport(self):
        from thermal_field_reconstructor.observation import (
            MEASUREMENT_FIELD_DIRECT, SensorPose2D, TopDownRectProjector)
        from thermal_field_reconstructor.thermal_mapping import project_pixels_to_world

        img = np.full((48, 64), 25.0, dtype=np.float32)
        obs = TopDownRectProjector(fov_x=4.0, fov_y=3.0).project(
            img, SensorPose2D(x=1.0, y=2.0, yaw=0.5), stamp_s=12.5)
        assert obs.stamp_s == 12.5
        assert obs.sensor_pose.frame_id == "world"
        assert obs.measurement_type == MEASUREMENT_FIELD_DIRECT
        assert obs.sample_wx.size == 48 * 64
        assert np.all(obs.confidence == 1.0)
        wxs, wys = project_pixels_to_world(64, 48, 4.0, 3.0, 3.0, -1.0, 0.0)
        assert wxs.shape == (48, 64)
        assert float(wxs.mean()) == pytest.approx(3.0, abs=1e-5)
        assert float(wys.mean()) == pytest.approx(-1.0, abs=1e-5)

    def test_yaw_rotates_footprint_and_size_mismatch_raises(self):
        from thermal_field_reconstructor.observation import (
            MEASUREMENT_FIELD_DIRECT, SensorPose2D, ThermalObservation, TopDownRectProjector)
        img = np.zeros((48, 64), dtype=np.float32)
        obs = TopDownRectProjector(fov_x=4.0, fov_y=3.0).project(
            img, SensorPose2D(x=0.0, y=0.0, yaw=math.pi / 2.0), 0.0)
        assert float(obs.sample_wy.max()) == pytest.approx(2.0, abs=1e-4)
        assert float(obs.sample_wx.max()) == pytest.approx(1.5, abs=1e-4)
        with pytest.raises(ValueError):
            ThermalObservation(
                stamp_s=0.0,
                sensor_pose=SensorPose2D(0.0, 0.0, 0.0),
                measurement_type=MEASUREMENT_FIELD_DIRECT,
                sample_wx=np.zeros(3, np.float32),
                sample_wy=np.zeros(3, np.float32),
                temperature=np.zeros(4, np.float32),
                confidence=np.ones(3, np.float32),
            )


class TestVisibility:
    def test_occupancy_grid_shape_validation(self):
        from thermal_field_reconstructor import visibility

        assert visibility.valid_occupancy_grid_shape([0] * 12, 4, 3)
        assert not visibility.valid_occupancy_grid_shape([], 0, 0)
        assert not visibility.valid_occupancy_grid_shape([0] * 11, 4, 3)
        assert visibility.occupancy_grid_has_known_cells([0, -1], 2, 1)
        assert not visibility.occupancy_grid_has_known_cells([-1, -1], 2, 1)

    def test_empty_map_all_visible_and_wall_blocks(self):
        from thermal_field_reconstructor import visibility
        occ = _make_occ()
        assert visibility.visible_mask(
            occ, 0.0, 0.0, np.array([3.0, -2.0]), np.array([3.0, 4.0])).tolist() == [True, True]
        occ.data[:, 28] = 100
        assert visibility.visible_mask(
            occ, 0.0, 0.0, np.array([4.0, 1.0]), np.array([0.0, 0.0])).tolist() == [False, True]
        assert visibility.visible_mask(
            occ, 0.0, 0.0, np.array([1.95]), np.array([0.0]), step_m=0.1).tolist() == [True]

    def test_unknown_out_of_bounds_occupied_and_reachable(self):
        from thermal_field_reconstructor import visibility
        occ = _make_occ()
        occ.data[:, :] = -1
        assert visibility.visible_mask(occ, 0.0, 0.0, np.array([20.0]), np.array([20.0])).tolist() == [True]
        occ = _make_occ()
        occ.data[20, 20] = 100
        out = visibility.occupied_at(
            occ, np.array([0.1, 1.0, 99.0]), np.array([0.1, 1.0, 99.0]))
        assert out.tolist() == [True, False, False]
        occ = _make_occ()
        occ.data[:, 28] = 100
        assert visibility.line_reachable(occ, 0.0, 0.0, 1.0, 0.0)
        assert not visibility.line_reachable(occ, 0.0, 0.0, 4.0, 0.0)
        assert visibility.line_reachable(None, 0.0, 0.0, 4.0, 0.0)

    def test_speed_budget(self):
        # 200 条 2.5m 射线必须远低于算力预算 (粗略上限 50ms)
        import time as _t
        from thermal_field_reconstructor import visibility
        occ = visibility.OccupancyView(
            -5.0, -5.0, 0.05, np.zeros((200, 200), dtype=np.int16))
        ang = np.linspace(0, 2 * math.pi, 200)
        t0 = _t.monotonic()
        visibility.visible_mask(occ, 0.0, 0.0,
                                2.5 * np.cos(ang), 2.5 * np.sin(ang), step_m=0.1)
        assert (_t.monotonic() - t0) < 0.05

    def test_strict_reachability_rejects_unknown_and_occupied_endpoint(self):
        from thermal_field_reconstructor import visibility

        occ = _make_occ(width=40, height=40, origin_x=-5.0, origin_y=-5.0)
        occ.data[:, :] = -1
        occ.data[18:23, 18:27] = 0
        assert visibility.line_reachable_known_free(occ, 0.0, 0.0, 1.0, 0.0)
        assert not visibility.line_reachable_known_free(occ, 0.0, 0.0, 3.0, 0.0)
        occ.data[20, 24] = 100
        assert not visibility.line_reachable_known_free(occ, 0.0, 0.0, 1.0, 0.0)

    def test_plannable_reachability_allows_unknown_path_but_requires_free_endpoint(self):
        from thermal_field_reconstructor import visibility

        occ = _make_occ(width=40, height=40, origin_x=-5.0, origin_y=-5.0)
        occ.data[:, :] = -1
        occ.data[20, 20] = 0
        occ.data[20, 24] = 0
        assert visibility.line_reachable_plannable(occ, 0.0, 0.0, 1.0, 0.0)
        assert not visibility.line_reachable_known_free(occ, 0.0, 0.0, 1.0, 0.0)
        occ.data[20, 24] = 100
        assert not visibility.line_reachable_plannable(occ, 0.0, 0.0, 1.0, 0.0)

    def test_reachability_cannot_skip_single_slam_cell(self):
        from thermal_field_reconstructor import visibility

        occ = visibility.OccupancyView(
            origin_x=0.0,
            origin_y=0.0,
            resolution=0.05,
            data=np.zeros((4, 24), dtype=np.int16),
        )
        y = 0.075
        x0, x1 = 0.025, 1.025
        occ.data[1, 10] = 100
        assert not visibility.line_reachable(occ, x0, y, x1, y)
        assert not visibility.line_reachable_plannable(occ, x0, y, x1, y)

        occ.data[1, 10] = -1
        assert visibility.line_reachable_plannable(occ, x0, y, x1, y)
        assert not visibility.line_reachable_known_free(occ, x0, y, x1, y)

    def test_corner_crossing_is_conservative_for_strict_motion(self):
        from thermal_field_reconstructor import visibility

        occ = visibility.OccupancyView(
            origin_x=0.0,
            origin_y=0.0,
            resolution=0.05,
            data=np.zeros((8, 8), dtype=np.int16),
        )
        occ.data[1, 2] = 100
        assert not visibility.line_reachable_known_free(
            occ, 0.075, 0.075, 0.275, 0.275)


class TestThreeStateFusion:
    def _grid(self):
        from thermal_field_reconstructor.thermal_mapping import WorldThermalGrid
        return WorldThermalGrid(center_x=0.0, center_y=0.0,
                                size_x_m=20.0, size_y_m=20.0, resolution=0.25)

    def _obs(self, temp=30.0, x=0.0, y=0.0, yaw=0.0, stamp=1.0):
        from thermal_field_reconstructor.observation import SensorPose2D, TopDownRectProjector
        img = np.full((48, 64), float(temp), dtype=np.float32)
        return TopDownRectProjector().project(
            img, SensorPose2D(x=float(x), y=float(y), yaw=float(yaw)), stamp)

    def test_no_occupancy_matches_legacy_integrate_image(self):
        from thermal_field_reconstructor.observation import SensorPose2D, TopDownRectProjector
        a, b = self._grid(), self._grid()
        img = np.random.default_rng(7).normal(25.0, 3.0, (48, 64)).astype(np.float32)
        a.integrate_image(img, 1.0, 0.5, 0.3, stamp_s=2.0)
        obs = TopDownRectProjector().project(img, SensorPose2D(1.0, 0.5, 0.3), 2.0)
        b.integrate_observation(obs)
        np.testing.assert_allclose(a.mean, b.mean, rtol=1e-6)
        np.testing.assert_array_equal(a.visit_count, b.visit_count)

    def test_view_state_transitions_and_sectors(self):
        from thermal_field_reconstructor import visibility
        from thermal_field_reconstructor.thermal_mapping import (
            N_VIEW_SECTORS, VIEW_BLOCKED_ONLY, VIEW_CLEAR, VIEW_NEVER)
        g = self._grid()
        occ = visibility.OccupancyView(-10.0, -10.0, 0.25,
                                       np.zeros((80, 80), dtype=np.int16))
        occ.data[:, 44] = 100
        g.integrate_observation(self._obs(30.0, 0.0, 0.0, 0.0), occupancy=occ)
        snap = g.snapshot(now_s=2.0)
        ix_blocked = int((1.8 - g.origin_x) / g.resolution)
        ix_clear = int((0.0 - g.origin_x) / g.resolution)
        iy_mid = int((0.0 - g.origin_y) / g.resolution)
        assert snap.view_state[iy_mid, ix_blocked] == VIEW_BLOCKED_ONLY
        assert snap.view_state[iy_mid, ix_clear] == VIEW_CLEAR
        assert snap.view_state[iy_mid, int((8.0 - g.origin_x) / g.resolution)] == VIEW_NEVER
        assert g.visit_count[iy_mid, ix_blocked] == 0
        assert g.blocked_count[iy_mid, ix_blocked] >= 1
        occ.data[:, 44] = 0
        g.integrate_observation(self._obs(30.0, 0.0, 0.0, 0.0), occupancy=occ)
        assert g.snapshot(now_s=3.0).view_state[iy_mid, ix_blocked] == VIEW_CLEAR
        assert N_VIEW_SECTORS == 8
        g2 = self._grid()
        g2.integrate_observation(self._obs(25.0, 1.5, 0.0, 0.0))
        east_bits = int(g2.view_sectors[iy_mid, ix_clear])
        g2.integrate_observation(self._obs(25.0, -1.5, 0.0, 0.0))
        both_bits = int(g2.view_sectors[iy_mid, ix_clear])
        assert east_bits != 0
        assert bin(both_bits).count("1") > bin(east_bits).count("1")


class TestResidual:
    def _meta(self):
        return dict(width=40, height=40, resolution=0.25, origin_x=-5.0, origin_y=-5.0)

    def test_predict_and_residual_semantics(self):
        from thermal_field_reconstructor import residual
        from thermal_field_reconstructor.thermal_mapping import VIEW_CLEAR, VIEW_NEVER
        m = self._meta()
        field = residual.predict_field(ambient_temp=22.0, sources=[(0.0, 0.0, 10.0, 1.0)], **m)
        iy = int((0.0 - m["origin_y"]) / m["resolution"])
        ix = int((0.0 - m["origin_x"]) / m["resolution"])
        assert field[iy, ix] == pytest.approx(32.0, abs=0.5)
        view = np.full((40, 40), VIEW_CLEAR, dtype=np.uint8)
        assert float(np.abs(residual.residual_field(field, field, view)).max()) == 0.0
        predicted = residual.predict_field(ambient_temp=22.0, sources=[], **m)
        r = residual.residual_field(field, predicted, view)
        assert r[iy, ix] > 8.0
        view[:, :] = VIEW_NEVER
        assert float(np.abs(residual.residual_field(field, predicted, view)).max()) == 0.0


class TestResidualTarget:
    def _args(self, **over):
        from thermal_field_reconstructor.thermal_mapping import VIEW_NEVER
        h = w = 60
        base = dict(
            robot_wx=0.0, robot_wy=0.0, width=w, height=h, resolution=0.25,
            origin_x=-7.5, origin_y=-7.5,
            residual=np.zeros((h, w), dtype=np.float32),
            view_state=np.full((h, w), VIEW_NEVER, dtype=np.uint8),
            last_seen_age_s=np.full((h, w), -1.0, dtype=np.float32),
        )
        base.update(over)
        return base

    def test_residual_mass_unseen_and_reachability(self):
        from thermal_motion_controller.planning import select_residual_target
        from thermal_field_reconstructor.thermal_mapping import VIEW_CLEAR, VIEW_NEVER
        a = self._args()
        a["view_state"][:, :] = VIEW_CLEAR
        a["last_seen_age_s"][:, :] = 1.0
        iy = int((0.0 + 7.5) / 0.25)
        ix = int((5.0 + 7.5) / 0.25)
        a["residual"][iy - 2:iy + 3, ix - 2:ix + 3] = 6.0
        t = select_residual_target(**a)
        assert t is not None
        assert math.hypot(t.x - 5.0, t.y) < 1.5
        assert t.reason == "residual_mass"
        b = self._args()
        b["view_state"][:, :] = VIEW_CLEAR
        b["view_state"][:, 40:] = VIEW_NEVER
        b["line_reachable_fn"] = lambda x0, y0, x1, y1: x1 <= 0.0
        t2 = select_residual_target(**b)
        assert t2 is not None
        assert hasattr(t2, "metadata_reachable")

    def test_cold_noise_cannot_outscore_unseen_coverage(self):
        from thermal_motion_controller.planning import select_residual_target
        from thermal_field_reconstructor.thermal_mapping import VIEW_CLEAR, VIEW_NEVER

        args = self._args()
        args["view_state"][:, :] = VIEW_CLEAR
        args["last_seen_age_s"][:, :] = 1.0
        args["residual"][28:33, 48:53] = 0.05
        args["view_state"][18:42, 2:18] = VIEW_NEVER
        target = select_residual_target(
            **args, residual_floor=1.5, footprint_radius=2.0)
        assert target is not None
        assert target.reason == "unseen"
        assert target.x < 0.0

    def test_footprint_gain_prefers_region_over_isolated_unseen_cell(self):
        from thermal_motion_controller.planning import select_residual_target
        from thermal_field_reconstructor.thermal_mapping import VIEW_CLEAR, VIEW_NEVER

        args = self._args()
        args["view_state"][:, :] = VIEW_CLEAR
        args["last_seen_age_s"][:, :] = 1.0
        args["view_state"][30, 43] = VIEW_NEVER
        args["view_state"][15:31, 3:19] = VIEW_NEVER
        target = select_residual_target(**args, footprint_radius=2.0)
        assert target is not None
        assert target.x < 0.0
        assert target.reason == "unseen"

    def test_unreachable_candidates_return_none(self):
        from thermal_motion_controller.planning import select_residual_target

        args = self._args(line_reachable_fn=lambda *_args: False)
        target = select_residual_target(**args, top_k=32)
        assert target is None

    def test_candidate_mask_filters_unknown_or_occupied_endpoints_before_ranking(self):
        from thermal_motion_controller.planning import select_residual_target

        args = self._args()
        candidate_mask = np.zeros((args["height"], args["width"]), dtype=bool)
        candidate_mask[20:40, 30:45] = True
        target = select_residual_target(**args, candidate_valid_mask=candidate_mask)
        assert target is not None
        iy = int((target.y - args["origin_y"]) / args["resolution"])
        ix = int((target.x - args["origin_x"]) / args["resolution"])
        assert candidate_mask[iy, ix]

    def test_full_grid_planning_speed_budget(self):
        import time as _time
        from thermal_motion_controller.planning import select_residual_target

        h = w = 200
        start = _time.monotonic()
        target = select_residual_target(
            robot_wx=0.0, robot_wy=0.0,
            width=w, height=h, resolution=0.25,
            origin_x=-25.0, origin_y=-25.0,
            residual=np.zeros((h, w), dtype=np.float32),
            view_state=np.zeros((h, w), dtype=np.uint8),
            last_seen_age_s=np.full((h, w), -1.0, dtype=np.float32),
            min_d=5.0, max_d=14.0, top_k=64,
            line_reachable_fn=lambda *_args: True,
        )
        elapsed_ms = (_time.monotonic() - start) * 1000.0
        assert target is not None
        assert elapsed_ms < 100.0


class TestWaypointDistanceContract:
    def test_operational_minimum_exceeds_arrival_radius(self):
        from thermal_motion_controller.planning import operational_waypoint_min_distance

        assert operational_waypoint_min_distance(
            configured_min_d=2.5,
            arrival_radius=2.5,
            resolution=0.25,
        ) == pytest.approx(3.0)

    def test_configured_minimum_remains_authoritative_when_safe(self):
        from thermal_motion_controller.planning import operational_waypoint_min_distance

        assert operational_waypoint_min_distance(
            configured_min_d=5.0,
            arrival_radius=2.5,
            resolution=0.25,
        ) == pytest.approx(5.0)

    def test_grid_resolution_sets_a_strict_quantization_margin(self):
        from thermal_motion_controller.planning import operational_waypoint_min_distance

        assert operational_waypoint_min_distance(
            configured_min_d=2.6,
            arrival_radius=2.5,
            resolution=1.0,
        ) == pytest.approx(4.5)

    def test_controller_wires_operational_bounds_into_residual_planner(self):
        controller = (ROBOT / "thermal_motion_controller" / "thermal_motion_controller"
                      / "controller_node.py").read_text()
        assert "operational_waypoint_min_distance" in controller
        assert "configured_min_d=self._residual_waypoint_min_d" in controller
        assert "min_d=effective_min_d" in controller
        assert "max_d=self._survey_wp_max_d" in controller
        assert "RESIDUAL_TARGET_REJECT/too_close" in controller
        assert "line_reachable_plannable" in controller
        assert "candidate_valid_mask=candidate_valid_mask" in controller
        assert "residual_floor=self._residual_planner_min_evidence" in controller
        assert "footprint_radius=self._residual_planner_footprint_radius" in controller
        assert "top_k=self._residual_planner_top_k" in controller
        assert "[RESIDUAL_PLAN]" in controller

    def test_residual_planner_parameters_live_in_yaml(self):
        import yaml

        params_path = ROBOT / "thermal_bringup" / "config" / "params.yaml"
        params = yaml.safe_load(params_path.read_text())["controller_node"]["ros__parameters"]
        assert params["residual_planner_min_evidence"] > 0.0
        assert params["residual_planner_footprint_radius"] > 0.0
        assert params["residual_planner_top_k"] >= 16
        assert params["residual_waypoint_min_d"] >= 3.0


class TestNav2GoalLifecycle:
    def test_goal_decision_rate_limits_every_state(self):
        from thermal_motion_controller.navigation_policy import (
            GOAL_KEEP, GOAL_REPLACE, GOAL_SEND, GOAL_WAIT, decide_goal_action)

        current = (1.0, 1.0)
        assert decide_goal_action(
            "active", current, current, now=2.0, last_send_t=1.0,
            min_interval_s=3.0) == GOAL_KEEP
        assert decide_goal_action(
            "active", current, (5.0, 1.0), now=2.0, last_send_t=1.0,
            min_interval_s=3.0) == GOAL_WAIT
        assert decide_goal_action(
            "active", current, (5.0, 1.0), now=4.1, last_send_t=1.0,
            min_interval_s=3.0) == GOAL_REPLACE
        assert decide_goal_action(
            "idle", None, (5.0, 1.0), now=2.0, last_send_t=1.0,
            min_interval_s=3.0) == GOAL_WAIT
        assert decide_goal_action(
            "idle", None, (5.0, 1.0), now=4.1, last_send_t=1.0,
            min_interval_s=3.0) == GOAL_SEND

    def test_controller_invalidates_stale_goal_callbacks(self):
        controller = (ROBOT / "thermal_motion_controller" / "thermal_motion_controller"
                      / "controller_node.py").read_text()
        assert "decide_goal_action" in controller
        assert "_nav2_goal_generation" in controller
        assert "generation != self._nav2_goal_generation" in controller
        assert "[NAV2_STALE_CANCEL]" in controller
        assert "nav2_goal_min_interval_s" in controller

    def test_nav2_goal_interval_lives_in_yaml(self):
        import yaml

        params_path = ROBOT / "thermal_bringup" / "config" / "params.yaml"
        params = yaml.safe_load(params_path.read_text())["controller_node"]["ros__parameters"]
        assert params["nav2_goal_min_interval_s"] >= 3.0


class TestDirectMotionGuard:
    def test_forward_clearance_uses_only_forward_cone_and_open_range(self):
        from thermal_motion_controller.navigation_policy import forward_clearance

        ranges = [0.20, 4.0, float("inf"), 3.0, 0.25]
        clearance = forward_clearance(
            ranges,
            angle_min=-math.pi,
            angle_increment=math.pi / 2.0,
            range_min=0.10,
            range_max=12.0,
            half_angle_rad=math.radians(20.0),
        )
        assert clearance == pytest.approx(12.0)

    def test_forward_clearance_rejects_invalid_or_missing_forward_samples(self):
        from thermal_motion_controller.navigation_policy import forward_clearance

        assert forward_clearance(
            [float("nan"), 0.01, float("nan")],
            angle_min=-0.2,
            angle_increment=0.2,
            range_min=0.10,
            range_max=12.0,
            half_angle_rad=0.3,
        ) is None
        assert forward_clearance(
            [2.0, 3.0],
            angle_min=1.0,
            angle_increment=0.2,
            range_min=0.10,
            range_max=12.0,
            half_angle_rad=0.3,
        ) is None

    def test_direct_motion_requires_fresh_clear_scan(self):
        from thermal_motion_controller.navigation_policy import (
            DIRECT_KNOWN_FREE,
            DIRECT_SCAN_GUARDED,
            DIRECT_STOP,
            decide_direct_motion,
        )

        assert decide_direct_motion(
            path_known_free=True,
            scan_clearance_m=2.0,
            scan_age_s=0.1,
            stop_distance_m=0.65,
            scan_stale_s=0.6,
        ) == DIRECT_KNOWN_FREE
        assert decide_direct_motion(
            path_known_free=False,
            scan_clearance_m=2.0,
            scan_age_s=0.1,
            stop_distance_m=0.65,
            scan_stale_s=0.6,
        ) == DIRECT_SCAN_GUARDED
        assert decide_direct_motion(
            path_known_free=True,
            scan_clearance_m=0.60,
            scan_age_s=0.1,
            stop_distance_m=0.65,
            scan_stale_s=0.6,
        ) == DIRECT_STOP
        assert decide_direct_motion(
            path_known_free=True,
            scan_clearance_m=2.0,
            scan_age_s=0.7,
            stop_distance_m=0.65,
            scan_stale_s=0.6,
        ) == DIRECT_STOP
        assert decide_direct_motion(
            path_known_free=True,
            scan_clearance_m=None,
            scan_age_s=0.1,
            stop_distance_m=0.65,
            scan_stale_s=0.6,
        ) == DIRECT_STOP

    def test_controller_wires_scan_guard_into_all_coarse_direct_paths(self):
        controller = (ROBOT / "thermal_motion_controller" / "thermal_motion_controller"
                      / "controller_node.py").read_text()
        package_xml = (ROBOT / "thermal_motion_controller" / "package.xml").read_text()

        assert "from sensor_msgs.msg import LaserScan" in controller
        assert "create_subscription(LaserScan" in controller
        assert "'/scan'" in controller
        assert "def _direct_guarded_cmd" in controller
        assert "decide_direct_motion" in controller
        assert "line_reachable_known_free" in controller
        assert "'direct_safe'" in controller
        assert controller.count("_direct_guarded_cmd(") >= 4
        assert "<exec_depend>sensor_msgs</exec_depend>" in package_xml

    def test_direct_guard_parameters_live_in_yaml(self):
        import yaml

        params_path = ROBOT / "thermal_bringup" / "config" / "params.yaml"
        params = yaml.safe_load(params_path.read_text())["controller_node"]["ros__parameters"]
        assert params["direct_scan_stop_m"] >= 0.5
        assert 0.0 < params["direct_scan_stale_s"] <= 1.0
        assert 10.0 <= params["direct_scan_half_angle_deg"] <= 60.0




class TestResidualStrategyWiring:
    def test_text_wiring(self):
        controller = (ROBOT / "thermal_motion_controller" / "thermal_motion_controller"
                      / "controller_node.py").read_text()
        runner = (ROBOT / "scripts" / "run_multiscenario_matrix.py").read_text()
        collector = (ROBOT / "scripts" / "collect_sim_data.py").read_text()
        assert "'residual'" in controller
        assert "select_residual_target" in controller
        assert '"residual"' in runner
        mapper = (ROBOT / "thermal_field_reconstructor" / "thermal_field_reconstructor"
                  / "thermal_mapper_node.py").read_text()
        assert "occupancy_grid_has_known_cells" in controller
        assert "occupancy_grid_has_known_cells" in mapper


class TestPairedStats:
    def test_paired_permutation_and_bootstrap(self):
        ms = _load_script("matrix_stats")
        assert ms.paired_permutation_test([2, 2, 2], [1, 1, 1]) == pytest.approx(0.25)
        assert ms.paired_permutation_test([1, 2, 3, 4, 5], [0, 0, 0, 0, 0]) == pytest.approx(2.0 / 32.0)
        assert ms.paired_permutation_test([1, 1], [1, 1]) == 1.0
        point, lo, hi = ms.cluster_bootstrap_ratio_ci([3, 3, 3, 3], [3, 3, 3, 3], n_boot=500, seed=1)
        assert point == 1.0 and lo == 1.0 and hi == 1.0
        point, lo, hi = ms.cluster_bootstrap_ratio_ci([3, 2, 4, 1], [4, 3, 4, 2], n_boot=1000, seed=7)
        assert lo <= point <= hi


class TestMatrixCompare:
    def test_compare_two_roots(self, tmp_path):
        mc = _load_script("matrix_compare")
        for label, recall in (("base", 0.4), ("cand", 0.8)):
            root = tmp_path / label
            runs = []
            for case in ("w__a", "w__b"):
                for seed in (101, 102, 103, 104, 105):
                    run_dir = root / case / f"seed{seed}"
                    run_dir.mkdir(parents=True)
                    (run_dir / "source_summary.json").write_text(json.dumps({
                        "confirmed_count": 2,
                        "localization_errors_m": [
                            {"truth_id": "S1", "error_m": 0.5},
                            {"truth_id": "S2", "error_m": 0.4}],
                    }))
                    (run_dir / "attribution.json").write_text(json.dumps({
                        "n_truth_sources": 3, "n_matched": 2,
                        "failure_counts": {"not_reached": 1},
                        "per_source": {}}))
                    runs.append({"name": case, "seed": seed, "passed": True,
                                 "source_recall": recall,
                                 "source_precision": 1.0,
                                 "duplicate_confirmations": 0,
                                 "time_to_first_source": 5.0,
                                 "path_length_m": 30.0,
                                 "missing_counts": []})
            (root / "matrix_summary.json").write_text(json.dumps({
                "n_runs": len(runs), "n_passed": len(runs), "runs": runs}))
        report = mc.compare_roots(tmp_path / "base", tmp_path / "cand")
        assert report["n_pairs"] == 10
        assert report["metrics"]["source_recall"]["p_value"] < 0.01
        md = mc.render_report(report, "base", "cand")
        assert "source_recall" in md and "precision" in md
