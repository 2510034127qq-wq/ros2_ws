#!/usr/bin/env python3
"""Phase 1 pure-module tests: visibility, residual planning, clearance, stats."""

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


class TestClearance:
    def _params(self, **over):
        from thermal_motion_controller.clearance import ClearanceParams
        kw = dict(source_rate_per_m2=0.01, p_detect_per_sector=0.7,
                  max_effective_sectors=4, residual_block_thresh=1.5,
                  epsilon=0.05)
        kw.update(over)
        return ClearanceParams(**kw)

    def test_clearance_probability_and_ece(self):
        from thermal_motion_controller import clearance
        from thermal_motion_controller.clearance import VIEW_CLEAR
        vs = np.zeros((40, 40), dtype=np.uint8)
        sec = np.zeros((40, 40), dtype=np.uint8)
        assert clearance.clearance_probability(vs, sec, 0.0625, self._params()) == pytest.approx(math.exp(-1.0))
        vs[:, :] = VIEW_CLEAR
        sec[:, :] = 0b00001111
        base = clearance.clearance_probability(vs, sec, 0.0625, self._params())
        assert base == pytest.approx(math.exp(-0.0081), abs=1e-4)
        resid = np.zeros((40, 40), dtype=np.float32)
        resid[10:14, 10:14] = 5.0
        assert clearance.clearance_probability(vs, sec, 0.0625, self._params(), residual=resid) < base
        claimed = np.array([0.9] * 10)
        assert clearance.expected_calibration_error(claimed, np.array([1] * 9 + [0])) == pytest.approx(0.0)
        assert clearance.expected_calibration_error(claimed, np.array([1] * 5 + [0] * 5)) == pytest.approx(0.4)

    def test_monte_carlo_calibration(self):
        """生成式自洽: 按模型采样世界与覆盖, 宣称概率应校准 (ECE < 0.05).

        这是 spec §3 清场概率的验收机制(校准而非阈值), 必须常驻回归。
        """
        from thermal_motion_controller import clearance
        from thermal_motion_controller.clearance import VIEW_CLEAR
        rng = np.random.default_rng(42)
        params = self._params(source_rate_per_m2=0.02)
        h = w = 20
        area = 0.25
        claimed, outcomes = [], []
        for _ in range(1500):
            vs = np.zeros((h, w), dtype=np.uint8)
            sec = np.zeros((h, w), dtype=np.uint8)
            covered = rng.random((h, w)) < rng.uniform(0.2, 0.95)
            vs[covered] = VIEW_CLEAR
            n_sec = rng.integers(1, 5, size=(h, w))
            sec[covered] = (np.left_shift(1, n_sec) - 1).astype(np.uint8)[covered]
            p_miss = clearance.cell_miss_prob(vs, sec, params)
            lam_cells = params.source_rate_per_m2 * area * p_miss
            n_undet = rng.poisson(lam_cells).sum()
            claimed.append(clearance.clearance_probability(vs, sec, area, params))
            outcomes.append(1 if n_undet == 0 else 0)
        ece = clearance.expected_calibration_error(
            np.array(claimed), np.array(outcomes), n_bins=10)
        assert ece < 0.05


class TestResidualStrategyWiring:
    def test_text_wiring(self):
        controller = (ROBOT / "thermal_motion_controller" / "thermal_motion_controller"
                      / "controller_node.py").read_text()
        runner = (ROBOT / "scripts" / "run_multiscenario_matrix.py").read_text()
        collector = (ROBOT / "scripts" / "collect_sim_data.py").read_text()
        assert "'residual'" in controller
        assert "select_residual_target" in controller
        assert "/thermal/clearance" in controller
        assert '"residual"' in runner
        assert "/thermal/clearance" in collector


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
