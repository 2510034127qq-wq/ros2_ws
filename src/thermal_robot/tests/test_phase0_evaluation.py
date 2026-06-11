#!/usr/bin/env python3
"""Phase 0 evaluation foundation unit tests with no ROS runtime dependency."""

import importlib.util
import math
import sys
import unittest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
SCRIPTS = WORKSPACE / 'src/thermal_robot/scripts'

for rel in [
    'src/thermal_robot/thermal_sensor_sim',
]:
    path = str(WORKSPACE / rel)
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestMatrixStats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ms = _load_script('matrix_stats')

    def test_rank_with_ties(self):
        self.assertEqual(self.ms._rank([1.0, 1.0, 2.0]), [1.5, 1.5, 3.0])

    def test_aggregate_case_runs_mean_std(self):
        runs = [
            {'passed': True, 'source_recall': 0.6, 'source_precision': 1.0,
             'duplicate_confirmations': 0, 'time_to_first_source': 30.0, 'path_length_m': 30.0},
            {'passed': True, 'source_recall': 0.4, 'source_precision': 1.0,
             'duplicate_confirmations': 0, 'time_to_first_source': 50.0, 'path_length_m': 50.0},
        ]
        agg = self.ms.aggregate_case_runs(runs)
        self.assertEqual(agg['n_runs'], 2)
        self.assertEqual(agg['n_passed'], 2)
        recall = agg['metrics']['source_recall']
        self.assertAlmostEqual(recall['mean'], 0.5, places=6)
        self.assertAlmostEqual(recall['std'], 0.1414, places=3)
        self.assertEqual(recall['n'], 2)

    def test_aggregate_skips_none_metric(self):
        runs = [{'passed': False, 'source_recall': None, 'source_precision': None,
                 'duplicate_confirmations': None, 'time_to_first_source': None,
                 'path_length_m': None}]
        agg = self.ms.aggregate_case_runs(runs)
        self.assertIsNone(agg['metrics']['source_recall'])

    def test_mann_whitney_exact_separated(self):
        result = self.ms.mann_whitney_u([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        self.assertEqual(result['method'], 'exact')
        self.assertAlmostEqual(result['u'], 0.0, places=6)
        self.assertAlmostEqual(result['p_value'], 0.1, places=6)

    def test_mann_whitney_identical_groups_high_p(self):
        result = self.ms.mann_whitney_u([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertGreater(result['p_value'], 0.5)

    def test_mann_whitney_empty_input(self):
        result = self.ms.mann_whitney_u([], [1.0])
        self.assertIsNone(result['p_value'])

    def test_render_markdown_report_contains_cases(self):
        summary = {
            'config': {'preset': 'phase0', 'strategy': 'full', 'seeds': [101, 102],
                       'duration_s': 120.0, 'warmup_s': 36.0},
            'cases': {
                'open__static2': {
                    'n_runs': 2, 'n_passed': 2,
                    'metrics': {'source_recall': {'mean': 1.0, 'std': 0.0, 'min': 1.0,
                                                  'max': 1.0, 'n': 2, 'values': [1.0, 1.0]},
                                'source_precision': {'mean': 1.0, 'std': 0.0, 'min': 1.0,
                                                     'max': 1.0, 'n': 2, 'values': [1.0, 1.0]},
                                'duplicate_confirmations': {'mean': 0.0, 'std': 0.0, 'min': 0.0,
                                                            'max': 0.0, 'n': 2, 'values': [0.0, 0.0]},
                                'time_to_first_source': None,
                                'path_length_m': None},
                    'failure_counts': {'not_reached': 0, 'occluded': 0,
                                       'timing_missed': 0, 'not_confirmed': 0},
                },
            },
        }
        text = self.ms.render_markdown_report(summary)
        self.assertIn('open__static2', text)
        self.assertIn('1.000', text)
        self.assertIn('not_reached', text)


SAMPLE_SDF = """<?xml version="1.0"?>
<sdf version="1.6">
  <world name="test_world">
    <include><uri>model://sun</uri></include>
    <model name="visual_only_marker">
      <static>true</static>
      <pose>-6.0 0 0 0 0 0</pose>
      <link name="link">
        <visual name="pad"><geometry><box><size>0.6 0.6 0.04</size></box></geometry></visual>
      </link>
    </model>
    <model name="wall_east">
      <static>true</static>
      <pose>2.0 0.0 0.45 0 0 0</pose>
      <link name="link">
        <collision name="collision"><geometry><box><size>2.0 1.0 0.9</size></box></geometry></collision>
      </link>
    </model>
    <model name="wall_rotated">
      <static>true</static>
      <pose>0.0 5.0 0.45 0 0 1.5707963</pose>
      <link name="link">
        <collision name="collision"><geometry><box><size>4.0 0.4 0.9</size></box></geometry></collision>
      </link>
    </model>
  </world>
</sdf>
"""


class TestWorldOccupancy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wo = _load_script('world_occupancy')

    def test_parse_boxes_skips_visual_only(self):
        boxes = self.wo.parse_world_boxes(SAMPLE_SDF)
        names = sorted(b.name for b in boxes)
        self.assertEqual(names, ['wall_east', 'wall_rotated'])

    def test_parse_box_pose_and_size(self):
        boxes = {b.name: b for b in self.wo.parse_world_boxes(SAMPLE_SDF)}
        wall = boxes['wall_east']
        self.assertAlmostEqual(wall.x, 2.0)
        self.assertAlmostEqual(wall.y, 0.0)
        self.assertAlmostEqual(wall.size_x, 2.0)
        self.assertAlmostEqual(wall.size_y, 1.0)

    def test_rasterize_marks_box_cells(self):
        boxes = self.wo.parse_world_boxes(SAMPLE_SDF)
        grid = self.wo.rasterize(boxes)
        self.assertTrue(self.wo.is_occupied(grid, 2.0, 0.0))
        self.assertFalse(self.wo.is_occupied(grid, -3.0, 0.0))

    def test_rasterize_respects_yaw(self):
        boxes = self.wo.parse_world_boxes(SAMPLE_SDF)
        grid = self.wo.rasterize(boxes)
        self.assertTrue(self.wo.is_occupied(grid, 0.0, 6.5))
        self.assertFalse(self.wo.is_occupied(grid, 1.5, 6.5))

    def test_line_of_sight_blocked_and_clear(self):
        boxes = self.wo.parse_world_boxes(SAMPLE_SDF)
        grid = self.wo.rasterize(boxes)
        self.assertFalse(self.wo.line_of_sight(grid, -1.0, 0.0, 5.0, 0.0))
        self.assertTrue(self.wo.line_of_sight(grid, -1.0, 2.0, 5.0, 2.0))

    def test_npz_roundtrip(self):
        import tempfile
        boxes = self.wo.parse_world_boxes(SAMPLE_SDF)
        grid = self.wo.rasterize(boxes)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'g.npz'
            self.wo.save_grid(grid, path)
            loaded = self.wo.load_grid(path)
        self.assertEqual(loaded.data.shape, grid.data.shape)
        self.assertTrue(self.wo.is_occupied(loaded, 2.0, 0.0))


class TestAttribution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.attr = _load_script('attribution')
        cls.wo = _load_script('world_occupancy')

    @staticmethod
    def _traj(n=50, wx=0.0, wy=0.0, yaw=0.0, dt=0.1):
        return [{'t': i * dt, 'wx': wx, 'wy': wy, 'yaw': yaw} for i in range(n)]

    @staticmethod
    def _truth(source_id, x, y, n=50, dt=0.1, active=True, amplitude=20.0, sigma=1.0):
        status = 'truth_active' if active else 'truth_inactive'
        return [{'t': i * dt, 'id': source_id, 'status': status, 'x': x, 'y': y,
                 'strength': amplitude, 'sigma': sigma} for i in range(n)]

    def test_visible_unmatched_is_not_confirmed(self):
        result = self.attr.compute_attribution(
            traj_rows=self._traj(),
            truth_rows=self._truth('A', 1.5, 0.0),
            matched_truth_ids=set(),
            estimate_rows=[],
            fov_x=4.0, fov_y=3.0, occupancy=None)
        rec = result['per_source']['A']
        self.assertTrue(rec['ever_in_fov'])
        self.assertGreaterEqual(rec['visible_total_s'], 1.0)
        self.assertEqual(rec['failure_class'], 'not_confirmed')

    def test_far_source_is_not_reached(self):
        result = self.attr.compute_attribution(
            traj_rows=self._traj(),
            truth_rows=self._truth('B', 15.0, 15.0),
            matched_truth_ids=set(),
            estimate_rows=[],
            fov_x=4.0, fov_y=3.0, occupancy=None)
        rec = result['per_source']['B']
        self.assertFalse(rec['ever_in_fov'])
        self.assertEqual(rec['failure_class'], 'not_reached')
        self.assertAlmostEqual(rec['min_center_dist_m'], math.hypot(15, 15), places=2)

    def test_wall_between_is_occluded(self):
        boxes = self.wo.parse_world_boxes(SAMPLE_SDF)
        grid = self.wo.rasterize(boxes)
        result = self.attr.compute_attribution(
            traj_rows=self._traj(wx=-0.5, wy=0.0, yaw=0.0),
            truth_rows=self._truth('C', 4.0, 0.0),
            matched_truth_ids=set(),
            estimate_rows=[],
            fov_x=10.0, fov_y=6.0, occupancy=grid)
        rec = result['per_source']['C']
        self.assertEqual(rec['failure_class'], 'occluded')
        self.assertGreater(rec['occluded_events'], 0)

    def test_inactive_in_fov_is_timing_missed(self):
        result = self.attr.compute_attribution(
            traj_rows=self._traj(),
            truth_rows=self._truth('D', 1.5, 0.0, active=False),
            matched_truth_ids=set(),
            estimate_rows=[],
            fov_x=4.0, fov_y=3.0, occupancy=None)
        rec = result['per_source']['D']
        self.assertEqual(rec['failure_class'], 'timing_missed')

    def test_matched_source_has_no_failure_class(self):
        result = self.attr.compute_attribution(
            traj_rows=self._traj(),
            truth_rows=self._truth('A', 1.5, 0.0),
            matched_truth_ids={'A'},
            estimate_rows=[{'t': 2.0, 'id': 'src_1', 'status': 'confirmed',
                            'x': 1.4, 'y': 0.1}],
            fov_x=4.0, fov_y=3.0, occupancy=None)
        rec = result['per_source']['A']
        self.assertIsNone(rec['failure_class'])
        self.assertEqual(result['failure_counts']['not_confirmed'], 0)

    def test_visibility_windows_recorded(self):
        result = self.attr.compute_attribution(
            traj_rows=self._traj(n=100),
            truth_rows=self._truth('A', 1.5, 0.0, n=100),
            matched_truth_ids=set(),
            estimate_rows=[],
            fov_x=4.0, fov_y=3.0, occupancy=None)
        windows = result['per_source']['A']['visible_windows']
        self.assertEqual(len(windows), 1)
        self.assertAlmostEqual(windows[0]['duration_s'], 9.9, delta=0.3)

    def test_failure_classes_single_source(self):
        ms = _load_script('matrix_stats')
        self.assertEqual(self.attr.FAILURE_CLASSES, ms.FAILURE_CLASSES)
        result = self.attr.compute_attribution(
            traj_rows=self._traj(),
            truth_rows=self._truth('A', 1.5, 0.0),
            matched_truth_ids=set(),
            estimate_rows=[],
            fov_x=4.0, fov_y=3.0, occupancy=None)
        self.assertEqual(tuple(result['failure_counts'].keys()), ms.FAILURE_CLASSES)


class TestScenarioRunSeed(unittest.TestCase):
    def _scenario(self):
        from thermal_sensor_sim.scenario import default_config_b_scenario
        return default_config_b_scenario(3)

    def test_seed_zero_is_noop(self):
        from thermal_sensor_sim.scenario import apply_run_seed
        sc = self._scenario()
        x0 = sc.sources[0].world_x
        seed0 = sc.sources[0].seed
        apply_run_seed(sc, 0, jitter_std_m=0.5)
        self.assertEqual(sc.sources[0].world_x, x0)
        self.assertEqual(sc.sources[0].seed, seed0)

    def test_seed_changes_source_seed_deterministically(self):
        from thermal_sensor_sim.scenario import apply_run_seed
        sc1, sc2 = self._scenario(), self._scenario()
        apply_run_seed(sc1, 101, jitter_std_m=0.3)
        apply_run_seed(sc2, 101, jitter_std_m=0.3)
        for a, b in zip(sc1.sources, sc2.sources):
            self.assertEqual(a.seed, b.seed)
            self.assertAlmostEqual(a.world_x, b.world_x, places=9)
            self.assertAlmostEqual(a.world_y, b.world_y, places=9)

    def test_different_seeds_differ(self):
        from thermal_sensor_sim.scenario import apply_run_seed
        sc1, sc2 = self._scenario(), self._scenario()
        apply_run_seed(sc1, 101, jitter_std_m=0.3)
        apply_run_seed(sc2, 102, jitter_std_m=0.3)
        self.assertNotAlmostEqual(sc1.sources[0].world_x, sc2.sources[0].world_x, places=9)
        self.assertNotEqual(sc1.sources[0].seed, sc2.sources[0].seed)

    def test_zero_jitter_keeps_positions(self):
        from thermal_sensor_sim.scenario import apply_run_seed
        sc = self._scenario()
        x0, y0 = sc.sources[0].world_x, sc.sources[0].world_y
        apply_run_seed(sc, 101, jitter_std_m=0.0)
        self.assertEqual((sc.sources[0].world_x, sc.sources[0].world_y), (x0, y0))
        self.assertNotEqual(sc.sources[0].seed, 1)


class TestStaticFiveScenario(unittest.TestCase):
    def test_loads_five_static_sources(self):
        from thermal_sensor_sim.scenario import load_scenario_file
        path = WORKSPACE / 'src/thermal_robot/thermal_bringup/config/scenarios/static_five_sources.yaml'
        sc = load_scenario_file(str(path))
        self.assertEqual(len(sc.sources), 5)
        for src in sc.sources:
            self.assertEqual(src.motion, 'static')
            self.assertGreaterEqual(src.amplitude, 14.0)


class TestMatrixRunnerConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = _load_script('run_multiscenario_matrix')

    def test_phase0_grid_is_4x6(self):
        cases = self.runner.phase0_cases()
        self.assertEqual(len(cases), 24)
        worlds = {c.world.name for c in cases}
        scenarios = {c.scenario.name for c in cases}
        self.assertEqual(len(worlds), 4)
        self.assertEqual(len(scenarios), 6)

    def test_phase0_case_names_use_class_keys(self):
        names = {c.name for c in self.runner.phase0_cases()}
        self.assertIn('open__static2', names)
        self.assertIn('boxes__dyn5', names)
        self.assertIn('walls__birthdeath', names)
        self.assertIn('mixed__static5', names)

    def test_filter_cases_by_world_and_case(self):
        cases = self.runner.phase0_cases()
        only_open = self.runner.filter_cases(cases, worlds='open', case_names='')
        self.assertEqual(len(only_open), 6)
        one = self.runner.filter_cases(cases, worlds='', case_names='open__static2')
        self.assertEqual(len(one), 1)
        two_worlds = self.runner.filter_cases(cases, worlds='open,boxes', case_names='')
        self.assertEqual(len(two_worlds), 12)

    def test_parse_seeds(self):
        self.assertEqual(self.runner.parse_seeds('101,102,103'), [101, 102, 103])
        self.assertEqual(self.runner.parse_seeds(' 7 '), [7])

    def test_occupancy_path_for_world(self):
        cases = self.runner.phase0_cases()
        boxes_case = next(c for c in cases if c.name.startswith('boxes__'))
        occ = self.runner.occupancy_path_for_world(boxes_case.world)
        self.assertTrue(str(occ).endswith('thermal_scene_obstacle_field.npz'))

    def test_scenario_fov_reads_yaml(self):
        path = WORKSPACE / 'src/thermal_robot/thermal_bringup/config/scenarios/dynamic_five_sources.yaml'
        self.assertEqual(self.runner.scenario_fov(path), (4.0, 3.0))

    def test_scenario_fov_custom_values(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'custom.yaml'
            path.write_text('sensor:\n  fov_x_m: 6.0\n  fov_y_m: 4.5\n')
            self.assertEqual(self.runner.scenario_fov(path), (6.0, 4.5))

    def test_scenario_fov_defaults_without_sensor_block(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'plain.yaml'
            path.write_text('sources: []\n')
            self.assertEqual(self.runner.scenario_fov(path), (4.0, 3.0))

    def test_legacy_presets_still_present(self):
        self.assertGreaterEqual(len(self.runner.REPRESENTATIVE_CASES), 4)
        self.assertGreaterEqual(len(self.runner.VARIABLE_SOURCE_CASES), 3)


if __name__ == '__main__':
    unittest.main()
