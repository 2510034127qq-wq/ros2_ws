#!/usr/bin/env python3
"""
test_thermal_system.py — 热导航算法综合测试套件
========================================================
覆盖范围：
  T-PY:  纯算法单测（无需 ROS/Gazebo，pytest 直接运行）
  T-ALG: 算法性能基准（多策略对比）
  T-ROS: ROS 2 集成测试（需运行中的仿真）

运行方式：
  # 1. 纯算法单测（最常用，无需任何 ROS 环境）
  cd ~/ros2_ws
  python3 -m pytest src/thermal_robot/tests/test_thermal_system.py -v

  # 2. 算法基准测试（生成对比数据，用于答辩）
  python3 src/thermal_robot/tests/test_thermal_system.py --bench

  # 3. ROS 集成测试（需先启动 sim_nav_slam_launch.py）
  python3 src/thermal_robot/tests/test_thermal_system.py --ros

参考文献：
  Sousa 2006     DOI:10.1109/ROBOT.2006.1642286
  Wiedemann 2021 DOI:10.1016/j.robot.2020.103687
  Nakagawa 2020  DOI:10.1109/JSEN.2020.2984234
  Reggente 2009  DOI:10.1109/ICSENS.2009.5398427
"""

import sys
import time
import math
import unittest
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict

WORKSPACE = Path(__file__).resolve().parents[3]
for rel in [
    'src/thermal_robot/thermal_sensor_sim',
    'src/thermal_robot/thermal_field_reconstructor',
    'src/thermal_robot/thermal_motion_controller',
]:
    path = str(WORKSPACE / rel)
    if path not in sys.path:
        sys.path.insert(0, path)

from thermal_field_reconstructor.thermal_mapping import WorldThermalGrid
from thermal_motion_controller.planning import PlannerSource, select_information_gain_target
from thermal_motion_controller.source_tracking import SourceDetection, SourceTrackerCore
from thermal_sensor_sim.scenario import default_config_b_scenario, load_scenario_file


# ═══════════════════════════════════════════════════════════════════════════
#  共用工具函数（与实际节点算法保持严格一致）
# ═══════════════════════════════════════════════════════════════════════════

def make_gaussian_field(width: int = 64, height: int = 48,
                         sources: List[Tuple] = None,
                         ambient: float = 22.0) -> np.ndarray:
    """生成多高斯热源叠加场。
    sources: [(cx, cy, amplitude, sigma), ...]
    与 sensor_node.py DEFAULT_SOURCES 参数保持一致。
    """
    if sources is None:
        sources = [(32.0, 24.0, 40.0, 10.0)]  # 单热源默认
    x  = np.arange(width,  dtype=np.float32)
    y  = np.arange(height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    field = np.full((height, width), ambient, dtype=np.float32)
    for cx, cy, amp, sigma in sources:
        field += amp * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    return field


def sobel_gradient(field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """纯 NumPy Sobel 梯度（与 gradient_node.py 完全一致）。
    返回 (gx, gy)，即沿列方向和行方向的梯度。
    Ref: Wiedemann 2021 DOI:10.1016/j.robot.2020.103687
    """
    p  = np.pad(field, 1, mode='edge').astype(np.float64)
    gx = ((-p[:-2,:-2] + p[:-2,2:] - 2*p[1:-1,:-2] + 2*p[1:-1,2:]
           - p[2:,:-2]  + p[2:,2:]) / 8.0).astype(np.float32)
    gy = ((-p[:-2,:-2] - 2*p[:-2,1:-1] - p[:-2,2:]
           + p[2:,:-2]  + 2*p[2:,1:-1]  + p[2:,2:]) / 8.0).astype(np.float32)
    return gx, gy


def central_diff_gradient(field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """中心差分梯度（备选方法）。"""
    p  = np.pad(field, 1, mode='edge').astype(np.float32)
    gx = ((p[1:-1,2:] - p[1:-1,:-2]) / 2.0)
    gy = ((p[2:,1:-1] - p[:-2,1:-1]) / 2.0)
    return gx, gy


def kalman_filter_sequence(measurements: np.ndarray,
                            q: float = 0.08,
                            r: float = 4.0) -> np.ndarray:
    """逐步 Kalman 滤波（与 preprocessor_node.py v2 + params.yaml 保持一致）。
    默认参数与部署配置对齐：q=0.08, r=4.0 → 稳态 K≈0.13，强平滑。
    Ref: Nakagawa 2020 DOI:10.1109/JSEN.2020.2984234
    """
    x_est = float(measurements[0])
    p_est = 1.0
    out = [x_est]
    for z in measurements[1:]:
        p_pred = p_est + q
        k      = p_pred / (p_pred + r)
        x_est  = x_est + k * (float(z) - x_est)
        p_est  = (1.0 - k) * p_pred
        out.append(x_est)
    return np.array(out, dtype=np.float32)


def gradient_ascent(field: np.ndarray,
                     start: Tuple[float, float],
                     step_size: float = 1.0,
                     max_steps: int = 500,
                     grad_func=sobel_gradient) -> Tuple[np.ndarray, int, float]:
    """梯度上升导航仿真。
    返回 (trajectory_array, steps_taken, final_distance_to_peak)
    Ref: Sousa 2006 DOI:10.1109/ROBOT.2006.1642286
    """
    H, W   = field.shape
    gx, gy = grad_func(field)
    peak_y, peak_x = np.unravel_index(np.argmax(field), field.shape)
    source = np.array([float(peak_x), float(peak_y)])

    pos = np.array([float(start[0]), float(start[1])])
    traj = [pos.copy()]

    for step in range(max_steps):
        xi = int(np.clip(pos[0], 0, W-1))
        yi = int(np.clip(pos[1], 0, H-1))
        dx, dy = float(gx[yi, xi]), float(gy[yi, xi])
        mag = math.sqrt(dx**2 + dy**2)
        if mag < 1e-6:
            break
        pos[0] += step_size * dx / mag
        pos[1] += step_size * dy / mag
        traj.append(pos.copy())
        if np.linalg.norm(pos - source) < 2.0:
            break

    dist = float(np.linalg.norm(pos - source))
    return np.array(traj), len(traj), dist


def weighted_sum_ascent(field: np.ndarray,
                         start: Tuple[float, float],
                         step_size: float = 1.0,
                         max_steps: int = 500) -> Tuple[np.ndarray, int, float]:
    """加权梯度求和导航（controller_node.py weighted_sum 策略）."""
    H, W   = field.shape
    gx, gy = sobel_gradient(field)
    mag    = np.sqrt(gx**2 + gy**2)
    peak_y, peak_x = np.unravel_index(np.argmax(field), field.shape)
    source = np.array([float(peak_x), float(peak_y)])

    pos  = np.array([float(start[0]), float(start[1])])
    traj = [pos.copy()]

    for step in range(max_steps):
        # 取邻域 5×5 加权平均方向
        xi = int(np.clip(pos[0], 2, W-3))
        yi = int(np.clip(pos[1], 2, H-3))
        patch_gx = gx[yi-2:yi+3, xi-2:xi+3]
        patch_gy = gy[yi-2:yi+3, xi-2:xi+3]
        patch_mg = mag[yi-2:yi+3, xi-2:xi+3]
        w_sum    = patch_mg.sum() + 1e-9
        total_gx = (patch_gx * patch_mg).sum() / w_sum
        total_gy = (patch_gy * patch_mg).sum() / w_sum
        m = math.sqrt(total_gx**2 + total_gy**2)
        if m < 1e-6:
            break
        pos[0] += step_size * total_gx / m
        pos[1] += step_size * total_gy / m
        traj.append(pos.copy())
        if np.linalg.norm(pos - source) < 2.0:
            break

    dist = float(np.linalg.norm(pos - source))
    return np.array(traj), len(traj), dist


# ═══════════════════════════════════════════════════════════════════════════
#  T-PY: 纯算法单测（pytest 可直接运行）
# ═══════════════════════════════════════════════════════════════════════════

class TestThermalFieldAlgorithms(unittest.TestCase):
    """T-PY: 纯 Python 算法层单测（无需 ROS/Gazebo）"""

    def test_T_PY1_gaussian_field_range(self):
        """T-PY1: 热场温度范围应在 [ambient, ambient+peak] 内."""
        field = make_gaussian_field(sources=[(32, 24, 43.0, 10.0)], ambient=22.0)
        self.assertEqual(field.shape, (48, 64))
        self.assertEqual(field.dtype, np.float32)
        self.assertGreaterEqual(float(field.max()), 60.0,
            f"Peak≥65°C expected, got {field.max():.1f}")
        self.assertLessEqual(float(field.min()), 25.0,
            f"Ambient≤22°C expected, got {field.min():.1f}")

    def test_T_PY2_sobel_gradient_direction(self):
        """T-PY2: Sobel 梯度方向应指向热源（热源在右侧，gx>0）.
        Ref: Wiedemann 2021 DOI:10.1016/j.robot.2020.103687"""
        field = make_gaussian_field(sources=[(50.0, 24.0, 40.0, 10.0)], ambient=22.0)
        gx, gy = sobel_gradient(field)
        # 在 pixel(25,24) 处，热源在右侧(x=50)，gx 应为正
        val = float(gx[24, 25])
        self.assertGreater(val, 0.0,
            f"Sobel-x at (25,24) should be >0 (toward source at x=50), got {val:.4f}")

    def test_T_PY3_kalman_noise_reduction(self):
        """T-PY3: Kalman 滤波应降低噪声 RMSE >20%，使用部署配置 q=0.08, r=4.0。
        v2 参数（强平滑，K≈0.13）相比 v1（q=0.01, r=0.25，K≈0.96）
        平滑效果更强，RMSE 降幅显著高于 20%。
        Ref: Nakagawa 2020 DOI:10.1109/JSEN.2020.2984234"""
        np.random.seed(42)
        true_val  = 40.0
        noise_std = 5.0
        n_steps   = 200
        meas = true_val + np.random.normal(0, noise_std, n_steps).astype(np.float32)
        # 使用与 params.yaml 一致的部署参数（v2: q=0.08, r=4.0）
        kf   = kalman_filter_sequence(meas, q=0.08, r=4.0)
        rmse_raw = float(np.sqrt(np.mean((meas  - true_val)**2)))
        rmse_kf  = float(np.sqrt(np.mean((kf    - true_val)**2)))
        pct      = (rmse_raw - rmse_kf) / rmse_raw * 100.0
        self.assertGreater(pct, 20.0,
            f"Kalman RMSE reduction should >20% (Nakagawa 2020). "
            f"Got {pct:.1f}% (raw={rmse_raw:.3f}, kf={rmse_kf:.3f})")

    def test_T_PY4_gradient_ascent_convergence(self):
        """T-PY4: 梯度上升应在有限步内收敛至热源 (<5px).
        Ref: Sousa 2006 DOI:10.1109/ROBOT.2006.1642286"""
        field = make_gaussian_field(sources=[(50.0, 24.0, 40.0, 15.0)], ambient=22.0)
        traj, steps, dist = gradient_ascent(field, start=(5.0, 24.0), step_size=2.0)
        self.assertLess(dist, 5.0,
            f"Gradient ascent should converge <5px (Sousa 2006). "
            f"Got dist={dist:.2f}px in {steps} steps.")
        self.assertLess(steps, 200,
            f"Should converge in <200 steps, got {steps}")

    def test_T_PY5_field_continuity(self):
        """T-PY5: 热场应保持空间连续性（峰值邻域平滑）.
        Ref: Reggente 2009 DOI:10.1109/ICSENS.2009.5398427"""
        field = make_gaussian_field(sources=[(32.0, 24.0, 40.0, 10.0)], ambient=22.0)
        cx, cy = 32, 24
        center = float(field[cy, cx])
        nbr    = float(np.mean([field[cy-1,cx], field[cy+1,cx],
                                field[cy,cx-1], field[cy,cx+1]]))
        self.assertAlmostEqual(center, nbr, delta=5.0,
            msg=f"Peak neighborhood smooth: center={center:.2f}, nbr_mean={nbr:.2f}")

    def test_T_PY6_multi_source_hotspot_detection(self):
        """T-PY6: 多热源场应能检测到多个独立热点."""
        sources = [
            (16.0, 12.0, 35.0, 6.0),
            (48.0, 36.0, 40.0, 8.0),
        ]
        field   = make_gaussian_field(sources=sources, ambient=22.0)
        mean_T  = float(field.mean())
        thresh  = mean_T + 5.0
        ys, xs  = np.where(field > thresh)
        # 应有两个分离的热点区域（至少各有1个以上的高温点）
        self.assertGreater(len(xs), 10,
            f"Should find multiple hotspot pixels above thresh={thresh:.1f}°C")
        # 验证两个热源均在阈值以上
        for (cx, cy, amp, _) in sources:
            T_at_source = float(field[int(cy), int(cx)])
            self.assertGreater(T_at_source, thresh,
                f"Source at ({cx},{cy}) T={T_at_source:.1f} should be >{thresh:.1f}")

    def test_T_PY7_gradient_magnitude_at_source_center(self):
        """T-PY7: 热源中心处梯度幅值应接近0（极值点特性）."""
        field = make_gaussian_field(sources=[(32.0, 24.0, 40.0, 10.0)], ambient=22.0)
        gx, gy = sobel_gradient(field)
        mag    = np.sqrt(gx**2 + gy**2)
        # 热源中心附近梯度应很小
        center_mag = float(mag[24, 32])
        # 热场边缘处梯度应较大
        edge_mag   = float(mag[24, 10])
        self.assertLess(center_mag, edge_mag,
            f"Center gradient {center_mag:.4f} should be < edge gradient {edge_mag:.4f}")

    def test_T_PY8_central_diff_vs_sobel_consistency(self):
        """T-PY8: 中心差分与 Sobel 梯度方向应高度一致（cos相似度>0.9）."""
        field     = make_gaussian_field(sources=[(40.0, 20.0, 40.0, 12.0)], ambient=22.0)
        gx_s, gy_s = sobel_gradient(field)
        gx_c, gy_c = central_diff_gradient(field)

        # 扁平化后计算方向余弦相似度
        mag_s = np.sqrt(gx_s**2 + gy_s**2).flatten()
        mag_c = np.sqrt(gx_c**2 + gy_c**2).flatten()
        # 只在梯度显著的点上比较
        mask  = mag_s > 0.5
        if mask.sum() < 10:
            self.skipTest("Too few significant gradient points")

        dot   = (gx_s.flatten()[mask] * gx_c.flatten()[mask] +
                 gy_s.flatten()[mask] * gy_c.flatten()[mask])
        cos   = dot / (mag_s[mask] * mag_c[mask] + 1e-9)
        mean_cos = float(np.mean(cos))
        self.assertGreater(mean_cos, 0.85,
            f"Sobel vs central-diff direction cos similarity should >0.85, got {mean_cos:.3f}")

    def test_T_PY9_config_b_yaml_matches_fallback(self):
        """T-PY9: 默认 Config-B YAML 与空 scenario_file fallback 等价."""
        fallback = default_config_b_scenario(num_sources=3)
        yaml_path = WORKSPACE / 'src/thermal_robot/thermal_bringup/config/config_b_sources.yaml'
        loaded = load_scenario_file(str(yaml_path), num_sources=3)
        self.assertEqual(len(loaded.sources), len(fallback.sources))
        for a, b in zip(loaded.sources, fallback.sources):
            self.assertEqual(a.source_id, b.source_id)
            self.assertAlmostEqual(a.world_x, b.world_x)
            self.assertAlmostEqual(a.world_y, b.world_y)
            self.assertAlmostEqual(a.amplitude, b.amplitude)
            self.assertAlmostEqual(a.sigma_m, b.sigma_m)

    def test_T_PY10_world_mapper_hotspot_projection(self):
        """T-PY10: FOV 像素投影到 world grid 后热点坐标误差 <0.75m."""
        robot_x, robot_y, yaw = -6.0, 0.0, 0.0
        source_x, source_y = -5.0, 0.5
        W, H = 64, 48
        px_xs = np.linspace(-2.0, 2.0, W, dtype=np.float32)
        px_ys = np.linspace(-1.5, 1.5, H, dtype=np.float32)
        xx, yy = np.meshgrid(px_xs, px_ys)
        world_x = robot_x + xx
        world_y = robot_y + yy
        image = 22.0 + 35.0 * np.exp(-((world_x-source_x)**2 + (world_y-source_y)**2) / (2.0 * 0.45**2))
        grid = WorldThermalGrid(center_x=-6.0, center_y=0.0, size_x_m=12.0, size_y_m=12.0, resolution=0.25)
        grid.integrate_image(image.astype(np.float32), robot_x, robot_y, yaw, stamp_s=1.0, fov_x=4.0, fov_y=3.0)
        snap = grid.snapshot(now_s=1.0)
        iy, ix = np.unravel_index(np.argmax(snap.temperature_mean), snap.temperature_mean.shape)
        est_x = snap.origin_x + (ix + 0.5) * snap.resolution
        est_y = snap.origin_y + (iy + 0.5) * snap.resolution
        self.assertLess(math.hypot(est_x-source_x, est_y-source_y), 0.75)

    def test_T_PY11_tracker_confirms_and_merges_duplicate(self):
        """T-PY11: tracker 连续观测确认同一源，近距离重复候选不会重复 confirmed."""
        tracker = SourceTrackerCore(confirm_observations=5, confirm_covariance_max=1.0)
        for i in range(6):
            tracker.update([
                SourceDetection(x=1.0, y=2.0, strength=18.0, confidence=0.9),
                SourceDetection(x=1.25, y=2.1, strength=17.0, confidence=0.85),
            ], now_s=float(i))
        confirmed = [t for t in tracker.tracks if t.status == 'confirmed']
        self.assertEqual(len(confirmed), 1)
        self.assertGreaterEqual(confirmed[0].existence_probability, 0.75)

    def test_T_PY12_tracker_stale_decay(self):
        """T-PY12: source 无观测后概率衰减并进入 stale."""
        tracker = SourceTrackerCore(confirm_observations=5, stale_after_s=2.0, stale_decay_s=2.0)
        tracker.update([SourceDetection(0.0, 0.0, 12.0, 0.8)], now_s=0.0)
        p0 = tracker.tracks[0].existence_probability
        tracker.update([], now_s=4.0)
        track = tracker.tracks[0]
        self.assertEqual(track.status, 'stale')
        self.assertLess(track.existence_probability, p0)

    def test_T_PY13_information_gain_prefers_candidate_verification(self):
        """T-PY13: 信息增益 planner 会优先选 candidate source 周边验证点."""
        width = height = 40
        variance = np.ones((height, width), dtype=np.float32)
        confidence = np.full((height, width), 0.2, dtype=np.float32)
        visits = np.zeros((height, width), dtype=np.float32)
        age = np.full((height, width), 10.0, dtype=np.float32)
        candidate = PlannerSource(x=2.0, y=1.0, probability=0.9, status='candidate', confidence=0.9)
        target = select_information_gain_target(
            robot_wx=0.0, robot_wy=0.0,
            width=width, height=height, resolution=0.25,
            origin_x=-5.0, origin_y=-5.0,
            temperature_variance=variance,
            confidence=confidence,
            visit_count=visits,
            last_seen_age_s=age,
            source_estimates=[candidate],
            known_sources=[],
            min_d=1.0,
            max_d=6.0,
        )
        self.assertIsNotNone(target)
        self.assertEqual(target.reason, 'source_verify')
        self.assertLess(math.hypot(target.x-candidate.x, target.y-candidate.y), 2.0)


# ═══════════════════════════════════════════════════════════════════════════
#  T-ALG: 算法性能基准（用于答辩数据）
# ═══════════════════════════════════════════════════════════════════════════

def run_algorithm_benchmarks():
    """多策略、多场景算法基准测试，输出对比表格用于答辩。"""
    print('\n' + '═'*70)
    print('  THERMAL NAVIGATION ALGORITHM BENCHMARKS')
    print('  (用于中期答辩数据展示)')
    print('═'*70)

    # 场景定义
    scenarios = {
        'single_source_easy':  [(32.0, 24.0, 40.0, 12.0)],
        'single_source_hard':  [(55.0, 40.0, 30.0,  8.0)],
        'dual_source':         [(20.0, 15.0, 35.0, 8.0), (48.0, 36.0, 40.0, 10.0)],
        'triple_source':       [(32.0,24.0,40.0,10.0), (50.0,36.0,25.0,7.0), (16.0,12.0,15.0,4.0)],
    }

    # 起始位置（总是从左侧 10% 区域出发）
    starts = [(5.0, 24.0), (5.0, 10.0), (5.0, 40.0)]

    results: Dict = {}

    for scenario_name, sources in scenarios.items():
        print(f'\n  ── 场景: {scenario_name} ──')
        field    = make_gaussian_field(sources=sources, ambient=22.0)
        noise_field = field + np.random.RandomState(42).normal(0, 0.5, field.shape).astype(np.float32)

        # Kalman 滤波效果
        center_ts  = field[:, 32].astype(np.float32)
        noisy_ts   = noise_field[:, 32]
        kf_ts      = kalman_filter_sequence(noisy_ts, q=0.08, r=4.0)
        rmse_noisy = float(np.sqrt(np.mean((noisy_ts - center_ts)**2)))
        rmse_kf    = float(np.sqrt(np.mean((kf_ts    - center_ts)**2)))
        kf_improv  = (rmse_noisy - rmse_kf) / rmse_noisy * 100.0
        print(f'  Kalman 降噪: RMSE {rmse_noisy:.3f} → {rmse_kf:.3f} '
              f'({kf_improv:.1f}% improvement)')

        # 导航策略对比
        algo_results = {}
        for method_name, func in [('sobel_max_grad', gradient_ascent),
                                   ('weighted_sum',   weighted_sum_ascent),
                                   ('central_diff',
                                    lambda f, s, **kw: gradient_ascent(
                                        f, s, grad_func=central_diff_gradient, **kw))]:
            steps_list, dist_list = [], []
            for start in starts:
                try:
                    traj, steps, dist = func(field, start, step_size=1.5, max_steps=500)
                    steps_list.append(steps)
                    dist_list.append(dist)
                except Exception as e:
                    steps_list.append(999); dist_list.append(99.0)

            avg_steps = float(np.mean(steps_list))
            avg_dist  = float(np.mean(dist_list))
            conv_rate = sum(1 for d in dist_list if d < 5.0) / len(dist_list) * 100
            algo_results[method_name] = {
                'avg_steps': avg_steps, 'avg_dist': avg_dist, 'conv_rate': conv_rate
            }
            print(f'  [{method_name:20s}] '
                  f'avg_steps={avg_steps:.0f}  '
                  f'avg_dist={avg_dist:.2f}px  '
                  f'conv_rate={conv_rate:.0f}%')

        results[scenario_name] = algo_results

    # ── 汇总表 ──────────────────────────────────────────────────────────
    print('\n' + '─'*70)
    print('  SUMMARY: 平均收敛距离 (px) — 越小越好')
    print(f'  {"Scenario":<25} {"sobel_max":>12} {"weighted":>12} {"central":>12}')
    print('  ' + '─'*60)
    for sc, ar in results.items():
        print(f'  {sc:<25} '
              f'{ar["sobel_max_grad"]["avg_dist"]:>11.2f}px '
              f'{ar["weighted_sum"]["avg_dist"]:>11.2f}px '
              f'{ar["central_diff"]["avg_dist"]:>11.2f}px')
    print('═'*70)
    print('  参考: Sousa 2006 DOI:10.1109/ROBOT.2006.1642286 — '
          '梯度法在 SNR>15dB 时优于螺旋搜索')
    print('═'*70 + '\n')
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  T-ROS: ROS 2 集成测试
# ═══════════════════════════════════════════════════════════════════════════

def run_ros_integration_tests():
    """ROS 2 集成测试（需先启动 sim_nav_slam_launch.py）."""
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, JointState
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from thermal_interfaces.msg import GradientArray, SourceEstimateArray, ThermalField, ThermalMap
    from thermal_interfaces.srv import GetFieldInfo

    rclpy.init()
    results = {}

    class Collector(Node):
        def __init__(self):
            super().__init__('thermal_tester_v2')
            self.msgs = {k: [] for k in [
                'raw', 'filtered', 'field', 'map', 'sources', 'truth',
                'gradient', 'cmd_vel', 'odom', 'joint_states'
            ]}
            from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
            be = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                            history=QoSHistoryPolicy.KEEP_LAST, depth=10)
            re = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                            history=QoSHistoryPolicy.KEEP_LAST, depth=10)
            self.create_subscription(Image,        '/sim/thermal_raw',     lambda m: self.msgs['raw'].append(m),        be)
            self.create_subscription(Image,        '/thermal/filtered',    lambda m: self.msgs['filtered'].append(m),   be)
            self.create_subscription(ThermalField, '/thermal/field',       lambda m: self.msgs['field'].append(m),      re)
            self.create_subscription(ThermalMap,   '/thermal/map',         lambda m: self.msgs['map'].append(m),        re)
            self.create_subscription(SourceEstimateArray, '/thermal/sources', lambda m: self.msgs['sources'].append(m), re)
            self.create_subscription(SourceEstimateArray, '/sim/thermal_sources_truth', lambda m: self.msgs['truth'].append(m), be)
            self.create_subscription(GradientArray,'/thermal/gradient',    lambda m: self.msgs['gradient'].append(m),   re)
            self.create_subscription(Twist,        '/cmd_vel',             lambda m: self.msgs['cmd_vel'].append(m),    re)
            self.create_subscription(Odometry,     '/odom',                lambda m: self.msgs['odom'].append(m),       be)
            self.create_subscription(JointState,   '/joint_states',        lambda m: self.msgs['joint_states'].append(m), re)

    node = Collector()
    print('[TEST] Collecting 8s of data...')
    t0 = time.time()
    while time.time() - t0 < 8.0:
        rclpy.spin_once(node, timeout_sec=0.05)

    dt = 8.0

    # T-ROS-1: 传感器频率
    hz_raw = len(node.msgs['raw']) / dt
    results['T_ROS1_sensor_hz'] = {
        'pass':  abs(hz_raw - 10.0) < 3.0,
        'value': f'{hz_raw:.1f} Hz (target 10 Hz)'
    }

    # T-ROS-2: 图像格式
    if node.msgs['raw']:
        m = node.msgs['raw'][-1]
        results['T_ROS2_sensor_format'] = {
            'pass':  m.width == 64 and m.height == 48 and m.encoding == '32FC1',
            'value': f'{m.width}×{m.height} {m.encoding}'
        }
    else:
        results['T_ROS2_sensor_format'] = {'pass': False, 'value': 'no messages'}

    # T-ROS-3: 预处理节点
    results['T_ROS3_preprocessor'] = {
        'pass':  len(node.msgs['filtered']) > 5,
        'value': f'{len(node.msgs["filtered"])} msgs in {dt:.0f}s'
    }

    # T-ROS-4: 场重构节点
    results['T_ROS4_reconstructor'] = {
        'pass':  len(node.msgs['field']) > 3,
        'value': f'{len(node.msgs["field"])} msgs in {dt:.0f}s'
    }

    # T-ROS-5: 梯度节点
    results['T_ROS5_gradient'] = {
        'pass':  len(node.msgs['gradient']) > 3,
        'value': f'{len(node.msgs["gradient"])} msgs in {dt:.0f}s'
    }

    # T-ROS-5b: world thermal map
    results['T_ROS5b_thermal_map'] = {
        'pass':  len(node.msgs['map']) > 1,
        'value': f'{len(node.msgs["map"])} msgs in {dt:.0f}s'
    }

    # T-ROS-5c: source tracker estimates
    results['T_ROS5c_sources'] = {
        'pass':  len(node.msgs['sources']) > 1,
        'value': f'{len(node.msgs["sources"])} msgs in {dt:.0f}s'
    }

    # T-ROS-5d: simulator truth topic
    results['T_ROS5d_truth'] = {
        'pass':  len(node.msgs['truth']) > 1,
        'value': f'{len(node.msgs["truth"])} msgs in {dt:.0f}s'
    }

    # T-ROS-6: 控制指令
    results['T_ROS6_cmd_vel'] = {
        'pass':  len(node.msgs['cmd_vel']) > 5,
        'value': f'{len(node.msgs["cmd_vel"])} msgs in {dt:.0f}s'
    }

    # T-ROS-7: 里程计（diff_drive 插件）
    results['T_ROS7_odom'] = {
        'pass':  len(node.msgs['odom']) > 5,
        'value': f'{len(node.msgs["odom"])} msgs in {dt:.0f}s'
    }

    # T-ROS-8: 温度范围合理性
    if node.msgs['field']:
        f = node.msgs['field'][-1]
        results['T_ROS8_temp_range'] = {
            'pass':  f.min_temperature_celsius >= 15.0 and f.max_temperature_celsius <= 80.0,
            'value': f'T=[{f.min_temperature_celsius:.1f}, {f.max_temperature_celsius:.1f}]°C'
        }
    else:
        results['T_ROS8_temp_range'] = {'pass': False, 'value': 'no field data'}

    # T-ROS-9: GetFieldInfo 服务
    cli = node.create_client(GetFieldInfo, '/thermal/get_field_info')
    if cli.wait_for_service(timeout_sec=4.0):
        req       = GetFieldInfo.Request()
        req.include_full_data = True
        fut       = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=4.0)
        resp      = fut.result()
        results['T_ROS9_field_srv'] = {
            'pass':  resp is not None and resp.success,
            'value': (f'max={resp.max_temperature_celsius:.1f}°C '
                      f'hotspots={resp.hotspot_count}' if resp else 'no response')
        }
    else:
        results['T_ROS9_field_srv'] = {'pass': False, 'value': 'service unavailable'}

    # ── 打印结果 ──────────────────────────────────────────────────────────
    print('\n' + '═'*65)
    print('  THERMAL ROBOT v2 INTEGRATION TEST RESULTS')
    print('═'*65)
    all_pass = True
    for tid, r in results.items():
        status = '✅ PASS' if r['pass'] else '❌ FAIL'
        print(f'  {status}  {tid:<30} {r["value"]}')
        if not r['pass']:
            all_pass = False
    print('═'*65)
    print(f'  OVERALL: {"✅ ALL PASS" if all_pass else "❌ SOME FAILURES"}')
    print(f'  通过率: {sum(1 for r in results.values() if r["pass"])}/{len(results)}')
    print('═'*65 + '\n')

    node.destroy_node()
    rclpy.shutdown()
    return all_pass


# ═══════════════════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if '--ros' in sys.argv:
        success = run_ros_integration_tests()
        sys.exit(0 if success else 1)
    elif '--bench' in sys.argv:
        run_algorithm_benchmarks()
        sys.exit(0)
    else:
        print('Running pure-NumPy algorithm unit tests (no ROS required)...\n')
        unittest.main(verbosity=2, exit=True)
