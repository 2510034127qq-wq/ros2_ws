#!/usr/bin/env python3
"""
collect_sim_data.py — 热导航仿真数据采集节点 v3 (SLAM + Nav2 版)
=================================================================
v3 新增（相比 v2）：
  - SLAM 位姿：从 TF(map→base_link) 获取，保存到 slam_trajectory.csv
  - Nav2 规划路径：订阅 /plan，统计路径点数、长度
  - /scan 统计：激光扫描点数、范围统计（验证SLAM输入）
  - nav2_events.csv：Nav2 goal 发送/接受/完成事件（通过 /rosout 解析）
  - metadata.json 新增 slam_available、nav2_available 字段

用法：
  # 终端1：启动仿真
  cd ~/ros2_ws && source install/setup.bash
  ros2 launch thermal_bringup sim_nav_slam_launch.py

  # 终端2：启动采集（仿真启动约20s后，等待SLAM就绪再运行）
  cd ~/ros2_ws && source install/setup.bash
  python3 ~/ros2_ws/src/thermal_robot/scripts/collect_sim_data.py

  # 仿真结束后，终端2 Ctrl+C，数据保存至：
  ~/ros2_ws/bags/collected/<YYYYMMDD_HHMMSS>/

收集内容（v3）：
  trajectory.csv       — 机器人位姿 (odom 坐标系，含 spawn 偏移)
  slam_trajectory.csv  — 机器人位姿 (map 坐标系，SLAM 修正) [v3新增]
  thermal_stats.csv    — raw/filtered 图像统计（每帧）
  field_stats.csv      — 热场重构统计 + 热点
  gradient_stats.csv   — 梯度幅值/方向统计
  cmd_vel.csv          — 速度指令
  scan_stats.csv       — 激光扫描统计 [v3新增]
  nav2_plan_stats.csv  — Nav2 规划路径统计 [v3新增]
  snapshots/           — 热图像完整阵列（每5s一张）
  metadata.json        — 配置与统计摘要
"""

import csv
import argparse
import json
import math
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path as NavPath
from sensor_msgs.msg import Image, LaserScan
from thermal_interfaces.msg import GradientArray, SourceEstimateArray, ThermalField, ThermalMap

# TF2
try:
    from tf2_ros import Buffer, TransformListener
    from tf2_ros import LookupException, ExtrapolationException, ConnectivityException
    TF2_AVAILABLE = True
except ImportError:
    TF2_AVAILABLE = False

# ─── 输出目录 ───────────────────────────────────────────────────────────────
OUTBASE = Path.home() / 'ros2_ws' / 'bags' / 'collected'
SNAPSHOT_INTERVAL_S = 5.0    # 每 N 秒保存一次热图完整阵列
SCAN_RECORD_INTERVAL = 0.5   # 每 0.5s 记录一次激光扫描统计
SPAWN_X = -6.0               # 机器人出生坐标（与 sensor_node/params 一致）
SPAWN_Y =  0.0

# ─── Config-B 热源参数（与 sensor_node.py v13 一致）──────────────────────
CONFIG_B_SOURCES = [
    {'name': 'SA_left', 'xy': [-1.0, 3.5],  'amp': 35.0, 'sigma': 1.1},
    {'name': 'SB_far',  'xy': [6.0,  -3.0], 'amp': 22.0, 'sigma': 0.9},
    {'name': 'SC_weak', 'xy': [-5.0, -5.5], 'amp': 16.0, 'sigma': 0.8},
]

# ─── QoS 预设 ───────────────────────────────────────────────────────────────
BE_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST, depth=5,
    durability=QoSDurabilityPolicy.VOLATILE)
RE_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST, depth=5,
    durability=QoSDurabilityPolicy.VOLATILE)
TRANSIENT_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST, depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


class DataCollector(Node):
    """订阅所有相关话题，缓存数据，关闭时保存至 CSV + numpy。"""

    def __init__(self, out_dir: Path):
        super().__init__('sim_data_collector')
        self._t0 = time.monotonic()
        self._out = out_dir
        self._saved = False
        (self._out / 'snapshots').mkdir(parents=True, exist_ok=True)

        # ── 数据缓冲区 ──────────────────────────────────────────────────────
        self._traj:        list = []   # odom轨迹
        self._slam_traj:   list = []   # SLAM(map frame)轨迹 [v3]
        self._th_stats:    list = []   # 热像统计
        self._field_stats: list = []   # 热场统计
        self._map_stats:   list = []   # world thermal map统计
        self._grad_stats:  list = []   # 梯度统计
        self._sources:     list = []   # tracker source estimates
        self._truth:       list = []   # simulator source truth
        self._source_events: list = []
        self._cmdvel:      list = []   # 速度指令
        self._scan_stats:  list = []   # 激光扫描统计 [v3]
        self._plan_stats:  list = []   # Nav2规划路径统计 [v3]
        self._snap_count   = 0
        self._last_snap    = 0.0
        self._last_scan_rec = 0.0

        # [v3] SLAM 可用性追踪
        self._slam_available = False
        self._slam_first_t   = None
        self._slam_pos_count = 0

        # [v3] Nav2 规划路径追踪
        self._nav2_plan_count = 0
        self._nav2_available  = False

        # 话题发布频率估计
        self._rate_buf: dict = {
            '/odom':              [],
            '/sim/thermal_raw':   [],
            '/thermal/filtered':  [],
            '/thermal/field':     [],
            '/thermal/map':       [],
            '/thermal/sources':   [],
            '/sim/thermal_sources_truth': [],
            '/thermal/gradient':  [],
            '/cmd_vel':           [],
            '/scan':              [],   # [v3]
            '/plan':              [],   # [v3]
        }

        # 最新帧缓存
        self._latest_raw:  np.ndarray | None = None
        self._latest_filt: np.ndarray | None = None
        self._last_source_status: dict = {}

        # ── [v3] TF2 监听器（SLAM位姿） ─────────────────────────────────────
        if TF2_AVAILABLE:
            self._tf_buffer   = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
            self.get_logger().info('[collector v3] TF2 已初始化，将记录 SLAM 位姿')
        else:
            self._tf_buffer = None
            self.get_logger().warn('[collector v3] TF2 不可用，只记录 odom 位姿')

        # ── 订阅器 ──────────────────────────────────────────────────────────
        self.create_subscription(Odometry,     '/odom',             self._odom_cb,   BE_QOS)
        self.create_subscription(Image,        '/sim/thermal_raw',  self._raw_cb,    BE_QOS)
        self.create_subscription(Image,        '/thermal/filtered', self._filt_cb,   BE_QOS)
        self.create_subscription(ThermalField, '/thermal/field',    self._field_cb,  RE_QOS)
        self.create_subscription(ThermalMap,   '/thermal/map',      self._map_cb,    RE_QOS)
        self.create_subscription(SourceEstimateArray, '/thermal/sources', self._sources_cb, RE_QOS)
        self.create_subscription(SourceEstimateArray, '/sim/thermal_sources_truth', self._truth_cb, BE_QOS)
        self.create_subscription(GradientArray,'/thermal/gradient', self._grad_cb,   RE_QOS)
        self.create_subscription(Twist,        '/cmd_vel',          self._cmd_cb,    BE_QOS)
        # [v3] 新增订阅
        self.create_subscription(LaserScan,    '/scan',             self._scan_cb,   BE_QOS)
        self.create_subscription(NavPath,      '/plan',             self._plan_cb,   RE_QOS)

        # [v3] 定时查询 TF（10Hz，与 controller_node 同频）
        if TF2_AVAILABLE:
            self.create_timer(0.1, self._tf_poll_cb)

        self.get_logger().info(
            f'[collector v3] 已订阅 12 个话题（含 /scan, /plan, map/sources/truth），输出→ {out_dir}')
        self.get_logger().info('[collector v3] Ctrl+C 停止并保存数据')

    # ──────────────────────────────────────────────────────────────────────────
    # 辅助方法
    # ──────────────────────────────────────────────────────────────────────────

    def _ts(self) -> float:
        return time.monotonic() - self._t0

    def _img_to_arr(self, msg: Image) -> np.ndarray:
        n = msg.width * msg.height
        return np.frombuffer(bytes(msg.data[:n * 4]), np.float32).reshape(
            msg.height, msg.width).copy()

    def _record_rate(self, topic: str, t: float):
        if topic not in self._rate_buf:
            return
        buf = self._rate_buf[topic]
        buf.append(t)
        if len(buf) > 200:
            buf.pop(0)

    def _snap_if_due(self, t: float):
        if t - self._last_snap < SNAPSHOT_INTERVAL_S:
            return
        self._last_snap = t
        idx = self._snap_count
        self._snap_count += 1
        if self._latest_raw is not None:
            np.save(self._out / 'snapshots' / f'raw_{idx:04d}.npy',  self._latest_raw)
        if self._latest_filt is not None:
            np.save(self._out / 'snapshots' / f'filt_{idx:04d}.npy', self._latest_filt)
        snap_idx_path = self._out / 'snapshots' / 'index.jsonl'
        with open(snap_idx_path, 'a') as f:
            f.write(json.dumps({'idx': idx, 't': t}) + '\n')

    # ──────────────────────────────────────────────────────────────────────────
    # [v3] TF 轮询（SLAM 位姿）
    # ──────────────────────────────────────────────────────────────────────────

    def _tf_poll_cb(self):
        """10Hz定时查询 TF(map→base_link)，记录 SLAM 修正位姿。"""
        if self._tf_buffer is None:
            return
        t = self._ts()
        try:
            tf = self._tf_buffer.lookup_transform(
                'map', 'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))
            x   = tf.transform.translation.x
            y   = tf.transform.translation.y
            q   = tf.transform.rotation
            siny = 2.0*(q.w*q.z + q.x*q.y)
            cosy = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
            yaw  = math.atan2(siny, cosy)
            self._slam_traj.append({'t': t, 'x': x, 'y': y, 'yaw': yaw})
            self._slam_pos_count += 1
            if not self._slam_available:
                self._slam_available = True
                self._slam_first_t   = t
                self.get_logger().info(
                    f'[collector v3] SLAM TF 可用 @ t={t:.1f}s '
                    f'pos=({x:.2f},{y:.2f})')
        except (LookupException, ExtrapolationException, ConnectivityException):
            pass
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────────
    # 回调方法
    # ──────────────────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        t = self._ts()
        self._record_rate('/odom', t)
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        q   = msg.pose.pose.orientation
        siny = 2.0*(q.w*q.z + q.x*q.y)
        cosy = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
        yaw = math.atan2(siny, cosy)
        vx  = msg.twist.twist.linear.x
        wz  = msg.twist.twist.angular.z
        # 世界坐标（加 spawn 偏移，与 sensor_node 一致）
        wx  = SPAWN_X + x
        wy  = SPAWN_Y + y
        self._traj.append({'t': t, 'x': x, 'y': y, 'wx': wx, 'wy': wy,
                           'yaw': yaw, 'vx': vx, 'wz': wz})

    def _raw_cb(self, msg: Image):
        t = self._ts()
        self._record_rate('/sim/thermal_raw', t)
        arr = self._img_to_arr(msg)
        self._latest_raw = arr
        self._th_stats.append({
            't':         t,
            'raw_min':   float(arr.min()),
            'raw_max':   float(arr.max()),
            'raw_mean':  float(arr.mean()),
            'raw_std':   float(arr.std()),
            'raw_p90':   float(np.percentile(arr, 90)),
            'raw_p99':   float(np.percentile(arr, 99)),
        })
        self._snap_if_due(t)

    def _filt_cb(self, msg: Image):
        t = self._ts()
        self._record_rate('/thermal/filtered', t)
        arr = self._img_to_arr(msg)
        self._latest_filt = arr
        if self._th_stats and abs(self._th_stats[-1]['t'] - t) < 0.15:
            rec = self._th_stats[-1]
        else:
            rec = {'t': t, 'raw_min': None, 'raw_max': None,
                   'raw_mean': None, 'raw_std': None,
                   'raw_p90': None, 'raw_p99': None}
            self._th_stats.append(rec)
        rec['filt_min']  = float(arr.min())
        rec['filt_max']  = float(arr.max())
        rec['filt_mean'] = float(arr.mean())
        rec['filt_std']  = float(arr.std())
        rec['filt_p90']  = float(np.percentile(arr, 90))
        rec['filt_p99']  = float(np.percentile(arr, 99))

    def _field_cb(self, msg: ThermalField):
        t = self._ts()
        self._record_rate('/thermal/field', t)
        hs  = msg.hotspots
        rec = {
            't':         t,
            'mean_t':    float(msg.mean_temperature_celsius),
            'max_t':     float(msg.max_temperature_celsius),
            'hs_count':  len(hs),
        }
        for i, h in enumerate(hs[:3]):
            rec[f'hs{i}_px']   = float(h.pixel_x)
            rec[f'hs{i}_py']   = float(h.pixel_y)
            rec[f'hs{i}_temp'] = float(h.temperature_celsius)
            rec[f'hs{i}_conf'] = float(h.confidence)
        self._field_stats.append(rec)

    def _map_cb(self, msg: ThermalMap):
        t = self._ts()
        self._record_rate('/thermal/map', t)
        if msg.width == 0 or msg.height == 0:
            return
        temp = np.asarray(msg.temperature_mean, dtype=np.float32)
        conf = np.asarray(msg.confidence, dtype=np.float32)
        visits = np.asarray(msg.visit_count, dtype=np.float32)
        seen = conf > 0.01
        hot = temp > 26.0
        self._map_stats.append({
            't': t,
            'width': int(msg.width),
            'height': int(msg.height),
            'resolution': float(msg.resolution),
            'seen_ratio': float(seen.mean()) if seen.size else 0.0,
            'mean_confidence': float(conf[seen].mean()) if np.any(seen) else 0.0,
            'max_temp': float(temp.max()) if temp.size else 0.0,
            'hot_cells': int(np.count_nonzero(hot & seen)),
            'visited_cells': int(np.count_nonzero(visits > 0)),
        })

    def _sources_cb(self, msg: SourceEstimateArray):
        t = self._ts()
        self._record_rate('/thermal/sources', t)
        for src in msg.sources:
            rec = self._source_record(t, src)
            self._sources.append(rec)
            key = src.id
            prev = self._last_source_status.get(key)
            if prev != src.status:
                self._source_events.append({
                    't': t,
                    'id': src.id,
                    'from_status': prev or '',
                    'to_status': src.status,
                    'x': float(src.position.x),
                    'y': float(src.position.y),
                    'probability': float(src.existence_probability),
                    'observations': int(src.observations),
                })
                self._last_source_status[key] = src.status

    def _truth_cb(self, msg: SourceEstimateArray):
        t = self._ts()
        self._record_rate('/sim/thermal_sources_truth', t)
        for src in msg.sources:
            self._truth.append(self._source_record(t, src))

    def _source_record(self, t: float, src):
        return {
            't': t,
            'id': src.id,
            'status': src.status,
            'x': float(src.position.x),
            'y': float(src.position.y),
            'strength': float(src.strength),
            'sigma': float(src.sigma),
            'probability': float(src.existence_probability),
            'confidence': float(src.confidence),
            'observations': int(src.observations),
        }

    def _grad_cb(self, msg: GradientArray):
        t = self._ts()
        self._record_rate('/thermal/gradient', t)
        grads = msg.gradients
        if not grads:
            self._grad_stats.append({'t': t, 'n': 0, 'mean_mag': 0.0,
                                     'max_mag': 0.0, 'mean_dx': 0.0,
                                     'mean_dy': 0.0, 'dominant_angle': 0.0,
                                     'peak_t': 0.0, 'peak_px': 0.0, 'peak_py': 0.0})
            return
        mags  = np.array([g.magnitude for g in grads], dtype=np.float32)
        dxs   = np.array([g.grad_x   for g in grads], dtype=np.float32)
        dys   = np.array([g.grad_y   for g in grads], dtype=np.float32)
        w     = mags / (mags.sum() + 1e-9)
        dom_angle = float(math.atan2((w*dys).sum(), (w*dxs).sum()))
        self._grad_stats.append({
            't':              t,
            'n':              len(grads),
            'mean_mag':       float(mags.mean()),
            'max_mag':        float(mags.max()),
            'mean_dx':        float(dxs.mean()),
            'mean_dy':        float(dys.mean()),
            'dominant_angle': dom_angle,
            'peak_t':         float(msg.peak_temperature_celsius),
            'peak_px':        float(msg.peak_pixel_x),
            'peak_py':        float(msg.peak_pixel_y),
        })

    def _cmd_cb(self, msg: Twist):
        t = self._ts()
        self._record_rate('/cmd_vel', t)
        self._cmdvel.append({
            't':     t,
            'lin_x': float(msg.linear.x),
            'ang_z': float(msg.angular.z),
        })

    def _scan_cb(self, msg: LaserScan):
        """[v3] 激光扫描统计（按间隔记录，避免大量数据）。"""
        t = self._ts()
        self._record_rate('/scan', t)
        if t - self._last_scan_rec < SCAN_RECORD_INTERVAL:
            return
        self._last_scan_rec = t

        ranges = np.array(msg.ranges, dtype=np.float32)
        valid  = ranges[(ranges > msg.range_min) & (ranges < msg.range_max)]
        n_valid = len(valid)
        n_total = len(ranges)

        self._scan_stats.append({
            't':             t,
            'n_rays':        n_total,
            'n_valid':       n_valid,
            'valid_ratio':   round(float(n_valid/n_total) if n_total>0 else 0.0, 3),
            'range_min':     float(valid.min())   if n_valid>0 else float('nan'),
            'range_max':     float(valid.max())   if n_valid>0 else float('nan'),
            'range_mean':    float(valid.mean())  if n_valid>0 else float('nan'),
            'range_median':  float(np.median(valid)) if n_valid>0 else float('nan'),
        })

    def _plan_cb(self, msg: NavPath):
        """[v3] Nav2 规划路径统计（路径长度、路径点数）。"""
        t = self._ts()
        self._record_rate('/plan', t)
        self._nav2_available = True
        self._nav2_plan_count += 1

        poses  = msg.poses
        n_pts  = len(poses)
        length = 0.0
        if n_pts > 1:
            for i in range(1, n_pts):
                dx = poses[i].pose.position.x - poses[i-1].pose.position.x
                dy = poses[i].pose.position.y - poses[i-1].pose.position.y
                length += math.hypot(dx, dy)

        self._plan_stats.append({
            't':            t,
            'plan_idx':     self._nav2_plan_count,
            'n_waypoints':  n_pts,
            'path_length':  round(length, 3),
            'start_x':      float(poses[0].pose.position.x) if n_pts>0 else float('nan'),
            'start_y':      float(poses[0].pose.position.y) if n_pts>0 else float('nan'),
            'goal_x':       float(poses[-1].pose.position.x) if n_pts>0 else float('nan'),
            'goal_y':       float(poses[-1].pose.position.y) if n_pts>0 else float('nan'),
        })

    def _source_level_summary(self, duration: float) -> dict:
        truth_latest = {}
        for rec in self._truth:
            if rec['status'] == 'truth_active':
                truth_latest[rec['id']] = rec
        if not truth_latest:
            for src in CONFIG_B_SOURCES:
                truth_latest[src['name']] = {
                    'id': src['name'],
                    'x': src['xy'][0],
                    'y': src['xy'][1],
                    'status': 'truth_active',
                }

        confirmed = {}
        first_confirm_t = {}
        for rec in self._sources:
            if rec['status'] != 'confirmed':
                continue
            confirmed[rec['id']] = rec
            first_confirm_t.setdefault(rec['id'], rec['t'])

        matches = []
        used_est = set()
        for tid, truth in truth_latest.items():
            best_id = None
            best_d = float('inf')
            for eid, est in confirmed.items():
                if eid in used_est:
                    continue
                d = math.hypot(est['x'] - truth['x'], est['y'] - truth['y'])
                if d < best_d:
                    best_id = eid
                    best_d = d
            if best_id is not None and best_d <= 1.5:
                used_est.add(best_id)
                matches.append({
                    'truth_id': tid,
                    'estimate_id': best_id,
                    'error_m': round(best_d, 3),
                    'time_s': round(first_confirm_t.get(best_id, confirmed[best_id]['t']), 3),
                })

        truth_count = len(truth_latest)
        confirmed_count = len(confirmed)
        recall = len(matches) / max(1, truth_count)
        precision = len(matches) / max(1, confirmed_count)
        times = [m['time_s'] for m in matches]
        duplicate_confirmations = max(0, confirmed_count - len(matches))
        path_length = 0.0
        if len(self._traj) > 1:
            for a, b in zip(self._traj, self._traj[1:]):
                path_length += math.hypot(b['wx'] - a['wx'], b['wy'] - a['wy'])
        return {
            'source_recall': round(recall, 3),
            'source_precision': round(precision, 3),
            'time_to_first_source': min(times) if times else None,
            'time_to_all_sources': max(times) if len(matches) == truth_count and times else None,
            'localization_errors_m': matches,
            'duplicate_confirmations': duplicate_confirmations,
            'path_length_m': round(path_length, 3),
            'nav2_goal_proxy': {
                'plans_observed': self._nav2_plan_count,
                'accepted': None,
                'succeeded': None,
                'failed': None,
            },
            'duration_s': round(duration, 3),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 保存
    # ──────────────────────────────────────────────────────────────────────────

    def save(self):
        if self._saved:
            return
        self._saved = True
        out = self._out
        duration = self._ts()
        print(f'\n[collector v3] 正在保存数据，运行时长 {duration:.1f}s ...')

        def write_csv(name: str, rows: list):
            if not rows:
                print(f'  [SKIP] {name} (无数据)')
                return
            path = out / name
            all_keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
                w.writeheader()
                w.writerows(rows)
            print(f'  ✓ {name} ({len(rows)} 行)')

        write_csv('trajectory.csv',      self._traj)
        write_csv('slam_trajectory.csv', self._slam_traj)   # [v3]
        write_csv('thermal_stats.csv',   self._th_stats)
        write_csv('field_stats.csv',     self._field_stats)
        write_csv('thermal_map_stats.csv', self._map_stats)
        write_csv('gradient_stats.csv',  self._grad_stats)
        write_csv('source_estimates.csv', self._sources)
        write_csv('thermal_sources_truth.csv', self._truth)
        write_csv('source_events.csv', self._source_events)
        write_csv('cmd_vel.csv',         self._cmdvel)
        write_csv('scan_stats.csv',      self._scan_stats)  # [v3]
        write_csv('nav2_plan_stats.csv', self._plan_stats)  # [v3]

        # 话题频率统计
        rate_summary = {}
        for topic, times in self._rate_buf.items():
            if len(times) >= 2:
                diffs = np.diff(times)
                rate_summary[topic] = {
                    'mean_hz': round(float(1.0/np.mean(diffs)), 2),
                    'std_hz':  round(float(np.std(1.0/diffs)), 3),
                    'min_hz':  round(float(1.0/np.max(diffs)), 3),
                    'max_hz':  round(float(1.0/np.min(diffs)), 3),
                    'n_msgs':  len(times),
                }
            else:
                rate_summary[topic] = {'mean_hz': 0.0, 'n_msgs': len(times)}

        # [v3] 元数据（更新为 Config-B 源参数）
        meta = {
            'version':           'v3_slam_nav2',
            'recorded_at':       datetime.now().isoformat(),
            'duration_s':        round(duration, 2),
            'n_snapshots':       self._snap_count,
            'slam_available':    self._slam_available,
            'slam_first_t':      self._slam_first_t,
            'slam_pos_count':    self._slam_pos_count,
            'nav2_available':    self._nav2_available,
            'nav2_plan_count':   self._nav2_plan_count,
            'topic_rates':       rate_summary,
            'counts': {
                'trajectory':    len(self._traj),
                'slam_traj':     len(self._slam_traj),
                'thermal_stats': len(self._th_stats),
                'field_stats':   len(self._field_stats),
                'map_stats':     len(self._map_stats),
                'grad_stats':    len(self._grad_stats),
                'source_estimates': len(self._sources),
                'truth_sources':  len(self._truth),
                'source_events':  len(self._source_events),
                'cmd_vel':       len(self._cmdvel),
                'scan_stats':    len(self._scan_stats),
                'nav2_plans':    len(self._plan_stats),
            },
            # Config-B 热源参数（与 sensor_node.py v13 一致）
            'sources': CONFIG_B_SOURCES,
            'spawn':             {'x': SPAWN_X, 'y': SPAWN_Y},
            'arrival_radius_m':  0.5,
            'ambient_temp_c':    22.0,
            'config':            'Config-B',
        }
        with open(out / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)
        print('  ✓ metadata.json')
        summary = self._source_level_summary(duration)
        with open(out / 'source_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print('  ✓ source_summary.json')
        print(f'  ✓ snapshots/ ({self._snap_count} 对)')

        # [v3] 打印 SLAM/Nav2 汇总
        slam_first = f'{self._slam_first_t:.1f}s' if self._slam_first_t is not None else 'N/A'
        print(f'\n  [SLAM] 可用={self._slam_available} '
              f'首次就绪={slam_first} '
              f'位姿数={self._slam_pos_count}')
        print(f'  [Nav2] 可用={self._nav2_available} '
              f'规划次数={self._nav2_plan_count}')

        print(f'\n[collector v3] 数据已保存至：{out}')
        print('[collector v3] 下一步运行：')
        print(f'  python3 ~/ros2_ws/src/thermal_robot/scripts/plot_all_figures.py {out}')
        print(f'  python3 ~/ros2_ws/src/thermal_robot/scripts/plot_slam_nav2.py {out}')


# ─── 入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='', help='output directory; default is bags/collected/<timestamp>')
    parser.add_argument('--duration', type=float, default=0.0, help='seconds to collect before saving and exiting')
    args = parser.parse_args()

    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else OUTBASE / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[collector v3] 输出目录：{out_dir}')
    print(f'[collector v3] SLAM+Nav2 版：额外记录 /scan, /plan, /thermal/map, /thermal/sources, truth')

    rclpy.init()
    node = DataCollector(out_dir)

    def _shutdown(sig, frame):
        print('\n[collector v3] 收到 Ctrl+C，正在停止...')
        node.save()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        if args.duration > 0.0:
            deadline = time.monotonic() + args.duration
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
