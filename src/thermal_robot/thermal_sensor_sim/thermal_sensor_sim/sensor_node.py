#!/usr/bin/env python3
"""
sensor_node.py — 热传感器仿真节点 v13: Config-B通用性验证配置
============================================================
v13 Config-B 通用性验证配置：
  SA_left: world=(-1,3.5)  A=35°C σ=1.1m  peak≈57°C  d_spawn=6.1m
  SB_far:  world=(6,-3)    A=22°C σ=0.9m  peak≈44°C  d_spawn=12.4m（远距测试）
  SC_weak: world=(-5,-5.5) A=16°C σ=0.8m  peak≈38°C  d_spawn=5.6m（弱源，margin=1°C）

参数适配：sample_min_trise=8°C（SC_weak@0.75m trise=10.3°C，余量2.3°C）
所有源间距>7m，不存在路径阻塞（excl_r=2.0m 验证通过）。
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from thermal_interfaces.msg import SourceEstimate, SourceEstimateArray

from thermal_sensor_sim.scenario import (
    SENSOR_FOV_X,
    SENSOR_FOV_Y,
    SPAWN_X,
    SPAWN_Y,
    SourceState,
    apply_run_seed,
    default_config_b_scenario,
    load_scenario_file,
)


class SensorNode(Node):
    def __init__(self):
        super().__init__('sensor_node')

        self.declare_parameter('publish_rate',  10.0)
        self.declare_parameter('frame_id',      'thermal_camera')
        self.declare_parameter('image_width',   64)
        self.declare_parameter('image_height',  48)
        self.declare_parameter('ambient_temp',  22.0)
        self.declare_parameter('noise_std',     0.5)
        self.declare_parameter('random_seed',   42)
        self.declare_parameter('num_sources',   3)
        self.declare_parameter('scenario_file', '')
        self.declare_parameter('scenario_seed', 0)
        self.declare_parameter('scenario_jitter_std_m', 0.0)

        self._rate    = float(self.get_parameter('publish_rate').value)
        self._frame   = self.get_parameter('frame_id').value
        self._W       = int(self.get_parameter('image_width').value)
        self._H       = int(self.get_parameter('image_height').value)
        self._ambient = float(self.get_parameter('ambient_temp').value)
        self._noise   = float(self.get_parameter('noise_std').value)
        seed          = int(self.get_parameter('random_seed').value)
        n_src         = int(self.get_parameter('num_sources').value)
        scenario_file = str(self.get_parameter('scenario_file').value or '')
        scenario_seed = int(self.get_parameter('scenario_seed').value)
        jitter_std    = float(self.get_parameter('scenario_jitter_std_m').value)

        noise_seed = scenario_seed if scenario_seed > 0 else seed
        self._rng     = np.random.default_rng(noise_seed)
        if scenario_file:
            self._scenario = load_scenario_file(scenario_file, num_sources=0)
            self._scenario_label = scenario_file
        else:
            self._scenario = default_config_b_scenario(n_src)
            self._scenario_label = 'Config-B fallback'
        apply_run_seed(self._scenario, scenario_seed, jitter_std_m=jitter_std)
        self._sources = self._scenario.sources
        self._spawn_x = self._scenario.spawn_x
        self._spawn_y = self._scenario.spawn_y
        self._fov_x = self._scenario.fov_x
        self._fov_y = self._scenario.fov_y
        self._t0      = time.monotonic()

        self._odom_x   = 0.0
        self._odom_y   = 0.0
        self._odom_yaw = 0.0

        px_xs = np.linspace(-self._fov_x/2, self._fov_x/2, self._W, dtype=np.float32)
        px_ys = np.linspace(-self._fov_y/2, self._fov_y/2, self._H, dtype=np.float32)
        self._px_xx, self._px_yy = np.meshgrid(px_xs, px_ys)

        pub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=5,
            durability=QoSDurabilityPolicy.VOLATILE)
        odom_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=5,
            durability=QoSDurabilityPolicy.VOLATILE)

        self._pub       = self.create_publisher(Image, '/sim/thermal_raw', pub_qos)
        self._truth_pub = self.create_publisher(SourceEstimateArray, '/sim/thermal_sources_truth', pub_qos)
        self._sub       = self.create_subscription(Odometry, '/odom', self._odom_cb, odom_qos)
        self._timer     = self.create_timer(1.0 / self._rate, self._cb)

        T_init = self._ambient + sum(
            s.amplitude * math.exp(
                -((self._spawn_x - s.world_x)**2 + (self._spawn_y - s.world_y)**2)
                / (2 * s.sigma_m**2))
            for s in self._sources)
        self.get_logger().info(
            f'sensor_node v13 | {self._W}×{self._H} | {self._rate}Hz '
            f'| spawn=({self._spawn_x},{self._spawn_y}) | T_init={T_init:.1f}°C '
            f'| scenario={self._scenario_label} | sources={len(self._sources)} '
            f'| run_seed={scenario_seed}')

    def _odom_cb(self, msg: Odometry):
        self._odom_x = msg.pose.pose.position.x
        self._odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._odom_yaw = math.atan2(siny_cosp, cosy_cosp)

    def _world_field_at_sensor(self, t: float) -> np.ndarray:
        world_robot_x = self._spawn_x + self._odom_x
        world_robot_y = self._spawn_y + self._odom_y
        yaw = self._odom_yaw

        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        world_xs = world_robot_x + cos_y * self._px_xx - sin_y * self._px_yy
        world_ys = world_robot_y + sin_y * self._px_xx + cos_y * self._px_yy

        field = np.full((self._H, self._W), self._ambient, dtype=np.float32)
        for src in self._scenario.active_states(t):
            sx, sy = src.x, src.y
            dx = world_xs - sx
            dy = world_ys - sy
            field += (src.amplitude *
                      np.exp(-(dx**2 + dy**2) / (2.0 * src.sigma_m**2))
                      ).astype(np.float32)

        if self._noise > 0:
            field += self._rng.standard_normal(
                (self._H, self._W)).astype(np.float32) * self._noise
        return field

    def _cb(self):
        t   = time.monotonic() - self._t0
        arr = self._world_field_at_sensor(t)

        msg              = Image()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame
        msg.height = self._H
        msg.width  = self._W
        msg.encoding     = '32FC1'
        msg.is_bigendian = False
        msg.step         = self._W * 4
        msg.data         = arr.tobytes()
        self._pub.publish(msg)
        self._publish_truth(msg.header, t)

        if int(t * self._rate) % 50 == 0:
            wx = self._spawn_x + self._odom_x
            wy = self._spawn_y + self._odom_y
            self.get_logger().info(
                f't={t:.0f}s world=({wx:.2f},{wy:.2f}) '
                f'odom=({self._odom_x:.2f},{self._odom_y:.2f}) '
                f'T=[{arr.min():.1f},{arr.max():.1f}]°C')

    def _publish_truth(self, header, t: float):
        truth = SourceEstimateArray()
        truth.header = header
        truth.header.frame_id = 'world'
        for state in self._scenario.all_states(t):
            src = self._truth_msg_from_state(header, state)
            truth.sources.append(src)
        self._truth_pub.publish(truth)

    def _truth_msg_from_state(self, header, state: SourceState) -> SourceEstimate:
        msg = SourceEstimate()
        msg.header = header
        msg.header.frame_id = 'world'
        msg.id = state.source_id
        msg.status = 'truth_active' if state.active else 'truth_inactive'
        msg.position.x = float(state.x)
        msg.position.y = float(state.y)
        msg.position.z = 0.0
        msg.covariance_xx = 0.0
        msg.covariance_xy = 0.0
        msg.covariance_yy = 0.0
        msg.strength = float(state.amplitude)
        msg.sigma = float(state.sigma_m)
        msg.existence_probability = 1.0 if state.active else 0.0
        msg.confidence = 1.0
        msg.observations = 1
        return msg


def main(args=None):
    rclpy.init(args=args)
    n = SensorNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
