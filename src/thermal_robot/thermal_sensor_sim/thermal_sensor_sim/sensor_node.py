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
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

SPAWN_X = -6.0
SPAWN_Y =  0.0

@dataclass
class WorldHeatSource:
    world_x: float
    world_y: float
    amplitude: float
    sigma_m: float

    def position(self, t: float) -> Tuple[float, float]:
        return (self.world_x, self.world_y)


# ★ Config-B: 通用性验证配置（不同空间分布 + 弱源测试）★
# 测试目标：
#   SA_left:  中偏上，强源，peak=57°C（验证梯度上升精度）
#   SB_far:   右下，中源，d_spawn=12.4m，peak=44°C（验证远距发现能力）
#   SC_weak:  左下近spawn，弱源，peak=38°C，sigma=0.8m（验证弱源检测）
# 所有源间距>7m（远超excl_r=2.0m），不存在路径阻塞问题
WORLD_SOURCES = [
    WorldHeatSource(world_x=-1.0, world_y= 3.5, amplitude=35.0, sigma_m=1.1),  # SA_left  peak≈57°C
    WorldHeatSource(world_x= 6.0, world_y=-3.0, amplitude=22.0, sigma_m=0.9),  # SB_far   peak≈44°C
    WorldHeatSource(world_x=-5.0, world_y=-5.5, amplitude=16.0, sigma_m=0.8),  # SC_weak  peak≈38°C
]

SENSOR_FOV_X = 4.0
SENSOR_FOV_Y = 3.0


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

        self._rate    = float(self.get_parameter('publish_rate').value)
        self._frame   = self.get_parameter('frame_id').value
        self._W       = int(self.get_parameter('image_width').value)
        self._H       = int(self.get_parameter('image_height').value)
        self._ambient = float(self.get_parameter('ambient_temp').value)
        self._noise   = float(self.get_parameter('noise_std').value)
        seed          = int(self.get_parameter('random_seed').value)
        n_src         = int(self.get_parameter('num_sources').value)

        self._rng     = np.random.default_rng(seed)
        self._sources = WORLD_SOURCES[:max(1, min(n_src, len(WORLD_SOURCES)))]
        self._t0      = time.monotonic()

        self._odom_x   = 0.0
        self._odom_y   = 0.0
        self._odom_yaw = 0.0

        px_xs = np.linspace(-SENSOR_FOV_X/2, SENSOR_FOV_X/2, self._W, dtype=np.float32)
        px_ys = np.linspace(-SENSOR_FOV_Y/2, SENSOR_FOV_Y/2, self._H, dtype=np.float32)
        self._px_xx, self._px_yy = np.meshgrid(px_xs, px_ys)

        pub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=5,
            durability=QoSDurabilityPolicy.VOLATILE)
        odom_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=5,
            durability=QoSDurabilityPolicy.VOLATILE)

        self._pub   = self.create_publisher(Image, '/sim/thermal_raw', pub_qos)
        self._sub   = self.create_subscription(Odometry, '/odom', self._odom_cb, odom_qos)
        self._timer = self.create_timer(1.0 / self._rate, self._cb)

        T_init = self._ambient + sum(
            s.amplitude * math.exp(
                -((SPAWN_X - s.world_x)**2 + (SPAWN_Y - s.world_y)**2)
                / (2 * s.sigma_m**2))
            for s in self._sources)
        self.get_logger().info(
            f'sensor_node v13 | {self._W}×{self._H} | {self._rate}Hz '
            f'| spawn=({SPAWN_X},{SPAWN_Y}) | T_init={T_init:.1f}°C '
            f'| SA@(-1,3.5) A=35 | SB@(6,-3) A=22 | SC@(-5,-5.5) A=16')

    def _odom_cb(self, msg: Odometry):
        self._odom_x = msg.pose.pose.position.x
        self._odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._odom_yaw = math.atan2(siny_cosp, cosy_cosp)

    def _world_field_at_sensor(self, t: float) -> np.ndarray:
        world_robot_x = SPAWN_X + self._odom_x
        world_robot_y = SPAWN_Y + self._odom_y
        yaw = self._odom_yaw

        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        world_xs = world_robot_x + cos_y * self._px_xx - sin_y * self._px_yy
        world_ys = world_robot_y + sin_y * self._px_xx + cos_y * self._px_yy

        field = np.full((self._H, self._W), self._ambient, dtype=np.float32)
        for src in self._sources:
            sx, sy = src.position(t)
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

        if int(t * self._rate) % 50 == 0:
            wx = SPAWN_X + self._odom_x
            wy = SPAWN_Y + self._odom_y
            self.get_logger().info(
                f't={t:.0f}s world=({wx:.2f},{wy:.2f}) '
                f'odom=({self._odom_x:.2f},{self._odom_y:.2f}) '
                f'T=[{arr.min():.1f},{arr.max():.1f}]°C')


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
