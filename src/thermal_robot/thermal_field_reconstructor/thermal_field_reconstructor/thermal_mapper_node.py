#!/usr/bin/env python3
"""thermal_mapper_node.py — fuse thermal images into a world-frame grid map."""

import math
import time

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from thermal_interfaces.msg import ThermalMap
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformListener

from thermal_field_reconstructor import visibility
from thermal_field_reconstructor.observation import SensorPose2D, TopDownRectProjector
from thermal_field_reconstructor.thermal_mapping import WorldThermalGrid


class ThermalMapperNode(Node):
    def __init__(self):
        super().__init__('thermal_mapper_node')
        self.declare_parameter('publish_rate', 5.0)
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('pose_source', 'odom')
        self.declare_parameter('spawn_x', -6.0)
        self.declare_parameter('spawn_y', 0.0)
        self.declare_parameter('map_size_x_m', 50.0)
        self.declare_parameter('map_size_y_m', 50.0)
        self.declare_parameter('map_resolution', 0.25)
        self.declare_parameter('sensor_fov_x', 4.0)
        self.declare_parameter('sensor_fov_y', 3.0)
        self.declare_parameter('ambient_temp', 22.0)
        self.declare_parameter('confidence_visit_scale', 6.0)
        self.declare_parameter('age_decay_s', 45.0)
        self.declare_parameter('unknown_variance', 100.0)
        self.declare_parameter('visibility_enabled', True)
        self.declare_parameter('occupied_threshold', 65)
        self.declare_parameter('visibility_ray_step_m', 0.1)

        g = self.get_parameter
        self._publish_rate = float(g('publish_rate').value)
        self._frame_id = str(g('frame_id').value)
        self._pose_source = str(g('pose_source').value).lower()
        self._spawn_x = float(g('spawn_x').value)
        self._spawn_y = float(g('spawn_y').value)
        self._fov_x = float(g('sensor_fov_x').value)
        self._fov_y = float(g('sensor_fov_y').value)
        self._visibility_enabled = bool(g('visibility_enabled').value)
        self._occupied_threshold = int(g('occupied_threshold').value)
        self._ray_step = float(g('visibility_ray_step_m').value)
        self._projector = TopDownRectProjector(fov_x=self._fov_x, fov_y=self._fov_y)
        self._occ_view = None
        self._fuse_count = 0
        self._grid = WorldThermalGrid(
            center_x=self._spawn_x,
            center_y=self._spawn_y,
            size_x_m=float(g('map_size_x_m').value),
            size_y_m=float(g('map_size_y_m').value),
            resolution=float(g('map_resolution').value),
            ambient_temp=float(g('ambient_temp').value),
            confidence_visit_scale=float(g('confidence_visit_scale').value),
            age_decay_s=float(g('age_decay_s').value),
            unknown_variance=float(g('unknown_variance').value),
        )

        self._odom_x = 0.0
        self._odom_y = 0.0
        self._yaw = 0.0
        self._wx = self._spawn_x
        self._wy = self._spawn_y
        self._tf_ready = False
        self._last_pub_s = 0.0
        self._t0 = time.monotonic()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        be = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        rel = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=3,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Image, '/thermal/filtered', self._image_cb, be)
        self.create_subscription(Odometry, '/odom', self._odom_cb, be)
        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, '/map', self._slam_map_cb, map_qos)
        self._pub = self.create_publisher(ThermalMap, '/thermal/map', rel)
        self.get_logger().info(
            f'thermal_mapper_node | grid={self._grid.width}x{self._grid.height} '
            f'res={self._grid.resolution:.2f}m frame={self._frame_id} '
            f'pose_source={self._pose_source} '
            f'spawn=({self._spawn_x:.1f},{self._spawn_y:.1f})')

    def _odom_cb(self, msg: Odometry):
        self._odom_x = msg.pose.pose.position.x
        self._odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._yaw = math.atan2(siny, cosy)
        if not self._tf_ready:
            self._wx = self._spawn_x + self._odom_x
            self._wy = self._spawn_y + self._odom_y

    def _slam_map_cb(self, msg: OccupancyGrid):
        self._occ_view = visibility.from_flat(
            msg.data, msg.info.width, msg.info.height,
            msg.info.origin.position.x + self._spawn_x,
            msg.info.origin.position.y + self._spawn_y,
            msg.info.resolution,
            occupied_threshold=self._occupied_threshold)

    def _update_pose(self):
        if self._pose_source == 'odom':
            self._wx = self._spawn_x + self._odom_x
            self._wy = self._spawn_y + self._odom_y
            return
        try:
            tf = self._tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.03),
            )
            self._wx = self._spawn_x + tf.transform.translation.x
            self._wy = self._spawn_y + tf.transform.translation.y
            q = tf.transform.rotation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self._yaw = math.atan2(siny, cosy)
            if not self._tf_ready:
                self._tf_ready = True
                self.get_logger().info(
                    f'[MAPPER_TF_READY] world=({self._wx:.2f},{self._wy:.2f})')
        except (LookupException, ExtrapolationException, ConnectivityException):
            self._wx = self._spawn_x + self._odom_x
            self._wy = self._spawn_y + self._odom_y

    def _image_cb(self, msg: Image):
        n = msg.width * msg.height
        arr = np.frombuffer(bytes(msg.data[:n * 4]), np.float32).reshape(msg.height, msg.width).copy()
        self._update_pose()
        now_s = time.monotonic() - self._t0
        pose = SensorPose2D(x=self._wx, y=self._wy, yaw=self._yaw)
        obs = self._projector.project(arr, pose, now_s)
        occ = self._occ_view if self._visibility_enabled else None
        t0 = time.monotonic()
        self._grid.integrate_observation(obs, occupancy=occ, ray_step_m=self._ray_step)
        fuse_ms = (time.monotonic() - t0) * 1000.0
        self._fuse_count += 1
        if self._fuse_count % 100 == 0:
            blocked_total = int((self._grid.blocked_count > 0).sum())
            clear_total = int((self._grid.visit_count > 0).sum())
            self.get_logger().info(
                f'[FUSE] n={self._fuse_count} {fuse_ms:.1f}ms '
                f'occ_map={"yes" if occ is not None else "no"} '
                f'cells_clear={clear_total} cells_blocked_only={blocked_total}')
        if now_s - self._last_pub_s >= 1.0 / max(self._publish_rate, 0.1):
            self._last_pub_s = now_s
            self._publish_map(msg.header, now_s)

    def _publish_map(self, header, now_s: float):
        snap = self._grid.snapshot(now_s)
        msg = ThermalMap()
        msg.header = header
        msg.header.frame_id = self._frame_id
        msg.width = snap.width
        msg.height = snap.height
        msg.resolution = float(snap.resolution)
        msg.origin_x = float(snap.origin_x)
        msg.origin_y = float(snap.origin_y)
        msg.temperature_mean = snap.temperature_mean.reshape(-1).astype(np.float32).tolist()
        msg.temperature_variance = snap.temperature_variance.reshape(-1).astype(np.float32).tolist()
        msg.confidence = snap.confidence.reshape(-1).astype(np.float32).tolist()
        msg.visit_count = snap.visit_count.reshape(-1).astype(np.uint32).tolist()
        msg.last_seen_age_s = snap.last_seen_age_s.reshape(-1).astype(np.float32).tolist()
        msg.view_state = snap.view_state.reshape(-1).astype(np.uint8).tolist()
        msg.view_sectors = snap.view_sectors.reshape(-1).astype(np.uint8).tolist()
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ThermalMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
