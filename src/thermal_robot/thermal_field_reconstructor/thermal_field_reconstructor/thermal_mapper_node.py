#!/usr/bin/env python3
"""thermal_mapper_node.py — fuse thermal images into a world-frame grid map."""

import math
import time

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from thermal_interfaces.msg import ThermalMap
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformListener

from thermal_field_reconstructor import visibility
from thermal_field_reconstructor.observation import SensorPose2D, TopDownRectProjector
from thermal_field_reconstructor.thermal_mapping import WorldThermalGrid
from thermal_field_reconstructor.perspective import CameraIntrinsics, SensorPose3D, PerspectiveProjector


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

        for name, default in [('sensor_model','a'),('camera_height_m',.6),
                              ('camera_pitch_rad',0.),('camera_yaw_rad',0.),('camera_roll_rad',0.),
                              ('camera_offset_x_m',0.),('camera_offset_y_m',0.),
                              ('camera_hfov_deg',57.),('depth_sync_tolerance_s',.06),
                              ('projection_near_m',.15),('projection_far_m',15.),
                              ('sector_memory_s',60.),('fusion_memory_s',1.e9),('pose_from_tf',False)]:
            self.declare_parameter(name,default)
        self._sensor_model=str(self.get_parameter('sensor_model').value)
        self._camera_info=None
        self._depth_frames=[]
        self._pending_images=[]
        self._last_image_stamp=-float('inf')
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
            fusion_memory_s=float(g('fusion_memory_s').value),
            sector_memory_s=float(g('sector_memory_s').value),
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
        self.create_subscription(Image, '/thermal/depth', self._depth_cb, be)
        self.create_subscription(CameraInfo, '/thermal/camera_info', self._camera_cb, be)
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
        if not visibility.occupancy_grid_has_known_cells(
                msg.data, msg.info.width, msg.info.height):
            if not getattr(self, '_empty_occ_warned', False):
                self._empty_occ_warned = True
                self.get_logger().warn(
                    f'[OCC_MAP_SKIP] unusable size={msg.info.width}x{msg.info.height} '
                    f'data={len(msg.data)} known=0; retaining previous valid map')
            return
        q=msg.info.origin.orientation
        origin_yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        view=visibility.from_flat(msg.data,msg.info.width,msg.info.height,
            msg.info.origin.position.x,msg.info.origin.position.y,msg.info.resolution,
            origin_yaw=origin_yaw)
        if self._pose_source=='odom':
            try:
                tf=self._tf_buffer.lookup_transform('odom',msg.header.frame_id,rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=.01))
                q=tf.transform.rotation;t=tf.transform.translation
                yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
                view=visibility.transform_view(view,t.x+self._spawn_x,t.y+self._spawn_y,yaw)
            except (LookupException,ExtrapolationException,ConnectivityException):
                return
        else:
            view=visibility.transform_view(view,self._spawn_x,self._spawn_y,0.)
        self._occ_view=view

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

    @staticmethod
    def _stamp(msg):
        return msg.header.stamp.sec+msg.header.stamp.nanosec/1e9

    def _camera_cb(self,msg):
        self._camera_info=CameraIntrinsics(msg.width,msg.height,msg.k[0],msg.k[4],msg.k[2],msg.k[5])

    def _depth_cb(self,msg):
        if msg.encoding!='32FC1':
            return
        self._depth_frames.append(msg)
        self._depth_frames=self._depth_frames[-20:]
        pending=self._pending_images
        self._pending_images=[]
        for image in pending: self._image_cb(image)

    def _image_cb(self, msg: Image):
        if msg.encoding!='32FC1':
            self.get_logger().error('thermal input must be Celsius 32FC1')
            return
        stamp=self._stamp(msg)
        if stamp<=self._last_image_stamp: return
        endian='>f4' if msg.is_bigendian else '<f4'
        arr=np.frombuffer(bytes(msg.data),dtype=endian).reshape(msg.height,msg.step//4)[:,:msg.width].copy()
        self._update_pose()
        # Keep grid time in the image clock, so replay uses recorded chronology.
        if not hasattr(self,"_image_time_origin"): self._image_time_origin=stamp
        now_s=stamp-self._image_time_origin
        if self._sensor_model=='a':
            pose=SensorPose2D(x=self._wx,y=self._wy,yaw=self._yaw,frame_id=self._frame_id)
            obs=self._projector.project(arr,pose,now_s)
        else:
            if not self._depth_frames:
                self._pending_images=(self._pending_images+[msg])[-10:]
                return
            depth_msg=min(self._depth_frames,key=lambda d:abs(self._stamp(d)-stamp))
            if abs(self._stamp(depth_msg)-stamp)>float(self.get_parameter('depth_sync_tolerance_s').value):
                self._pending_images=(self._pending_images+[msg])[-10:]
                return
            if depth_msg.header.frame_id!=msg.header.frame_id:
                self.get_logger().warn('[PROJECTION] depth must be registered into the thermal optical frame')
                return
            if bool(self.get_parameter('pose_from_tf').value) and self._camera_info is None:
                # Hardware projection requires measured calibration, not the
                # nominal simulation field of view.
                return
            depth=np.frombuffer(bytes(depth_msg.data),dtype='>f4' if depth_msg.is_bigendian else '<f4')
            depth=depth.reshape(depth_msg.height,depth_msg.step//4)[:,:depth_msg.width]
            k=self._camera_info or CameraIntrinsics.from_hfov(msg.width,msg.height,
                float(self.get_parameter('camera_hfov_deg').value))
            g=lambda name:float(self.get_parameter(name).value)
            ox,oy=g('camera_offset_x_m'),g('camera_offset_y_m')
            pose=SensorPose3D(self._wx+math.cos(self._yaw)*ox-math.sin(self._yaw)*oy,
                self._wy+math.sin(self._yaw)*ox+math.cos(self._yaw)*oy,g('camera_height_m'),
                self._yaw+g('camera_yaw_rad'),g('camera_pitch_rad'),g('camera_roll_rad'),self._frame_id)
            if bool(self.get_parameter('pose_from_tf').value):
                try:
                    tf=self._tf_buffer.lookup_transform(self._frame_id,msg.header.frame_id,
                        rclpy.time.Time.from_msg(msg.header.stamp),
                        timeout=rclpy.duration.Duration(seconds=.02))
                    # TF is optical -> world; recover robot-convention RPY.
                    q=tf.transform.rotation
                    optical=np.array([[1-2*(q.y*q.y+q.z*q.z),2*(q.x*q.y-q.z*q.w),2*(q.x*q.z+q.y*q.w)],
                        [2*(q.x*q.y+q.z*q.w),1-2*(q.x*q.x+q.z*q.z),2*(q.y*q.z-q.x*q.w)],
                        [2*(q.x*q.z-q.y*q.w),2*(q.y*q.z+q.x*q.w),1-2*(q.x*q.x+q.y*q.y)]])
                    r=optical@np.array([[0,0,1],[-1,0,0],[0,-1,0]]).T
                    t=tf.transform.translation
                    pose=SensorPose3D(t.x,t.y,t.z,math.atan2(r[1,0],r[0,0]),
                        math.asin(float(np.clip(-r[2,0],-1,1))),math.atan2(r[2,1],r[2,2]),self._frame_id)
                except (LookupException,ExtrapolationException,ConnectivityException):
                    return
            try:
                obs=PerspectiveProjector(k,g('projection_near_m'),g('projection_far_m')).project(arr,depth,pose,now_s)
            except ValueError as exc:
                self.get_logger().warn(f'[PROJECTION] {exc}')
                return
        self._last_image_stamp=stamp
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
        msg.measurement_type = 'field_direct' if self._sensor_model=='a' else 'surface_radiance'
        msg.last_view_distance_m=snap.last_view_distance_m.reshape(-1).tolist()
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
