#!/usr/bin/env python3
"""reconstructor_node.py — Thermal Field Reconstructor Node (Route A).

Sub     : /thermal/filtered          sensor_msgs/Image 32FC1
Pub     : /thermal/field             thermal_interfaces/ThermalField
Service : /thermal/get_field_info    thermal_interfaces/GetFieldInfo

Hotspot detection: NMS on pixels above (mean + hotspot_min_temp_delta).
Ref: Reggente & Lilienthal 2009 DOI:10.1109/ICSENS.2009.5398427
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from thermal_interfaces.msg import ThermalField, ThermalPoint
from thermal_interfaces.srv import GetFieldInfo


class ReconstructorNode(Node):
    def __init__(self):
        super().__init__('reconstructor_node')
        self.declare_parameter('ambient_temp',22.0)
        self.declare_parameter('reconstruction_method',  'direct_linear')
        self.declare_parameter('frame_id',               'thermal_camera')
        self.declare_parameter('resolution_x',           0.01)
        self.declare_parameter('resolution_y',           0.01)
        self.declare_parameter('origin_x',               0.0)
        self.declare_parameter('origin_y',               0.0)
        self.declare_parameter('hotspot_min_temp_delta', 5.0)
        self.declare_parameter('hotspot_max_count',      5)
        self.declare_parameter('hotspot_min_distance',   5)

        self._method   = self.get_parameter('reconstruction_method').value
        self._frame    = self.get_parameter('frame_id').value
        self._rx       = float(self.get_parameter('resolution_x').value)
        self._ry       = float(self.get_parameter('resolution_y').value)
        self._ox       = float(self.get_parameter('origin_x').value)
        self._oy       = float(self.get_parameter('origin_y').value)
        self._hs_dt    = float(self.get_parameter('hotspot_min_temp_delta').value)
        self._hs_max   = int(self.get_parameter('hotspot_max_count').value)
        self._hs_dist  = int(self.get_parameter('hotspot_min_distance').value)
        self._last     = None

        sub_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                             history=QoSHistoryPolicy.KEEP_LAST, depth=5,
                             durability=QoSDurabilityPolicy.VOLATILE)
        pub_qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                             history=QoSHistoryPolicy.KEEP_LAST, depth=5,
                             durability=QoSDurabilityPolicy.VOLATILE)
        self._sub = self.create_subscription(Image, '/thermal/filtered', self._cb, sub_qos)
        self._pub = self.create_publisher(ThermalField, '/thermal/field', pub_qos)
        self._srv = self.create_service(GetFieldInfo, '/thermal/get_field_info', self._srv_cb)
        self.get_logger().info(f'reconstructor_node | method={self._method}')

    def _hotspots(self, arr, mean_t):
        thr = mean_t + self._hs_dt
        arr_max = float(arr.max())
        # temp_range: span from detection threshold to global maximum.
        # Used to normalise confidence: 0 at threshold, 1 at peak.
        temp_range = arr_max - thr
        ys, xs = np.where(arr > thr)
        if len(xs) == 0: return []
        temps = arr[ys, xs]
        order = np.argsort(temps)[::-1]
        ys, xs, temps = ys[order], xs[order], temps[order]
        accepted = []
        for i in range(len(xs)):
            cx, cy, ct = float(xs[i]), float(ys[i]), float(temps[i])
            if any(((cx-p.pixel_x)**2+(cy-p.pixel_y)**2)**0.5 < self._hs_dist
                   for p in accepted):
                continue
            tp = ThermalPoint()
            tp.pixel_x = cx; tp.pixel_y = cy
            tp.temperature_celsius = ct
            tp.intensity   = float(ct / (arr_max + 1e-9))
            # Confidence: how far above the detection threshold this hotspot sits,
            # relative to the full above-threshold range.  Replaces the previous
            # hardcoded 1.0 which made the field meaningless for downstream consumers.
            tp.confidence  = float(np.clip((ct - thr) / (temp_range + 1e-9), 0.0, 1.0))
            accepted.append(tp)
            if len(accepted) >= self._hs_max: break
        return accepted

    def _cb(self, msg):
        n = msg.width * msg.height
        arr = np.frombuffer(bytes(msg.data[:n*4]), np.float32).reshape(
            msg.height, msg.width).copy()

        arr=np.nan_to_num(arr,nan=float(self.get_parameter('ambient_temp').value),posinf=float(self.get_parameter('ambient_temp').value),neginf=float(self.get_parameter('ambient_temp').value))
        t_min  = float(arr.min());  t_max  = float(arr.max())
        t_mean = float(arr.mean()); t_std  = float(arr.std())

        tf = ThermalField()
        tf.header = msg.header
        if not tf.header.frame_id: tf.header.frame_id = self._frame
        tf.width  = msg.width;  tf.height = msg.height
        tf.resolution_x = self._rx; tf.resolution_y = self._ry
        tf.origin_x     = self._ox; tf.origin_y     = self._oy
        tf.data = arr.flatten().tolist()
        tf.min_temperature_celsius  = t_min
        tf.max_temperature_celsius  = t_max
        tf.mean_temperature_celsius = t_mean
        tf.std_temperature_celsius  = t_std
        tf.hotspots = self._hotspots(arr, t_mean)
        tf.reconstruction_method = self._method
        tf.source_encoding = msg.encoding
        self._last = tf
        self._pub.publish(tf)

    def _srv_cb(self, req, res):
        if self._last is None:
            res.success = False; res.message = 'No data yet'; return res
        f = self._last
        res.success = True; res.message = 'OK'
        res.width   = f.width; res.height = f.height
        res.min_temperature_celsius  = f.min_temperature_celsius
        res.max_temperature_celsius  = f.max_temperature_celsius
        res.mean_temperature_celsius = f.mean_temperature_celsius
        res.std_temperature_celsius  = f.std_temperature_celsius
        res.hotspot_count = len(f.hotspots)
        if req.include_full_data:
            res.hotspot_temperatures = [h.temperature_celsius for h in f.hotspots]
        res.reconstruction_method = f.reconstruction_method
        res.last_update_stamp = f.header.stamp
        return res


def main(args=None):
    rclpy.init(args=args)
    n = ReconstructorNode()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    finally:
        n.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__': main()
