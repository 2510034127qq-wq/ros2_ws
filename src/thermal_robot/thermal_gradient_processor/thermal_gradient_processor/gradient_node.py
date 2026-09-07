#!/usr/bin/env python3
"""gradient_node.py — Thermal Gradient Processor Node v2.

Changes from v1:
  - peak_temperature_celsius now reports arr.max() (hottest pixel in FOV),
    not arr[py, px] (temperature at max-gradient pixel, which is at the
    plume edge ≈ σ from center). The old behaviour caused the controller's
    peak detector to fire at plume-edge temperature (~35 °C for S0) rather
    than at source-centre temperature, so 'FOUND' was declared 3–4 m away.
  - Added peak_pixel_x/y referring to the hottest pixel (consistent with
    the corrected peak_temperature_celsius).

Sub : /thermal/field     thermal_interfaces/ThermalField
Pub : /thermal/gradient  thermal_interfaces/GradientArray  10 Hz
method param: "sobel" (default) | "central_diff"

Sobel kernels (Wiedemann 2021 DOI:10.1016/j.robot.2020.103687):
  Gx = [[-1,0,1],[-2,0,2],[-1,0,1]] / 8
  Gy = [[-1,-2,-1],[0,0,0],[1,2,1]] / 8
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)
from thermal_interfaces.msg import ThermalField, GradientArray, Gradient


class GradientNode(Node):
    def __init__(self):
        super().__init__('gradient_node')
        self.declare_parameter('method',           'sobel')
        self.declare_parameter('subsample_stride', 4)

        self._method  = self.get_parameter('method').value
        self._stride  = int(self.get_parameter('subsample_stride').value)

        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=5,
                         durability=QoSDurabilityPolicy.VOLATILE)
        self._sub = self.create_subscription(ThermalField, '/thermal/field', self._cb, qos)
        self._pub = self.create_publisher(GradientArray, '/thermal/gradient', qos)
        self.get_logger().info(
            f'gradient_node v2 | method={self._method} stride={self._stride} '
            f'| peak_temp=arr.max()')

    def _sobel(self, arr):
        p = np.pad(arr, 1, mode='edge')
        gx = ((-p[:-2, :-2] + p[:-2, 2:] - 2*p[1:-1, :-2] + 2*p[1:-1, 2:]
               - p[2:,  :-2] + p[2:,  2:]) / 8.0)
        gy = ((-p[:-2, :-2] - 2*p[:-2, 1:-1] - p[:-2, 2:]
               + p[2:,  :-2] + 2*p[2:,  1:-1] + p[2:,  2:]) / 8.0)
        return gx.astype(np.float32), gy.astype(np.float32)

    def _central_diff(self, arr):
        p  = np.pad(arr, 1, mode='edge')
        gx = ((p[1:-1, 2:] - p[1:-1, :-2]) / 2.0).astype(np.float32)
        gy = ((p[2:,  1:-1] - p[:-2, 1:-1]) / 2.0).astype(np.float32)
        return gx, gy

    def _cb(self, msg):
        H, W = msg.height, msg.width
        arr  = np.array(msg.data, dtype=np.float32).reshape(H, W)

        if self._method == 'central_diff':
            gx, gy = self._central_diff(arr)
        else:
            gx, gy = self._sobel(arr)

        mag = np.sqrt(gx**2 + gy**2)
        di  = np.arctan2(gy, gx)

        # Sub-sample for the published gradient array
        ys = np.arange(0, H, self._stride)
        xs = np.arange(0, W, self._stride)

        grads = []
        for y in ys:
            for x in xs:
                g = Gradient()
                g.header              = msg.header
                g.pixel_x             = float(x)
                g.pixel_y             = float(y)
                g.grad_x              = float(gx[y, x])
                g.grad_y              = float(gy[y, x])
                g.magnitude           = float(mag[y, x])
                g.direction_rad       = float(di[y, x])
                g.temperature_celsius = float(arr[y, x])
                grads.append(g)

        # ── v2 fix: hottest pixel location (not max-gradient pixel) ──────────
        hot_idx = int(np.argmax(arr))
        hot_py, hot_px = divmod(hot_idx, W)

        ga = GradientArray()
        ga.header                   = msg.header
        ga.gradients                = grads
        ga.max_magnitude            = float(mag.max())
        ga.peak_pixel_x             = uint32_safe(hot_px)
        ga.peak_pixel_y             = uint32_safe(hot_py)
        # ★ v2 fix: report maximum temperature in FOV, not edge temperature ★
        ga.peak_temperature_celsius = float(arr.max())
        ga.width = W
        ga.height = H
        ga.method                   = self._method
        self._pub.publish(ga)


def uint32_safe(v):
    return max(0, int(v))


def main(args=None):
    rclpy.init(args=args)
    n = GradientNode()
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
