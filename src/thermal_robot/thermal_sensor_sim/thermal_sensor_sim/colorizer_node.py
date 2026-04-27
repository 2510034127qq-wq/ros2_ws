"""
colorizer_node.py  —  32FC1 → RGB8 (inferno) converter
========================================================
Subscribes : /sim/thermal_raw       sensor_msgs/Image  32FC1  BEST_EFFORT
Publishes  : /sim/thermal_colorized sensor_msgs/Image  rgb8   BEST_EFFORT

QoS FIX: sensor_node publishes with BEST_EFFORT reliability.
This node must subscribe with BEST_EFFORT to avoid the QoS mismatch warning
  "New publisher discovered on topic '/sim/thermal_raw',
   offering incompatible QoS. No messages will be received."

Parameters
----------
t_min : float  lower clamp  (default 22.0 °C)
t_max : float  upper clamp  (default 65.0 °C)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

import matplotlib
matplotlib.use('Agg')
from matplotlib import colormaps
_CMAP = colormaps['inferno']

# Match sensor_node QoS: BEST_EFFORT + KEEP_LAST(10)
_QOS_SENSOR = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


class ColorizerNode(Node):
    def __init__(self):
        super().__init__('colorizer_node')
        self.declare_parameter('t_min', 22.0)
        self.declare_parameter('t_max', 65.0)

        # Publish with same BEST_EFFORT QoS so downstream tools match easily
        self._pub = self.create_publisher(Image, '/sim/thermal_colorized', _QOS_SENSOR)
        self._sub = self.create_subscription(
            Image, '/sim/thermal_raw', self._cb, _QOS_SENSOR)

        t_min = self.get_parameter('t_min').value
        t_max = self.get_parameter('t_max').value
        self.get_logger().info(
            f'colorizer_node | 32FC1→RGB8 | inferno | '
            f't=[{t_min}, {t_max}]°C | QoS=BEST_EFFORT')

    def _cb(self, msg: Image):
        t_min = self.get_parameter('t_min').value
        t_max = self.get_parameter('t_max').value

        raw  = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        norm = np.clip((raw - t_min) / (t_max - t_min + 1e-9), 0.0, 1.0)
        rgb  = (_CMAP(norm)[:, :, :3] * 255).astype(np.uint8)

        out = Image()
        out.header       = msg.header
        out.height       = msg.height
        out.width        = msg.width
        out.encoding     = 'rgb8'
        out.is_bigendian = 0
        out.step         = msg.width * 3
        out.data         = rgb.tobytes()
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ColorizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
