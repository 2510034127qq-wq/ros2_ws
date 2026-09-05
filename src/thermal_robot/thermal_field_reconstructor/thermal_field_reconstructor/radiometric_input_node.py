#!/usr/bin/env python3
"""ROS radiometric input or Linux Y16 UVC capture, with explicit temperature units."""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from .perspective import radiometric_celsius


class RadiometricInputNode(Node):
    def __init__(self):
        super().__init__('radiometric_input_node')
        defaults={'input_mode':'ros','device':'/dev/video0','frame_id':'thermal_optical_frame',
            'width':160,'height':120,'rate_hz':8.6,'kelvin_scale':.01,'kelvin_offset':0.,
            'raw_units':'tlinear','min_c':-40.,'max_c':400.}
        for k,v in defaults.items():self.declare_parameter(k,v)
        self.g=lambda name:self.get_parameter(name).value
        if self.g('raw_units') not in ('tlinear','celsius'):
            raise ValueError('input must be calibrated TLinear or Celsius; raw DN is unsupported')
        self.pub=self.create_publisher(Image,'/thermal/raw',qos_profile_sensor_data)
        self.capture=None
        if self.g('input_mode')=='ros':
            self.create_subscription(Image,'/thermal/radiometric_input',self._image,qos_profile_sensor_data)
        elif self.g('input_mode')=='uvc':
            import cv2
            self.capture=cv2.VideoCapture(str(self.g('device')),cv2.CAP_V4L2)
            self.capture.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('Y','1','6',' '))
            self.capture.set(cv2.CAP_PROP_CONVERT_RGB,0)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,int(self.g('width')))
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,int(self.g('height')))
            if not self.capture.isOpened():raise RuntimeError('cannot open Y16 UVC device')
            self.create_timer(1/float(self.g('rate_hz')),self._capture)
        else: raise ValueError('input_mode must be ros or uvc')

    def _publish(self,array,header):
        arr=np.asarray(array)
        if self.g('raw_units')=='tlinear':
            arr=radiometric_celsius(arr,float(self.g('kelvin_scale')),float(self.g('kelvin_offset')))
        else: arr=arr.astype(np.float32)
        valid=np.isfinite(arr)&(arr>=float(self.g('min_c')))&(arr<=float(self.g('max_c')))
        arr=np.where(valid,arr,np.nan).astype(np.float32)
        out=Image();out.header=header;out.height,out.width=arr.shape
        out.encoding='32FC1';out.step=out.width*4;out.data=arr.tobytes()
        self.pub.publish(out)

    def _image(self,msg):
        expected='32FC1' if self.g('raw_units')=='celsius' else '16UC1'
        if msg.encoding not in (expected,'mono16' if expected=='16UC1' else expected):
            self.get_logger().error(f'expected {expected}, received {msg.encoding}; image rejected')
            return
        dtype=np.dtype(('>' if msg.is_bigendian else '<')+('f4' if expected=='32FC1' else 'u2'))
        arr=np.frombuffer(bytes(msg.data),dtype=dtype).reshape(msg.height,msg.step//dtype.itemsize)[:,:msg.width]
        self._publish(arr,msg.header)

    def _capture(self):
        ok,arr=self.capture.read()
        if not ok:return
        if arr.dtype!=np.uint16 or arr.ndim!=2:
            self.get_logger().error('UVC returned non-Y16 image; refusing pseudocolor conversion')
            return
        from std_msgs.msg import Header
        header=Header();header.stamp=self.get_clock().now().to_msg();header.frame_id=str(self.g('frame_id'))
        self._publish(arr,header)

    def destroy_node(self):
        if self.capture is not None:self.capture.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args);node=RadiometricInputNode()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
