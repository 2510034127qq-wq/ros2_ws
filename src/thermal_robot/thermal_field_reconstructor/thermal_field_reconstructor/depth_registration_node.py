#!/usr/bin/env python3
"""Calibrated optical depth registration for thermal observations and bag replay."""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image,CameraInfo
from tf2_ros import Buffer,TransformListener,TransformException
from .perspective import CameraIntrinsics,register_depth


class DepthRegistrationNode(Node):
    def __init__(self):
        super().__init__('depth_registration_node')
        self.declare_parameter('thermal_frame','thermal_optical_frame')
        self.buffer=Buffer();self.listener=TransformListener(self.buffer,self)
        self.dk=self.tk=None
        self.create_subscription(CameraInfo,'/depth/camera_info',self._depth_info,qos_profile_sensor_data)
        self.create_subscription(CameraInfo,'/thermal/camera_info',self._thermal_info,qos_profile_sensor_data)
        self.create_subscription(Image,'/depth/image',self._depth,qos_profile_sensor_data)
        self.pub=self.create_publisher(Image,'/thermal/depth',qos_profile_sensor_data)

    @staticmethod
    def intrinsics(msg):
        return CameraIntrinsics(msg.width,msg.height,msg.k[0],msg.k[4],msg.k[2],msg.k[5])

    def _depth_info(self,msg):self.dk=self.intrinsics(msg)
    def _thermal_info(self,msg):self.tk=self.intrinsics(msg)

    def _depth(self,msg):
        if self.dk is None or self.tk is None:return
        if msg.encoding not in ('32FC1','16UC1'):return
        dtype=np.dtype(('>' if msg.is_bigendian else '<')+('f4' if msg.encoding=='32FC1' else 'u2'))
        depth=np.frombuffer(bytes(msg.data),dtype=dtype).reshape(msg.height,msg.step//dtype.itemsize)[:,:msg.width].astype(float)
        if msg.encoding=='16UC1':depth*=.001
        thermal_frame=str(self.get_parameter('thermal_frame').value)
        try:
            tf=self.buffer.lookup_transform(thermal_frame,msg.header.frame_id,
                rclpy.time.Time.from_msg(msg.header.stamp),timeout=rclpy.duration.Duration(seconds=.02))
        except TransformException:return
        q=tf.transform.rotation;t=tf.transform.translation
        transform=np.eye(4)
        transform[:3,:3]=[[1-2*(q.y*q.y+q.z*q.z),2*(q.x*q.y-q.z*q.w),2*(q.x*q.z+q.y*q.w)],
            [2*(q.x*q.y+q.z*q.w),1-2*(q.x*q.x+q.z*q.z),2*(q.y*q.z-q.x*q.w)],
            [2*(q.x*q.z-q.y*q.w),2*(q.y*q.z+q.x*q.w),1-2*(q.x*q.x+q.y*q.y)]]
        transform[:3,3]=[t.x,t.y,t.z]
        registered=register_depth(depth,self.dk,self.tk,transform)
        out=Image();out.header=msg.header;out.header.frame_id=thermal_frame
        out.height,out.width=registered.shape;out.encoding='32FC1';out.step=out.width*4;out.data=registered.tobytes()
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args);node=DepthRegistrationNode()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
