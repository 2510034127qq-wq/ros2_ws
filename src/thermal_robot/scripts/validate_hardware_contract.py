#!/usr/bin/env python3
"""Synthetic ROS input check for the real UGV overlay (no hardware motion).

Publishes padded big-endian TLinear and millimetre depth with known optical TF.
Checks the actual radiometry -> registration -> filtering -> mapper -> tracker
chain. This is software evidence, not camera or robot calibration evidence.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, Twist
from thermal_interfaces.msg import ThermalMap, SourceEstimateArray, BeliefState
from tf2_ros import StaticTransformBroadcaster


class Fixture(Node):
    def __init__(self):
        super().__init__('hardware_contract_fixture')
        self.raw=None;self.depth=None;self.map=None;self.sources=[];self.health=[]
        self.motion_messages=0
        self.pubs={topic:self.create_publisher(cls,topic,qos_profile_sensor_data)
            for topic,cls in [('/thermal/radiometric_input',Image),('/depth/image',Image),
                              ('/thermal/camera_info',CameraInfo),('/depth/camera_info',CameraInfo)]}
        for topic,cls,attr in [('/thermal/raw',Image,'raw'),('/thermal/depth',Image,'depth'),
                                ('/thermal/map',ThermalMap,'map')]:
            self.create_subscription(cls,topic,lambda msg,a=attr:setattr(self,a,msg),qos_profile_sensor_data)
        self.create_subscription(SourceEstimateArray,'/thermal/sources',
            lambda msg:setattr(self,'sources',msg.sources),qos_profile_sensor_data)
        self.create_subscription(BeliefState,'/thermal/belief',
            lambda msg:self.health.append(msg.health),qos_profile_sensor_data)
        self.create_subscription(Twist,'/cmd_vel',self._motion,qos_profile_sensor_data)
        self.broadcaster=StaticTransformBroadcaster(self)
        transforms=[]
        for child in ('thermal_optical_frame','depth_optical_frame','base_link'):
            tf=TransformStamped();tf.header.frame_id='map';tf.child_frame_id=child
            tf.header.stamp=self.get_clock().now().to_msg()
            tf.transform.translation.x=1.;tf.transform.translation.y=2.;tf.transform.translation.z=.6
            q=tf.transform.rotation
            if child=='base_link':q.w=1.
            else:q.x=-.5;q.y=.5;q.z=-.5;q.w=.5
            transforms.append(tf)
        self.broadcaster.sendTransform(transforms)
        self.create_timer(.1,self.publish)

    def _motion(self,msg):
        self.motion_messages+=1

    def publish(self):
        stamp=self.get_clock().now().to_msg()
        for prefix,frame in [('thermal','thermal_optical_frame'),('depth','depth_optical_frame')]:
            info=CameraInfo();info.header.stamp=stamp;info.header.frame_id=frame
            info.width=16;info.height=12
            info.k=[20.,0.,7.5,0.,20.,5.5,0.,0.,1.]
            info.p=[20.,0.,7.5,0.,0.,20.,5.5,0.,0.,0.,1.,0.]
            info.distortion_model='plumb_bob';info.d=[0.]*5
            self.pubs['/'+prefix+'/camera_info'].publish(info)
        thermal=np.full((12,17),29515,dtype='>u2')
        thermal[3:9,5:11]=33300;thermal[0,0]=0
        depth=np.full((12,17),3000,dtype='>u2');depth[0,0]=0
        for topic,frame,arr in [('/thermal/radiometric_input','thermal_optical_frame',thermal),
                                ('/depth/image','depth_optical_frame',depth)]:
            msg=Image();msg.header.stamp=stamp;msg.header.frame_id=frame
            msg.width=16;msg.height=12;msg.step=34;msg.encoding='16UC1';msg.is_bigendian=True
            msg.data=arr.tobytes();self.pubs[topic].publish(msg)

    def report(self):
        failures=[]
        def check(ok,name):
            if not ok:failures.append(name)
        check(self.raw is not None,'raw_missing')
        check(self.depth is not None,'registered_depth_missing')
        check(self.map is not None,'map_missing')
        if self.raw is not None:
            raw=np.frombuffer(bytes(self.raw.data),dtype='<f4').reshape(12,16)
            check(self.raw.header.frame_id=='thermal_optical_frame','optical_frame_lost')
            check(abs(float(raw[5,7])-59.85)<.01,'temperature_conversion')
            check(np.isnan(raw[0,0]),'invalid_temperature_not_masked')
        if self.depth is not None:
            depth=np.frombuffer(bytes(self.depth.data),dtype='<f4').reshape(12,16)
            check(abs(float(depth[5,7])-3.)<1e-5,'depth_units_or_registration')
            check(np.isnan(depth[0,0]),'invalid_depth_not_masked')
        hot_xy=[]
        if self.map is not None:
            m=self.map;t=np.asarray(m.temperature_mean).reshape(m.height,m.width)
            # Ground-projected cells pool the vertical face: six hot rows at
            # 59.85 C and six ambient rows at 22 C yield 40.925 C, not 59.85 C.
            check(abs(float(t.max())-40.925)<.03,'surface_column_fusion')
            rows,cols=np.where(t>30)
            hot_xy=np.column_stack((m.origin_x+(cols+.5)*m.resolution,
                                    m.origin_y+(rows+.5)*m.resolution)).tolist()
            check(m.header.frame_id=='map' and m.measurement_type=='surface_radiance','map_contract')
            check(len(hot_xy)>0,'hot_surface_missing')
            if hot_xy:
                check(all(abs(x-4.)<=m.resolution and abs(y-2.)<.6 for x,y in hot_xy),
                      'optical_to_world_projection')
        confirmed=[s for s in self.sources if s.status=='confirmed']
        check(len(confirmed)==1,'source_confirmation_or_duplicates')
        check('ready' in self.health,'slow_worker_not_ready')
        check(self.motion_messages==0,'motion_disabled_contract')
        return dict(passed=not failures,failures=failures,hot_cells_xy=hot_xy,
            confirmed_sources=[dict(id=s.id,x=s.position.x,y=s.position.y) for s in confirmed],
            belief_ready_messages=self.health.count('ready'),motion_messages=self.motion_messages,
            input='synthetic padded big-endian TLinear and mm depth; static optical TF',
            limitation='No physical camera, calibration, UVC or UGV motion verified.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--domain',type=int,default=159)
    parser.add_argument('--duration',type=float,default=18.)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ['ROS_DOMAIN_ID']=str(args.domain);os.environ['ROS_LOG_DIR']=str(out/'roslog')
    rclpy.init();node=Fixture();child=None
    try:
        with (out/'launch.log').open('w') as log:
            child=subprocess.Popen(['ros2','launch','thermal_bringup','ugv_thermal_launch.py',
                'enable_motion:=false'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+args.duration
            while time.monotonic()<deadline:
                if child.poll() is not None:raise RuntimeError('hardware overlay stopped')
                rclpy.spin_once(node,timeout_sec=.1)
            report=node.report();(out/'result.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report,indent=2))
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        if child is not None:
            try:child.send_signal(signal.SIGINT)
            except ProcessLookupError:pass
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    raise SystemExit(0 if report['passed'] else 2)


if __name__=='__main__':main()
