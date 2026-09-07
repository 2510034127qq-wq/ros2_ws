#!/usr/bin/env python3
"""Finite ROS software-contract probe. No truth-dependent control or recall gate."""
import argparse
import collections
import json
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image,LaserScan
from nav_msgs.msg import Odometry,Path as NavPath
from geometry_msgs.msg import Twist
from thermal_interfaces.msg import ThermalMap,SourceEstimateArray,BeliefState,GradientArray,ThermalField
from gazebo_msgs.msg import ModelStates


class Probe(Node):
    def __init__(self,out):
        super().__init__('software_contract_probe')
        self.out=out;self.counts=collections.Counter();self.health=collections.Counter()
        self.sources=set();self.belief_sources=set();self.cpu=[];self.rows=[];self.positions=[]
        self.max_hot=None;self.max_observed_cells=0;self.nonzero_cmd=0;self.valid_depth=0
        self.snapshots={};self.source_rows=[];self.truth=[];self.truth_rows=[];self.physical_truth=[]
        self.fresh_location_errors=[];self.inactive_confirmations=0
        self.truth_errors=[];self.last_received={}
        self.commands=[];self.yaws=[]
        self.body_errors=[];self.body_failures=[]
        self.create_subscription(ModelStates,'/thermal_scene/model_states',
                                 self.check_body,qos_profile_sensor_data)
        subscriptions={'raw':(Image,'/sim/thermal_raw'),'filtered':(Image,'/thermal/filtered'),
            'depth':(Image,'/thermal/depth'),'field':(ThermalField,'/thermal/field'),
            'gradient':(GradientArray,'/thermal/gradient'),'map':(ThermalMap,'/thermal/map'),
            'sources':(SourceEstimateArray,'/thermal/sources'),'belief':(BeliefState,'/thermal/belief'),
            'odom':(Odometry,'/odom'),'scan':(LaserScan,'/scan'),'cmd_vel':(Twist,'/cmd_vel'),
            'plan':(NavPath,'/plan'),
            'truth':(SourceEstimateArray,'/sim/thermal_sources_truth')}
        for name,(cls,topic) in subscriptions.items():
            self.create_subscription(cls,topic,lambda msg,name=name:self.receive(name,msg),qos_profile_sensor_data)

    def check_body(self,msg):
        # Static bodies can be checked together against Gazebo's physical state;
        # no asynchronous per-body service request needs to be held outstanding.
        poses=dict(zip(msg.name,msg.pose))
        for target in self.physical_truth:
            name='thermal_body_'+target.id
            if name not in poses:
                self.body_failures.append(name)
                continue
            position=poses[name].position
            self.body_errors.append(float(np.hypot(
                position.x-target.position.x,position.y-target.position.y)))

    def receive(self,name,msg):
        self.counts[name]+=1
        self.last_received[name]=time.monotonic()
        if name in ('raw','filtered','depth'):
            arr=np.frombuffer(bytes(msg.data),dtype=np.float32).reshape(msg.height,msg.width)
            self.snapshots[name]=arr.copy()
            if name=='depth':self.valid_depth=max(self.valid_depth,int(np.isfinite(arr).sum()))
            if name=='raw' and np.isfinite(arr).any():
                self.max_hot=max(self.max_hot or -1e9,float(np.nanmax(arr)))
        elif name=='map':
            temp=np.asarray(msg.temperature_mean).reshape(msg.height,msg.width)
            self.snapshots['map']=temp
            self.snapshots['map_geometry']=np.array([msg.origin_x,msg.origin_y,msg.resolution])
            self.max_observed_cells=max(self.max_observed_cells,int(np.count_nonzero(msg.visit_count)))
        elif name=='belief':
            self.health[msg.health]+=1;self.cpu.append(float(msg.compute_ms))
            self.belief_sources.update(s.id for s in msg.sources)
            self.rows.append(dict(t=time.monotonic(),health=msg.health,mode=msg.mode,
                                 cpu_ms=msg.compute_ms,cardinality=list(msg.cardinality_pmf)))
        elif name=='sources':
            self.sources.update(s.id for s in msg.sources if s.status=='confirmed')
            for s in msg.sources:self.source_rows.append(dict(t=time.monotonic(),id=s.id,status=s.status,
                x=s.position.x,y=s.position.y,p=s.existence_probability,
                age=s.age_s))
            for s in msg.sources:
                if s.status=='confirmed' and self.truth:
                    self.truth_errors.append(min(np.hypot(s.position.x-t.position.x,
                        s.position.y-t.position.y) for t in self.truth))
                if s.status=='confirmed' and self.physical_truth:
                    nearest=min(self.physical_truth,key=lambda t:np.hypot(s.position.x-t.position.x,
                                                                                  s.position.y-t.position.y))
                    self.inactive_confirmations+=int(nearest.status!='truth_active')
                    if s.age_s<=1.5:
                        self.fresh_location_errors.append(float(np.hypot(s.position.x-nearest.position.x,
                                                                         s.position.y-nearest.position.y)))
        elif name=='truth':
            self.physical_truth=msg.sources
            self.truth=[s for s in msg.sources if s.status=='truth_active']
            self.truth_rows.append(dict(t=time.monotonic(),sources=[dict(id=s.id,
                x=s.position.x,y=s.position.y,status=s.status) for s in msg.sources]))
        elif name=='odom':
            self.positions.append((msg.pose.pose.position.x,msg.pose.pose.position.y))
            q=msg.pose.pose.orientation
            self.yaws.append(float(np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))))
        elif name=='cmd_vel':
            self.nonzero_cmd+=int(abs(msg.linear.x)+abs(msg.angular.z)>1e-4)
            self.commands.append((time.monotonic(),msg.linear.x,msg.angular.z))

    def save(self,args):
        required=['raw','filtered','field','gradient','map','sources','odom','scan','cmd_vel']
        if args.sensor_model=='b':required+=['depth']
        if args.belief_mode!='off':required+=['belief']
        failures=[f'missing:{name}' for name in required if self.counts[name]==0]
        if self.max_observed_cells==0:failures.append('no_map_observations')
        if args.sensor_model=='b' and self.valid_depth==0:failures.append('no_valid_depth')
        if args.sensor_model=='b':
            if not self.body_errors:failures.append('no_physical_surface_geometry')
            if self.body_failures or (self.body_errors and max(self.body_errors)>.25):
                failures.append('physical_surface_geometry_out_of_sync')
        if args.belief_mode!='off' and self.health[args.expected_health]==0:
            failures.append('belief_never_'+args.expected_health)
        for name in required:
            timeout_s=5.
            if name in self.last_received and time.monotonic()-self.last_received[name]>timeout_s:
                failures.append('stopped:'+name)
        if self.nonzero_cmd==0:failures.append('no_motion_command')
        distance=float(np.linalg.norm(np.diff(np.asarray(self.positions),axis=0),axis=1).sum()) if len(self.positions)>1 else 0.
        if distance<args.min_path_m:failures.append('insufficient_actual_motion')
        report=dict(passed=not failures,failures=failures,counts=dict(self.counts),belief_health=dict(self.health),
                    max_observed_cells=self.max_observed_cells,valid_depth_pixels=self.valid_depth,
                    confirmed_ids=sorted(self.sources),belief_ids=sorted(self.belief_sources),
                    belief_compute_ms_p99=float(np.percentile(self.cpu,99)) if self.cpu else None,
                    path_length_m=distance,nonzero_commands=self.nonzero_cmd,max_temperature_c=self.max_hot,
                    sensor_model=args.sensor_model,strategy=args.strategy,belief_mode=args.belief_mode,
                    expected_health=args.expected_health,
                    yaw_range_rad=float(np.ptp(np.unwrap(self.yaws))) if self.yaws else 0.,
                    physical_geometry_samples=len(self.body_errors),
                    physical_geometry_error_max_m=max(self.body_errors) if self.body_errors else None,
                    confirmed_nearest_truth_error_p95_m=float(np.percentile(self.truth_errors,95)) if self.truth_errors else None,
                    confirmed_nearest_truth_error_max_m=float(max(self.truth_errors)) if self.truth_errors else None,
                    fresh_nearest_physical_source_error_p95_m=float(np.percentile(self.fresh_location_errors,95)) if self.fresh_location_errors else None,
                    confirmed_inactive_nearest_frames=self.inactive_confirmations,
                    note='Nearest-source distances are diagnostics, not matched precision or recall; use detection_metrics.json for physical-source matching.')
        (self.out/'probe.json').write_text(json.dumps(report,indent=2))
        (self.out/'belief.json').write_text(json.dumps(self.rows,indent=2))
        (self.out/'sources.json').write_text(json.dumps(self.source_rows,indent=2))
        (self.out/'truth.json').write_text(json.dumps(self.truth_rows,indent=2))
        (self.out/'commands.json').write_text(json.dumps(self.commands))
        np.savez_compressed(self.out/'snapshots.npz',**self.snapshots)
        print(json.dumps(report,indent=2),flush=True)
        return report['passed']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--duration',type=float,default=60.)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--sensor-model',choices=['a','b'],default='a')
    parser.add_argument('--strategy',choices=['full','frontier','levy','residual','fast','dual','gp_ucb'],default='dual')
    parser.add_argument('--belief-mode',choices=['online','shadow','off'],default='online')
    parser.add_argument('--expected-health',default='ready')
    parser.add_argument('--min-path-m',type=float,default=.5)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    rclpy.init();node=Probe(args.out)
    try:
        deadline=time.monotonic()+args.duration
        while rclpy.ok() and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.1)
        success=node.save(args)
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    raise SystemExit(0 if success else 2)

if __name__=='__main__':main()
