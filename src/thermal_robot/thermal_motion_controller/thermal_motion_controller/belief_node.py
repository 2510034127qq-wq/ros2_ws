#!/usr/bin/env python3
"""Independent slow belief worker; failure cannot block the controller process."""
import time
from dataclasses import fields
import numpy as np
import rclpy
from rclpy.node import Node
from thermal_interfaces.msg import ThermalMap, BeliefState, SourceEstimate
from .belief import SourceBelief, BeliefParams
from .source_tracking import SourceTrackerCore
from thermal_field_reconstructor.residual import predict_field


class BeliefNode(Node):
    def __init__(self):
        super().__init__('belief_node')
        defaults=BeliefParams()
        for f in fields(defaults): self.declare_parameter(f.name,getattr(defaults,f.name))
        self.declare_parameter('mode','online')
        self.declare_parameter('update_rate',1.)
        self.declare_parameter('ambient_temp',22.)
        self.declare_parameter('freshness_s',1.5)
        self.declare_parameter('residual_birth_threshold',3.)
        self.declare_parameter('detection_merge_m',.5)
        self.params=BeliefParams(**{f.name:self.get_parameter(f.name).value for f in fields(defaults)})
        self.belief=SourceBelief(self.params)
        self.mode=str(self.get_parameter('mode').value)
        if self.mode not in ('off','shadow','online'): raise ValueError('invalid belief mode')
        self.ambient=float(self.get_parameter('ambient_temp').value)
        self.freshness=float(self.get_parameter('freshness_s').value)
        self.extractor=SourceTrackerCore(ambient_temp=self.ambient,
            min_temp_rise=float(self.get_parameter('residual_birth_threshold').value),
            merge_radius_m=float(self.get_parameter('detection_merge_m').value),
            max_detection_age_s=self.freshness,max_detections=self.params.max_sources)
        self.extractor.estimator_model="kalman"
        self.latest=None; self.last_stamp=None; self.revision=0
        self.create_subscription(ThermalMap,'/thermal/map',self._map,3)
        self.pub=self.create_publisher(BeliefState,'/thermal/belief',3)
        self.create_timer(1/float(self.get_parameter('update_rate').value),self._tick)
        self.get_logger().info(f'[BELIEF] mode={self.mode} independent worker')

    def _map(self,msg):
        self.latest=msg

    def _tick(self):
        msg=self.latest
        if self.mode=='off' or msg is None: return
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        if stamp==self.last_stamp: return
        self.last_stamp=stamp
        start=time.perf_counter()
        out=BeliefState();out.header=msg.header;out.mode=self.mode
        try:
            shape=(msg.height,msg.width)
            temp=np.asarray(msg.temperature_mean).reshape(shape)
            conf=np.asarray(msg.confidence).reshape(shape)
            age=np.asarray(msg.last_seen_age_s).reshape(shape)
            kwargs=(msg.resolution,msg.origin_x,msg.origin_y,age)
            self.extractor.measurement_type=msg.measurement_type or "field_direct"
            detections=self.extractor.extract_detections(temp,conf,*kwargs)
            births=detections
            if msg.measurement_type!="surface_radiance":
                # Gaussian residual births apply only to direct field observations.
                sources=[(*c.position.state[:2],c.amplitude,c.sigma)
                         for c in self.belief.clusters if c.confirmed]
                predicted=predict_field(msg.width,msg.height,msg.resolution,msg.origin_x,
                                        msg.origin_y,self.ambient,sources)
                births=self.extractor.extract_detections(temp-predicted+self.ambient,conf,*kwargs)
            def visibility(x,y):
                ix=int(np.floor((x-msg.origin_x)/msg.resolution))
                iy=int(np.floor((y-msg.origin_y)/msg.resolution))
                if not (0<=ix<msg.width and 0<=iy<msg.height): return 0.
                return float(0<=age[iy,ix]<=self.freshness and conf[iy,ix]>.1)
            snapshot=dict(temperature_mean=temp,confidence=conf,last_seen_age_s=age,
                resolution=msg.resolution,origin_x=msg.origin_x,origin_y=msg.origin_y,
                ambient=self.ambient,measurement_type=msg.measurement_type)
            good=self.belief.update(detections,stamp,visibility,births,snapshot)
            out.compute_ms=float((time.perf_counter()-start)*1000)
            out.health=self.belief.health if out.compute_ms<=self.params.budget_ms else 'over_budget'
            if good and out.health=='ready':
                self.revision+=1
                out.cardinality_pmf=self.belief.cardinality().tolist()
                for c in self.belief.clusters:
                    s=SourceEstimate();s.header=msg.header;s.id=c.label
                    s.status='confirmed' if c.confirmed else 'candidate'
                    if stamp-c.last_seen_s>self.freshness: s.status='stale' if c.confirmed else 'candidate'
                    s.position.x,s.position.y=map(float,c.position.state[:2])
                    s.covariance_xx=float(c.position.covariance[0,0]);s.covariance_xy=float(c.position.covariance[0,1])
                    s.covariance_yy=float(c.position.covariance[1,1])
                    s.strength=float(c.amplitude);s.sigma=float(c.sigma)
                    s.existence_probability=float(c.probability);s.confidence=float(c.probability)
                    s.observations=c.hits;s.age_s=float(stamp-c.last_seen_s)
                    out.sources.append(s)
            out.revision=self.revision
        except Exception as exc:
            out.health='error:'+type(exc).__name__
            self.get_logger().error(f'[BELIEF_ERROR] {exc}')
        self.pub.publish(out)
        self.get_logger().info(f'[BELIEF] {out.health} n={len(out.sources)} ms={out.compute_ms:.1f} '
                               f'count={list(out.cardinality_pmf)}')


def main(args=None):
    rclpy.init(args=args);node=BeliefNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
