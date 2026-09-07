#!/usr/bin/env python3
"""Independent slow belief worker; failure cannot block the controller process."""
import time
from dataclasses import fields
import numpy as np
import rclpy
from rclpy.node import Node
from thermal_interfaces.msg import ThermalMap, BeliefState, SourceEstimate, SourceEstimateArray
from .belief import SourceBelief, BeliefParams
from .source_tracking import TrackedSource


class BeliefNode(Node):
    def __init__(self):
        super().__init__('belief_node')
        defaults=BeliefParams()
        for f in fields(defaults): self.declare_parameter(f.name,getattr(defaults,f.name))
        self.declare_parameter('mode','online')
        self.declare_parameter('update_rate',1.)
        self.declare_parameter('ambient_temp',22.)
        self.declare_parameter('freshness_s',1.5)
        self.params=BeliefParams(**{f.name:self.get_parameter(f.name).value for f in fields(defaults)})
        self.belief=SourceBelief(self.params)
        self.mode=str(self.get_parameter('mode').value)
        if self.mode not in ('off','shadow','online'): raise ValueError('invalid belief mode')
        self.ambient=float(self.get_parameter('ambient_temp').value)
        self.freshness=float(self.get_parameter('freshness_s').value)
        self.latest=None; self.latest_map=None; self.last_stamp=None; self.revision=0
        self.create_subscription(ThermalMap,'/thermal/map',self._map,3)
        self.create_subscription(SourceEstimateArray,'/thermal/sources',self._sources,3)
        self.pub=self.create_publisher(BeliefState,'/thermal/belief',3)
        self.create_timer(1/float(self.get_parameter('update_rate').value),self._tick)
        self.get_logger().info(f'[BELIEF] mode={self.mode} independent worker')

    def _map(self,msg):
        self.latest_map=msg

    def _sources(self,msg):
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
            registry=[TrackedSource(s.id,s.position.x,s.position.y,s.strength,s.sigma,
                s.existence_probability,s.confidence,s.observations,stamp-s.age_s,
                s.covariance_xx,s.covariance_xy,s.covariance_yy,s.status) for s in msg.sources]
            snapshot=None
            m=self.latest_map
            if m is not None:
                map_stamp=m.header.stamp.sec+m.header.stamp.nanosec/1e9
                if m.header.frame_id==msg.header.frame_id and abs(stamp-map_stamp)<=self.freshness:
                    shape=(m.height,m.width)
                    snapshot=dict(temperature_mean=np.asarray(m.temperature_mean).reshape(shape),
                        confidence=np.asarray(m.confidence).reshape(shape),
                        last_seen_age_s=np.asarray(m.last_seen_age_s).reshape(shape),
                        resolution=m.resolution,origin_x=m.origin_x,origin_y=m.origin_y,
                        ambient=self.ambient,measurement_type=m.measurement_type)
            good=self.belief.update(registry,stamp,snapshot)
            out.compute_ms=float((time.perf_counter()-start)*1000)
            out.health=self.belief.health if out.compute_ms<=self.params.budget_ms else 'over_budget'
            if good and out.health=='ready':
                self.revision+=1
                out.cardinality_pmf=self.belief.cardinality().tolist()
                for c in self.belief.clusters:
                    s=SourceEstimate();s.header=msg.header;s.id=c.label
                    s.status='confirmed' if c.confirmed else 'candidate'
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
