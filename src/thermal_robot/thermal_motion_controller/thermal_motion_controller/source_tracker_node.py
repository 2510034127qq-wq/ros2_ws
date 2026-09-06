#!/usr/bin/env python3
"""source_tracker_node.py — source candidates from the world thermal map."""

import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from thermal_interfaces.msg import SourceEstimate, SourceEstimateArray, ThermalMap, BeliefState

from thermal_motion_controller.source_tracking import SourceTrackerCore, TrackedSource
from thermal_motion_controller.runtime_policy import slow_output_usable


class SourceTrackerNode(Node):
    def __init__(self):
        super().__init__('source_tracker_node')
        self.declare_parameter('ambient_temp', 22.0)
        self.declare_parameter('min_temp_rise', 4.0)
        self.declare_parameter('min_confidence', 0.12)
        self.declare_parameter('max_detections', 12)
        self.declare_parameter('gate_m', 1.25)
        self.declare_parameter('merge_radius_m', 1.0)
        self.declare_parameter('duplicate_radius_m', 3.5)
        self.declare_parameter('confirm_probability', 0.75)
        self.declare_parameter('confirm_observations', 5)
        self.declare_parameter('confirm_covariance_max', 0.9)
        self.declare_parameter('stale_after_s', 12.0)
        self.declare_parameter('stale_decay_s', 20.0)
        self.declare_parameter('duplicate_memory_s', 60.0)
        self.declare_parameter('update_alpha_min', 0.08)
        self.declare_parameter('max_detection_age_s', 8.0)

        g = self.get_parameter
        self._tracker = SourceTrackerCore(
            ambient_temp=float(g('ambient_temp').value),
            min_temp_rise=float(g('min_temp_rise').value),
            min_confidence=float(g('min_confidence').value),
            max_detections=int(g('max_detections').value),
            gate_m=float(g('gate_m').value),
            merge_radius_m=float(g('merge_radius_m').value),
            duplicate_radius_m=float(g('duplicate_radius_m').value),
            confirm_probability=float(g('confirm_probability').value),
            confirm_observations=int(g('confirm_observations').value),
            confirm_covariance_max=float(g('confirm_covariance_max').value),
            stale_after_s=float(g('stale_after_s').value),
            stale_decay_s=float(g('stale_decay_s').value),
            duplicate_memory_s=float(g('duplicate_memory_s').value),
            update_alpha_min=float(g('update_alpha_min').value),
            max_detection_age_s=float(g('max_detection_age_s').value),
        )
        for name, default in [('estimator_model', 'legacy'), ('update_rate', 2.0),
                              ('position_noise_std', .05), ('measurement_variance', .15),
                              ('association_gate_chi2', 9.21), ('max_tracks', 32),
                              ('cold_evidence_decay_s', 2.),
                              ('slow_prior_enabled', True),('strategy','fast'),('slow_timeout_s',3.)]:
            self.declare_parameter(name, default)
        self._tracker.estimator_model = str(g('estimator_model').value)
        self._tracker.position_noise_std = float(g('position_noise_std').value)
        self._tracker.measurement_variance = float(g('measurement_variance').value)
        self._tracker.association_gate_chi2 = float(g('association_gate_chi2').value)
        self._tracker.max_tracks = int(g('max_tracks').value)
        self._tracker.cold_evidence_decay_s = float(g('cold_evidence_decay_s').value)
        self._update_period = 1.0 / float(g('update_rate').value)
        self._last_update = -float('inf')
        self._slow_prior_enabled = bool(g('slow_prior_enabled').value)
        self._strategy=str(g('strategy').value)
        self._slow_timeout=float(g('slow_timeout_s').value)
        self._last_prior_revision = -1
        self._last_log_s = 0.0
        self._last_confirmed = set()

        rel = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=3,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(ThermalMap, '/thermal/map', self._map_cb, rel)
        self.create_subscription(BeliefState, '/thermal/belief', self._prior_cb, rel)
        self._pub = self.create_publisher(SourceEstimateArray, '/thermal/sources', rel)
        self.get_logger().info('source_tracker_node | /thermal/map -> /thermal/sources')

    def _map_cb(self, msg: ThermalMap):
        if msg.width == 0 or msg.height == 0:
            return
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        if not hasattr(self,'_stamp_origin'): self._stamp_origin=stamp
        now_s = stamp-self._stamp_origin
        if self._tracker.estimator_model == 'kalman' and now_s-self._last_update < self._update_period:
            return
        self._last_update = now_s
        started=time.perf_counter()
        temp = np.asarray(msg.temperature_mean, dtype=np.float32).reshape((msg.height, msg.width))
        conf = np.asarray(msg.confidence, dtype=np.float32).reshape((msg.height, msg.width))
        age = np.asarray(msg.last_seen_age_s, dtype=np.float32).reshape((msg.height, msg.width))
        self._tracker.measurement_type=msg.measurement_type or "field_direct"
        tracks = self._tracker.update_from_map(
            temp, conf, msg.resolution, msg.origin_x, msg.origin_y, now_s, age)
        out = SourceEstimateArray()
        out.header = msg.header
        out.header.frame_id = msg.header.frame_id
        for track in sorted(tracks, key=lambda t: (t.status != 'confirmed', -t.existence_probability)):
            out.sources.append(self._to_msg(msg.header, track))
        self._pub.publish(out)
        self.get_logger().info(f'[FAST_TIMING] update_ms={(time.perf_counter()-started)*1000:.3f}')
        self._log_status(tracks, now_s)

    def _to_msg(self, header, track: TrackedSource) -> SourceEstimate:
        msg = SourceEstimate()
        msg.header = header
        msg.header.frame_id = header.frame_id
        msg.id = track.track_id
        msg.status = track.status
        msg.position.x = float(track.x)
        msg.position.y = float(track.y)
        msg.position.z = 0.0
        msg.covariance_xx = float(track.covariance_xx)
        msg.covariance_xy = float(track.covariance_xy)
        msg.covariance_yy = float(track.covariance_yy)
        msg.strength = float(track.strength)
        msg.sigma = float(track.sigma)
        msg.existence_probability = float(track.existence_probability)
        msg.confidence = float(track.confidence)
        msg.observations = int(track.observations)
        msg.age_s = float(max(0., self._last_update-track.last_seen_s))
        return msg

    def _prior_cb(self, msg):
        if (not self._slow_prior_enabled or self._tracker.estimator_model != 'kalman'
                or msg.mode != 'online' or msg.health != 'ready'
                or msg.revision == self._last_prior_revision):
            return
        age = (self.get_clock().now().nanoseconds -
               (msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec))/1e9
        if not slow_output_usable(self._strategy,msg.mode,msg.health,0.,age,
                                  timeout_s=self._slow_timeout):
            return
        self._last_prior_revision = msg.revision
        # Conservative covariance intersection: slow evidence shares observations
        # with fast tracks and must not be counted as independent measurements.
        used = set()
        for src in msg.sources:
            if src.existence_probability < .9:
                continue
            choices = [(np.hypot(t.x-src.position.x,t.y-src.position.y),t)
                       for t in self._tracker.tracks if t.track_id not in used
                       and t.status == 'confirmed'
                       and t.existence_probability >= self._tracker.confirm_probability
                       and self._last_update-t.last_seen_s <= self._tracker.max_detection_age_s]
            if not choices:
                break
            distance, track = min(choices,key=lambda pair:pair[0])
            if distance > self._tracker.gate_m:
                continue
            f = self._tracker._filters.get(track.track_id)
            if f is None:
                continue
            slow_cov = np.diag([max(src.covariance_xx,.1),max(src.covariance_yy,.1)])
            ia,ib=np.linalg.inv(f.covariance),np.linalg.inv(slow_cov)
            cov=np.linalg.inv(.9*ia+.1*ib)
            prior=np.array([src.position.x,src.position.y])
            f.state=cov@(.9*ia@f.state+.1*ib@prior)
            f.covariance=cov
            used.add(track.track_id)

    def _log_status(self, tracks, now_s: float):
        confirmed = {t.track_id for t in tracks if t.status == 'confirmed'}
        new_confirmed = confirmed - self._last_confirmed
        for tid in sorted(new_confirmed):
            tr = next(t for t in tracks if t.track_id == tid)
            self.get_logger().info(
                f'[SOURCE_CONFIRMED] {tr.track_id} pos=({tr.x:.2f},{tr.y:.2f}) '
                f'p={tr.existence_probability:.2f} obs={tr.observations}')
        self._last_confirmed = confirmed
        if now_s - self._last_log_s < 5.0:
            return
        self._last_log_s = now_s
        visible = [t for t in tracks if t.status in ('candidate', 'confirmed', 'stale')]
        summary = ', '.join(
            f'{t.track_id}:{t.status[0]} p={t.existence_probability:.2f} '
            f'@({t.x:.1f},{t.y:.1f})'
            for t in visible[:6]
        ) or 'none'
        self.get_logger().info(f'[SOURCE_TRACKS] {summary}')


def main(args=None):
    rclpy.init(args=args)
    node = SourceTrackerNode()
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
