#!/usr/bin/env python3
"""Post-launch Nav2 health check for the thermal robot stack."""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict

import rclpy
from geometry_msgs.msg import Twist
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformListener


NAV2_LIFECYCLE_NODES = (
    "/planner_server",
    "/controller_server",
    "/bt_navigator",
)


class Nav2HealthCheck(Node):
    def __init__(self):
        super().__init__("thermal_nav2_health_check")
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.plan_count = 0
        self.cmd_vel_count = 0
        self.create_subscription(Path, "/plan", self._plan_cb, qos)
        self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_cb, qos)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _plan_cb(self, _msg: Path) -> None:
        self.plan_count += 1

    def _cmd_vel_cb(self, _msg: Twist) -> None:
        self.cmd_vel_count += 1

    def lifecycle_state(self, node_name: str, timeout_s: float) -> Dict:
        client = self.create_client(GetState, f"{node_name}/get_state")
        if not client.wait_for_service(timeout_sec=timeout_s):
            return {"ok": False, "state": "service_unavailable"}
        future = client.call_async(GetState.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        response = future.result()
        if response is None:
            return {"ok": False, "state": "no_response"}
        state = response.current_state
        return {
            "ok": state.id == State.PRIMARY_STATE_ACTIVE,
            "state": state.label,
            "id": int(state.id),
        }

    def tf_available(self) -> bool:
        for target, source in (("map", "base_link"), ("odom", "base_link")):
            try:
                self.tf_buffer.lookup_transform(
                    target,
                    source,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.05),
                )
                return True
            except (LookupException, ConnectivityException, ExtrapolationException):
                continue
        return False


def wait_for_counts(node: Nav2HealthCheck, timeout_s: float) -> Dict[str, bool]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.plan_count > 0 and node.cmd_vel_count > 0 and node.tf_available():
            break
    return {
        "plan": node.plan_count > 0,
        "cmd_vel": node.cmd_vel_count > 0,
        "tf": node.tf_available(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=20.0, help="total topic/TF wait time")
    parser.add_argument("--service-timeout", type=float, default=3.0, help="per lifecycle service wait time")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON only")
    args = parser.parse_args()

    rclpy.init()
    node = Nav2HealthCheck()
    try:
        lifecycle = {
            name: node.lifecycle_state(name, args.service_timeout)
            for name in NAV2_LIFECYCLE_NODES
        }
        topics = wait_for_counts(node, args.timeout)
        result = {
            "lifecycle": lifecycle,
            "topics": {
                **topics,
                "plan_count": node.plan_count,
                "cmd_vel_count": node.cmd_vel_count,
            },
        }
        ok = all(item["ok"] for item in lifecycle.values()) and all(topics.values())
        result["ok"] = ok
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print("[nav2_health] lifecycle:")
            for name, item in lifecycle.items():
                status = "OK" if item["ok"] else "FAIL"
                print(f"  {status:4s} {name}: {item['state']}")
            print("[nav2_health] topics/tf:")
            print(f"  {'OK' if topics['plan'] else 'FAIL':4s} /plan count={node.plan_count}")
            print(f"  {'OK' if topics['cmd_vel'] else 'FAIL':4s} /cmd_vel count={node.cmd_vel_count}")
            print(f"  {'OK' if topics['tf'] else 'FAIL':4s} TF map/odom -> base_link")
        return 0 if ok else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
