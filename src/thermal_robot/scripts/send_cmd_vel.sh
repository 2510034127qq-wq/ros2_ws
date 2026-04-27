#!/bin/bash
# send_cmd_vel.sh
# ===============
# Test /cmd_vel → robot movement.
# NOTE: For a position-controlled robot, /cmd_vel moves the robot via
#       the planar move plugin in Gazebo. The robot's pose is updated in
#       /odom and /tf. The leg joints remain held at standing posture.
#
# Usage:
#   bash scripts/send_cmd_vel.sh            # forward 0.1 m/s for 3 s
#   bash scripts/send_cmd_vel.sh 0.2 0.3    # forward 0.2, rotate 0.3

LIN=${1:-0.1}
ANG=${2:-0.0}

echo "=== Sending /cmd_vel: linear.x=$LIN angular.z=$ANG ==="
ros2 topic pub --times 30 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: ${LIN}, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: ${ANG}}}"

echo ""
echo "=== Checking /odom ==="
ros2 topic echo /odom --once

echo ""
echo "=== Done. Check Gazebo for robot displacement. ==="
