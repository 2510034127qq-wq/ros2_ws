#!/bin/bash
# Run this BEFORE ros2 launch to clear any leftover Gazebo processes.
# Usage: bash kill_gz.sh
echo "Killing leftover Gazebo processes..."
pkill -9 -f gzserver 2>/dev/null && echo "  killed gzserver" || echo "  no gzserver found"
pkill -9 -f gzclient 2>/dev/null && echo "  killed gzclient" || echo "  no gzclient found"
sleep 2
echo "Done. You can now run: ros2 launch thermal_bringup sim_nav_slam_launch.py"
