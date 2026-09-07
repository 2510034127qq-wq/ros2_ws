"""Shared fast/slow admission rules, independent of ROS and wall-clock sources."""
import math
import numpy as np


class CoverageSweep:
    """Full camera azimuth sweep, measured by odometry rather than elapsed time."""
    def __init__(self, distance_m=4., interval_s=60.):
        self.distance_m = distance_m
        self.interval_s = interval_s
        self.origin = None
        self.completed_s = -math.inf
        self.previous_yaw = None
        self.rotation = 0.

    def step(self, xy, yaw, now_s):
        if self.previous_yaw is None:
            due = (self.origin is None or now_s-self.completed_s >= self.interval_s
                   or math.dist(xy, self.origin) >= self.distance_m)
            if not due:
                return False
            self.previous_yaw, self.rotation = yaw, 0.
        else:
            delta = math.atan2(math.sin(yaw-self.previous_yaw), math.cos(yaw-self.previous_yaw))
            self.rotation += delta
            self.previous_yaw = yaw
        if self.rotation >= 2*math.pi-.05:
            self.origin, self.completed_s = tuple(xy), now_s
            self.previous_yaw = None
            return False
        return True


def exploration_goal_due(goal, xy, now_s, started_s, arrival_m=.6, timeout_s=45.):
    """Retain an exploration goal until actually reached or its deadline expires."""
    return (goal is None or math.dist(goal, xy) <= arrival_m
            or now_s-started_s >= timeout_s)


class ExplorationProgress:
    """Retarget a physically stuck robot, with a temporary failed-goal exclusion."""
    def __init__(self, timeout_s=12., cooldown_s=45., exclusion_m=2.):
        self.timeout_s, self.cooldown_s, self.exclusion_m = timeout_s, cooldown_s, exclusion_m
        self.xy, self.stamp_s = None, 0.
        self.failed = []

    def reset(self, xy, now_s):
        self.xy, self.stamp_s = tuple(xy), now_s

    def stalled(self, xy, now_s):
        if self.xy is None or math.dist(self.xy, xy) >= .25:
            self.reset(xy, now_s)
        return now_s-self.stamp_s >= self.timeout_s

    def reject(self, goal, now_s):
        self.failed.append((*goal, now_s+self.cooldown_s))

    def allowed(self, x, y, now_s):
        self.failed = [v for v in self.failed if v[2] > now_s]
        valid = np.ones(np.broadcast(x,y).shape, dtype=bool)
        for gx, gy, _ in self.failed:
            valid &= np.hypot(x-gx, y-gy) >= self.exclusion_m
        return valid


def slow_output_usable(strategy, mode, health, stamp_s, now_s,
                       receipt_age_s=0., timeout_s=3.):
    if strategy!='dual' or mode!='online' or health!='ready':
        return False
    if not all(math.isfinite(x) for x in (stamp_s,now_s,receipt_age_s,timeout_s)):
        return False
    return 0<=now_s-stamp_s<=timeout_s and 0<=receipt_age_s<=timeout_s


def surface_approach_waypoint(robot_xy, source_xy, standoff_m, occupancy,
                              robot_radius_m=.35, max_step_m=2.):
    """Choose a safe local step toward a source; Nav2 owns the route around walls.

    Prefer a final standoff within this step. Otherwise advance through a
    known-free footprint without requiring the distant standoff to be mapped.
    Strict distance progress prevents cycling between intermediate waypoints.
    """
    if occupancy is None:return None
    from thermal_field_reconstructor.visibility import known_free_at
    angles=np.linspace(0.,2*math.pi,24,endpoint=False)
    directions=np.column_stack((np.cos(angles),np.sin(angles)))
    robot,source=np.asarray(robot_xy),np.asarray(source_xy)
    points=np.vstack((source+standoff_m*directions,
                      robot+.5*max_step_m*directions,robot+max_step_m*directions))
    offsets=robot_radius_m*directions
    footprint=points[:,None,:]+np.vstack((np.zeros((1,2)),offsets))[None,:,:]
    valid=known_free_at(occupancy,footprint[:,:,0],footprint[:,:,1]).all(axis=1)
    travel=np.linalg.norm(points-robot,axis=1)
    remaining=np.linalg.norm(points-source,axis=1)
    valid &= ((travel>.5) & (travel<=max_step_m+1e-6)
              & (remaining>=standoff_m-1e-6)
              & (remaining<np.linalg.norm(robot-source)-.25))
    if valid[:24].any():
        choices=points[:24][valid[:24]]
        point=choices[np.argmin(np.linalg.norm(choices-robot,axis=1))]
        return float(point[0]),float(point[1])
    choices=points[valid]
    if not len(choices):return None
    point=choices[np.argmin(remaining[valid]+.2*travel[valid])]
    return float(point[0]),float(point[1])
