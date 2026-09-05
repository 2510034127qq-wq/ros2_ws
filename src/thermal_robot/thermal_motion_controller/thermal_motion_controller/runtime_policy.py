"""Shared fast/slow admission rules, independent of ROS and wall-clock sources."""
import math
import numpy as np


def slow_output_usable(strategy, mode, health, stamp_s, now_s,
                       receipt_age_s=0., timeout_s=3.):
    if strategy!='dual' or mode!='online' or health!='ready':
        return False
    if not all(math.isfinite(x) for x in (stamp_s,now_s,receipt_age_s,timeout_s)):
        return False
    return 0<=now_s-stamp_s<=timeout_s and 0<=receipt_age_s<=timeout_s


def surface_approach_waypoint(robot_xy, source_xy, standoff_m, occupancy, robot_radius_m=.35):
    """Choose a known-free standoff footprint; Nav2 owns the route around walls."""
    if occupancy is None:return None
    from thermal_field_reconstructor.visibility import known_free_at
    angles=np.linspace(0.,2*math.pi,24,endpoint=False)
    points=np.asarray(source_xy)+standoff_m*np.column_stack((np.cos(angles),np.sin(angles)))
    offsets=robot_radius_m*np.column_stack((np.cos(angles),np.sin(angles)))
    footprint=points[:,None,:]+np.vstack((np.zeros((1,2)),offsets))[None,:,:]
    valid=known_free_at(occupancy,footprint[:,:,0],footprint[:,:,1]).all(axis=1)
    choices=points[valid]
    if not len(choices):return None
    point=choices[np.argmin(np.linalg.norm(choices-np.asarray(robot_xy),axis=1))]
    return float(point[0]),float(point[1])
