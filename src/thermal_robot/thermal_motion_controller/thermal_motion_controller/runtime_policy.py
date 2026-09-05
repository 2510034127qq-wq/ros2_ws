"""Shared fast/slow admission rules, independent of ROS and wall-clock sources."""
import math


def slow_output_usable(strategy, mode, health, stamp_s, now_s,
                       receipt_age_s=0., timeout_s=3.):
    if strategy!='dual' or mode!='online' or health!='ready':
        return False
    if not all(math.isfinite(x) for x in (stamp_s,now_s,receipt_age_s,timeout_s)):
        return False
    return 0<=now_s-stamp_s<=timeout_s and 0<=receipt_age_s<=timeout_s
