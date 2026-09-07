#!/usr/bin/env python3
"""
controller_node.py — Thermal Gradient Navigation Controller v31
================================================================
Generalizable multi-source thermal navigation algorithm.
Core principle: robot navigates purely from sensor signals, never knows
source coordinates a priori.

v31 fixes vs v30 (targeted, algorithm logic unchanged):
  FIX-1: Thermal navigation pose frame.
    The thermal simulator renders images from Gazebo/odom pose, so the world
    thermal map and controller use odom-aligned world coordinates by default.
    SLAM TF is still tracked for Nav2 map-goal conversion.

  FIX-2: COARSE_SURVEY direct navigation fallback.
    When Nav2 unavailable/failed, robot must still move toward waypoint.
    Added scan-guarded direct /cmd_vel fallback when nav2_state != NAV2_ACTIVE.
    This was the primary cause of robot stopping after ~40s.

  FIX-3: Nav2 goal frame correction.
    Nav2 expects goals in 'map' frame.
    Odom-world targets are converted through the live map pose when TF exists;
    the static spawn offset remains the fallback.

  FIX-4: FRONTIER_NAV direct fallback robustness.
    Fail-closed linear commands using fresh forward LaserScan evidence.

Generalizability design:
  - Keeps thermal projection and controller in the same physical pose frame
  - Works without SLAM (uses odom-based position)
  - Works without Nav2 when /scan is live (guarded direct /cmd_vel fallback)
  - Gradient ascent, Lévy flight, belief map logic unchanged
  - All state transitions driven by thermal sensor signals only

Architecture (layered):
  TARGET POLICY (source_seek):
    target_selection.py chooses frontier/coarse/departure world targets
    and execution preference; this node only executes phases and fallbacks.
  FINE states (direct /cmd_vel):
    ASCENT, CONVERGE, SAMPLE, AT_PEAK, RELOCATE, ESCAPE
  COARSE states (Nav2 preferred + scan-guarded direct fallback):
    FRONTIER_NAV, COARSE_SURVEY, DEPARTURE (direct only, target may be unmapped)

Refs:
  Macenski 2020 DOI:10.1109/IROS45743.2020.9341207 (Nav2)
  Macenski 2021 arXiv:2010.10195 (slam_toolbox)
  Sousa 2006    DOI:10.1109/ROBOT.2006.1642286 (gradient source seeking)
  Wiedemann 2021 DOI:10.1016/j.robot.2020.103687 (thermal gradient navigation)
"""

import math
import time
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from thermal_interfaces.msg import GradientArray, SourceEstimateArray, ThermalMap
from thermal_motion_controller.planning import (
    PlannerWeights,
    operational_waypoint_min_distance,
    select_residual_target,
)
from thermal_motion_controller.navigation_policy import (
    DIRECT_STOP,
    GOAL_KEEP,
    GOAL_REPLACE,
    GOAL_SEND,
    decide_direct_motion,
    decide_goal_action,
    forward_clearance,
)
from thermal_motion_controller.belief import source_information_gain
from thermal_motion_controller.runtime_policy import (
    slow_output_usable, surface_approach_waypoint, CoverageSweep, exploration_goal_due,
    ExplorationProgress)
from thermal_motion_controller.gp_ucb import GaussianProcessUCB, GPParams
from thermal_interfaces.msg import BeliefState
from thermal_motion_controller.target_selection import (
    EXECUTION_DIRECT_FIRST,
    EXECUTION_NAV2_PREFERRED,
    SourceSeekConfig,
    SourceSeekContext,
    StrategyTarget,
    SourceSeekTargetSelector,
)
from thermal_field_reconstructor import residual as fr_residual
from thermal_field_reconstructor import visibility as fr_visibility

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ExtrapolationException, ConnectivityException
from nav2_msgs.action import NavigateToPose

# ── State machine constants ────────────────────────────────────────────────
STATE_ASCENT        = 'ASCENT'
STATE_CONVERGE      = 'CONVERGE'
STATE_SAMPLE        = 'SAMPLE'
STATE_AT_PEAK       = 'AT_PEAK'
STATE_RELOCATE      = 'RELOCATE'
STATE_DEPARTURE     = 'DEPARTURE'
STATE_ESCAPE        = 'ESCAPE'
STATE_FRONTIER_NAV  = 'FRONTIER_NAV'
STATE_COARSE_SURVEY = 'COARSE_SURVEY'
STATE_SURVEY_PAUSE  = 'SURVEY_PAUSE'

NAV2_IDLE     = 'idle'
NAV2_SENDING  = 'sending'
NAV2_ACTIVE   = 'active'
NAV2_DONE     = 'done'

COARSE_STATES = {STATE_DEPARTURE, STATE_COARSE_SURVEY, STATE_FRONTIER_NAV}
FINE_STATES   = {STATE_ASCENT, STATE_CONVERGE, STATE_SAMPLE,
                 STATE_RELOCATE, STATE_ESCAPE, STATE_AT_PEAK}

EMODE_SOURCE_AVOID = 'source_avoid'
EMODE_CENTROID_RET = 'centroid_return'

_ESCAPE_EXIT_FACTOR    = 1.35
_N_CALIB               = 25
_WARM_TEMP_DELTA       = 3.0
_ASCENT_EXIT_GM_FACTOR = 0.5
_ASCENT_EXIT_TRISE     = 0.8


# ════════════════════════════════════════════════════════════════════════════
# ThermalBeliefMap  (unchanged from v30)
# ════════════════════════════════════════════════════════════════════════════

class ThermalBeliefMap:
    """
    Occupancy-like grid recording visit density and thermal observations.
    All coordinates are in world frame.
    """
    def __init__(self, center_x: float, center_y: float,
                 size_m: float = 50.0, resolution: float = 0.5):
        self.res = resolution
        self.n   = int(size_m / resolution)
        self.ox  = center_x - size_m / 2.0
        self.oy  = center_y - size_m / 2.0
        self.visit = np.zeros((self.n, self.n), dtype=np.float32)
        self.tmax  = np.zeros((self.n, self.n), dtype=np.float32)
        self.cold  = np.zeros((self.n, self.n), dtype=np.float32)
        self.excl  = np.zeros((self.n, self.n), dtype=bool)

    def _ci(self, wx, wy):
        i = int((wx - self.ox) / self.res)
        j = int((wy - self.oy) / self.res)
        return int(np.clip(i, 0, self.n-1)), int(np.clip(j, 0, self.n-1))

    def _world(self, i, j):
        return self.ox+(i+0.5)*self.res, self.oy+(j+0.5)*self.res

    def update(self, wx, wy, trise, visit_sigma=1.0, heat_sigma=1.2):
        ci, cj = self._ci(wx, wy)
        r_v = max(1, int(visit_sigma/self.res+0.5))
        i0,i1 = max(0,ci-r_v), min(self.n,ci+r_v+1)
        j0,j1 = max(0,cj-r_v), min(self.n,cj+r_v+1)
        ii,jj = np.mgrid[i0:i1,j0:j1]
        dx=(ii-ci).astype(np.float32)*self.res
        dy=(jj-cj).astype(np.float32)*self.res
        w=np.exp(-(dx*dx+dy*dy)/(2.0*visit_sigma**2))
        self.visit[i0:i1,j0:j1] += w
        if trise > 0.5:
            r_h=max(1,int(heat_sigma/self.res+0.5))
            hi0,hi1=max(0,ci-r_h),min(self.n,ci+r_h+1)
            hj0,hj1=max(0,cj-r_h),min(self.n,cj+r_h+1)
            hi,hj=np.mgrid[hi0:hi1,hj0:hj1]
            hdx=(hi-ci).astype(np.float32)*self.res
            hdy=(hj-cj).astype(np.float32)*self.res
            hw=trise*np.exp(-(hdx*hdx+hdy*hdy)/(2.0*heat_sigma**2))
            self.tmax[hi0:hi1,hj0:hj1]=np.maximum(
                self.tmax[hi0:hi1,hj0:hj1],hw.astype(np.float32))
        elif trise < 0.3:
            self.cold[ci,cj]=min(self.cold[ci,cj]+1.0,20.0)

    def mark_excluded(self, sx, sy, radius):
        ci,cj=self._ci(sx,sy)
        r=int(radius/self.res)+2
        i0,i1=max(0,ci-r),min(self.n,ci+r+1)
        j0,j1=max(0,cj-r),min(self.n,cj+r+1)
        ii,jj=np.mgrid[i0:i1,j0:j1]
        wx_g=self.ox+(ii+0.5)*self.res
        wy_g=self.oy+(jj+0.5)*self.res
        d=np.sqrt((wx_g-sx)**2+(wy_g-sy)**2)
        self.excl[i0:i1,j0:j1]|=(d<=radius)

    def suppress_confirmed_source(self, sx, sy, radius, suppression_visit=5000.0):
        ci, cj = self._ci(sx, sy)
        r = int(radius / self.res) + 2
        i0, i1 = max(0, ci-r), min(self.n, ci+r+1)
        j0, j1 = max(0, cj-r), min(self.n, cj+r+1)
        ii, jj = np.mgrid[i0:i1, j0:j1]
        wx_g = self.ox + (ii+0.5)*self.res
        wy_g = self.oy + (jj+0.5)*self.res
        d = np.sqrt((wx_g-sx)**2+(wy_g-sy)**2)
        mask = d <= radius
        self.tmax[i0:i1, j0:j1][mask]  = 0.0
        self.visit[i0:i1, j0:j1][mask] += suppression_visit

    def best_frontier(self, robot_wx, robot_wy,
                      min_d=3.0, max_d=14.0, dist_sigma=8.0,
                      heat_prior=0.3,
                      known_sources=None, safe_dist=4.5):
        n=self.n
        ii,jj=np.mgrid[0:n,0:n]
        wx_g=(self.ox+(ii+0.5)*self.res).astype(np.float32)
        wy_g=(self.oy+(jj+0.5)*self.res).astype(np.float32)
        dist=np.sqrt((wx_g-robot_wx)**2+(wy_g-robot_wy)**2)
        valid=(~self.excl)&(dist>=min_d)&(dist<=max_d)
        if known_sources:
            for sx,sy in known_sources:
                d_src=np.sqrt((wx_g-sx)**2+(wy_g-sy)**2)
                valid&=(d_src>=safe_dist)
        if not valid.any():
            if known_sources and safe_dist>3.0:
                return self.best_frontier(robot_wx,robot_wy,min_d,max_d,
                    dist_sigma,heat_prior,known_sources,safe_dist*0.6)
            return None
        novelty=1.0/(1.0+self.visit)
        cold_penalty=np.maximum(0.0,1.0-self.cold*0.25)
        dist_factor=np.exp(-0.5*(dist/dist_sigma)**2)
        score=(self.tmax+heat_prior)*novelty*cold_penalty*dist_factor
        score[~valid]=-1.0
        if score.max()<0.0:
            return None
        bi,bj=np.unravel_index(score.argmax(),score.shape)
        bwx,bwy=self._world(int(bi),int(bj))
        return bwx,bwy,float(score[bi,bj])

    def region_novelty(self, robot_wx, robot_wy, radius=5.0):
        n=self.n
        ii,jj=np.mgrid[0:n,0:n]
        wx_g=(self.ox+(ii+0.5)*self.res).astype(np.float32)
        wy_g=(self.oy+(jj+0.5)*self.res).astype(np.float32)
        dist=np.sqrt((wx_g-robot_wx)**2+(wy_g-robot_wy)**2)
        mask=(dist<=radius)&(~self.excl)
        if not mask.any():
            return 0.0
        return float((1.0/(1.0+self.visit[mask])).mean())


# ════════════════════════════════════════════════════════════════════════════
# ControllerNode v31
# ════════════════════════════════════════════════════════════════════════════

class ControllerNode(Node):

    def __init__(self):
        super().__init__('controller_node')

        # ── Parameter declarations (identical to v30) ─────────────────────
        self.declare_parameter('publish_rate',               10.0)
        self.declare_parameter('max_linear_vel',             0.28)
        self.declare_parameter('max_angular_vel',            0.5)
        self.declare_parameter('kp_angular',                 1.2)
        self.declare_parameter('ang_smooth_alpha',           0.55)
        self.declare_parameter('lin_smooth_alpha',           0.4)
        self.declare_parameter('min_gradient_mag',           0.15)
        self.declare_parameter('peak_temp_delta',           15.0)
        self.declare_parameter('ambient_temp',              -1.0)
        self.declare_parameter('ambient_update_margin',      1.5)
        self.declare_parameter('rearm_cool_delta',           6.0)
        self.declare_parameter('relocate_dist',              3.0)
        self.declare_parameter('relocate_lin_vel',           0.2)
        self.declare_parameter('peak_hold_s',                1.5)
        self.declare_parameter('known_source_radius', 2.0)
        self.declare_parameter('source_repulsion_k',         0.8)
        self.declare_parameter('source_repulsion_min_dist',  0.5)
        self.declare_parameter('source_exclusion_radius',    2.0)
        self.declare_parameter('min_escape_dist',            2.0)
        self.declare_parameter('spawn_x',                   -6.0)
        self.declare_parameter('spawn_y',                    0.0)
        self.declare_parameter('thermal_pose_source',       'odom')
        self.declare_parameter('stuck_timeout',             90.0)
        self.declare_parameter('stuck_dist_thresh',          0.5)
        self.declare_parameter('min_ascent_temp_rise',       0.5)
        self.declare_parameter('thermal_lost_temp_thresh',   0.8)
        self.declare_parameter('thermal_lost_timeout',      20.0)
        self.declare_parameter('centroid_arrival_r',         3.0)
        self.declare_parameter('max_search_radius',         10.0)
        self.declare_parameter('negative_trise_bail',        0.3)
        self.declare_parameter('frontier_update_interval',   8.0)
        self.declare_parameter('frontier_arrival_r',         2.5)
        self.declare_parameter('frontier_nav_lin_vel',       0.25)
        self.declare_parameter('levy_rounds_trigger',        2)
        self.declare_parameter('levy_region_novelty_thr',    0.15)
        self.declare_parameter('levy_mu',                    1.5)
        self.declare_parameter('levy_scale',                 3.0)
        self.declare_parameter('levy_min_step',              3.0)
        self.declare_parameter('levy_max_step',             12.0)
        self.declare_parameter('escape_frontier_bias',       0.35)
        self.declare_parameter('frontier_safe_buf',          0.3)
        self.declare_parameter('heat_sigma',                 1.2)
        self.declare_parameter('escape_loop_max',            3)
        self.declare_parameter('escape_loop_window_s',      30.0)
        self.declare_parameter('pre_peak_thresh_ratio',      0.25)
        self.declare_parameter('converge_lin_vel',           0.10)
        self.declare_parameter('converge_max_s',            90.0)
        self.declare_parameter('sample_hold_s',              3.0)
        self.declare_parameter('converge_to_sample_ratio',  0.65)
        self.declare_parameter('converge_cold_exit_s',       8.0)
        self.declare_parameter('sample_center_pixel_radius', 12)
        self.declare_parameter('converge_sticky_max',         3)
        self.declare_parameter('converge_sticky_return_vel',  0.18)
        self.declare_parameter('converge_best_of_n_ratio',    0.90)
        self.declare_parameter('post_confirm_rounds',        10)
        self.declare_parameter('post_confirm_min_d',          7.0)
        self.declare_parameter('post_confirm_dist_sigma',    15.0)
        self.declare_parameter('random_seed',                 0)
        self.declare_parameter('strategy',                    'full')
        self.declare_parameter('residual_planner_min_evidence', 1.5)
        self.declare_parameter('residual_planner_footprint_radius', 2.0)
        self.declare_parameter('residual_planner_top_k', 64)
        self.declare_parameter('residual_waypoint_min_d', 3.0)
        self.declare_parameter('sample_min_trise',            8.0)
        self.declare_parameter('post_confirm_cooldown_s',   60.0)
        self.declare_parameter('adaptive_thresholds_enabled', True)
        self.declare_parameter('adapt_pk_ratio',              0.80)
        self.declare_parameter('adapt_sample_ratio',          0.65)
        self.declare_parameter('adapt_min_signal',            5.0)
        self.declare_parameter('post_confirm_departure_dist', 12.0)
        self.declare_parameter('survey_pause_interval',      20.0)
        self.declare_parameter('survey_sense_s',              4.0)
        self.declare_parameter('survey_detect_thresh',        1.5)
        self.declare_parameter('coarse_transit_detect_thresh', 5.0)
        self.declare_parameter('survey_waypoint_min_d',       5.0)
        self.declare_parameter('survey_waypoint_max_d',      14.0)
        self.declare_parameter('survey_novelty_safe_dist',    4.0)
        self.declare_parameter('survey_waypoint_timeout_s',  60.0)
        self.declare_parameter('fine_mode_entry_thresh',      2.0)
        self.declare_parameter('frontier_cold_timeout_s',    32.0)
        self.declare_parameter('departure_timeout_s',       120.0)
        self.declare_parameter('departure_speed',             0.25)
        self.declare_parameter('departure_progress_timeout_s', 8.0)
        self.declare_parameter('departure_progress_min_delta', 0.25)
        self.declare_parameter('planner_information_gain_weight', 1.0)
        self.declare_parameter('planner_source_probability_weight', 1.4)
        self.declare_parameter('planner_coverage_gain_weight', 0.7)
        self.declare_parameter('planner_travel_cost_weight', 0.45)
        self.declare_parameter('planner_duplicate_penalty_weight', 1.2)
        self.declare_parameter('planner_risk_penalty_weight', 0.2)
        self.declare_parameter('planner_map_stale_s', 5.0)
        self.declare_parameter('planner_candidate_verify_radius', 3.5)
        self.declare_parameter('tracker_verify_probability', 0.75)
        self.declare_parameter('tracker_verify_observations', 5)
        self.declare_parameter('tracker_verify_max_dist', 3.8)
        self.declare_parameter('tracker_verify_arrival_r', 1.2)
        self.declare_parameter('tracker_verify_min_elapsed_s', 6.0)
        self.declare_parameter('tracker_approach_lin_vel', 0.18)
        self.declare_parameter('post_confirm_guard_near_dist', 6.0)
        self.declare_parameter('departure_directional_weight', 0.55)
        self.declare_parameter('coverage_directional_weight', 0.65)
        self.declare_parameter('coverage_ring_min_d', 4.0)
        self.declare_parameter('coverage_ring_max_d', 9.0)
        self.declare_parameter('coverage_ring_fov_radius', 3.0)
        self.declare_parameter('coverage_ring_angles', 24)
        self.declare_parameter('coverage_ring_rings', 3)
        self.declare_parameter('coverage_recent_yaw_penalty', 0.45)
        self.declare_parameter('coverage_recent_yaw_window', 5)
        self.declare_parameter('source_set_expansion_min_sources', 2)
        self.declare_parameter('source_set_expansion_max_d', 13.0)
        self.declare_parameter('source_set_expansion_directional_weight', 0.35)
        self.declare_parameter('source_set_outward_directional_weight', 0.85)
        self.declare_parameter('source_set_outward_bonus', 0.35)
        self.declare_parameter('source_set_lateral_directional_weight', 0.95)
        self.declare_parameter('source_set_lateral_bonus', 0.45)
        self.declare_parameter('source_set_lateral_max_d', 12.0)
        self.declare_parameter('source_set_direct_first_s', 32.0)
        self.declare_parameter('post_confirm_direct_first_s', 24.0)
        self.declare_parameter('post_confirm_immediate_departure', True)
        self.declare_parameter('nav2_progress_timeout_s', 4.0)
        self.declare_parameter('nav2_progress_min_delta', 0.35)
        self.declare_parameter('nav2_stall_direct_s', 16.0)
        self.declare_parameter('nav2_goal_min_interval_s', 3.0)
        self.declare_parameter('direct_scan_stop_m', 0.65)
        self.declare_parameter('direct_scan_stale_s', 0.6)
        self.declare_parameter('direct_scan_half_angle_deg', 30.0)

        # ── Read parameters ───────────────────────────────────────────────
        g = self.get_parameter
        self._rate         = float(g('publish_rate').value)
        self._max_lin      = float(g('max_linear_vel').value)
        self._max_ang      = float(g('max_angular_vel').value)
        self._kp_ang       = float(g('kp_angular').value)
        self._alpha_ang    = float(g('ang_smooth_alpha').value)
        self._alpha_lin    = float(g('lin_smooth_alpha').value)
        self._min_gmag     = float(g('min_gradient_mag').value)
        self._pk_tdelta    = float(g('peak_temp_delta').value)
        _ambient_param     = float(g('ambient_temp').value)
        self._amb_margin   = float(g('ambient_update_margin').value)
        self._rearm_delta  = float(g('rearm_cool_delta').value)
        self._reloc_dist   = float(g('relocate_dist').value)
        self._reloc_lin    = float(g('relocate_lin_vel').value)
        self._peak_hold    = float(g('peak_hold_s').value)
        self._rev_r        = float(g('known_source_radius').value)
        self._rep_k        = float(g('source_repulsion_k').value)
        self._rep_min      = float(g('source_repulsion_min_dist').value)
        self._excl_r       = float(g('source_exclusion_radius').value)
        self._min_esc_dist = float(g('min_escape_dist').value)
        self._spawn_x      = float(g('spawn_x').value)
        self._spawn_y      = float(g('spawn_y').value)
        self._pose_source  = str(g('thermal_pose_source').value).lower()
        self._stuck_to     = float(g('stuck_timeout').value)
        self._stuck_thr    = float(g('stuck_dist_thresh').value)
        self._min_asc_rise = float(g('min_ascent_temp_rise').value)
        self._lost_t_thr   = float(g('thermal_lost_temp_thresh').value)
        self._lost_timeout = float(g('thermal_lost_timeout').value)
        self._centroid_arr = float(g('centroid_arrival_r').value)
        self._max_srch_r   = float(g('max_search_radius').value)
        self._neg_bail     = float(g('negative_trise_bail').value)
        self._frontier_upd = float(g('frontier_update_interval').value)
        self._frontier_r   = float(g('frontier_arrival_r').value)
        self._frontier_lin = float(g('frontier_nav_lin_vel').value)
        self._levy_rnd_thr = int(g('levy_rounds_trigger').value)
        self._levy_nov_thr = float(g('levy_region_novelty_thr').value)
        self._levy_mu      = float(g('levy_mu').value)
        self._levy_scale   = float(g('levy_scale').value)
        self._levy_min     = float(g('levy_min_step').value)
        self._levy_max     = float(g('levy_max_step').value)
        self._esc_fr_bias  = float(g('escape_frontier_bias').value)
        self._fr_safe_buf  = float(g('frontier_safe_buf').value)
        self._heat_sigma   = float(g('heat_sigma').value)
        self._esc_loop_max = int(g('escape_loop_max').value)
        self._esc_loop_win = float(g('escape_loop_window_s').value)
        self._pre_pk_ratio = float(g('pre_peak_thresh_ratio').value)
        self._conv_lin     = float(g('converge_lin_vel').value)
        self._conv_max_s   = float(g('converge_max_s').value)
        self._sample_hold  = float(g('sample_hold_s').value)
        self._conv_sample_ratio  = float(g('converge_to_sample_ratio').value)
        self._conv_cold_exit_s   = float(g('converge_cold_exit_s').value)
        self._sample_center_px_r = int(g('sample_center_pixel_radius').value)
        self._conv_sticky_max     = int(g('converge_sticky_max').value)
        self._conv_sticky_ret_v   = float(g('converge_sticky_return_vel').value)
        self._conv_bon_ratio      = float(g('converge_best_of_n_ratio').value)
        self._pc_rounds           = int(g('post_confirm_rounds').value)
        self._pc_min_d            = float(g('post_confirm_min_d').value)
        self._pc_dist_sigma       = float(g('post_confirm_dist_sigma').value)
        self._random_seed         = int(g('random_seed').value)
        self._strategy_mode       = str(g('strategy').value or 'full')
        if self._strategy_mode not in ('full', 'frontier', 'levy', 'residual', 'fast', 'dual', 'gp_ucb'):
            self.get_logger().warn(f'unknown strategy={self._strategy_mode}, using full')
            self._strategy_mode = 'full'
        for name,default in [('sensor_model','a'),('slow_timeout_s',3.),

                             ('residual_enabled',True),
                             ('posterior_information_weight',1.),('surface_standoff_m',1.2),
                             ('posterior_detection_probability',.85),('posterior_false_alarm_probability',.03),
                             ('posterior_measurement_variance',.2),('surface_robot_radius_m',.35),
                             ('surface_waypoint_step_m',2.),
                             ('surface_blocked_wait_s',.5),
                             ('surface_approach_speed',.2),('surface_confirm_hold_s',1.),
                             ('surface_approach_timeout_s',20.),('surface_retry_cooldown_s',20.),
                             ('exploration_goal_timeout_s',45.),('exploration_arrival_m',.6),
                             ('camera_sweep_speed_rad_s',.55),('camera_sweep_distance_m',4.),
                             ('camera_sweep_interval_s',60.),
                             ('exploration_stall_s',12.),('exploration_retry_s',45.),
                             ('exploration_failed_radius_m',2.),
                             ('exploration_turn_weight',.25),
                             ('surface_source_max_age_s',1.5),('thermal_image_width',64),
                             ('thermal_image_height',48),('thermal_fov_x_m',4.),('thermal_fov_y_m',3.),

                             ('gp_max_samples',64),
                             ('gp_max_candidates',256),('gp_length_scale_m',2.),('gp_beta',2.),
                             ('gp_signal_std_c',12.),('gp_noise_std_c',1.),('gp_travel_weight',.2)]:
            self.declare_parameter(name,default)
        self._sensor_model=str(g('sensor_model').value)
        self._slow_timeout=float(g('slow_timeout_s').value)
        self._slow_msg=None;self._slow_t=-float('inf')
        self._residual_enabled=bool(g('residual_enabled').value)
        self._posterior_weight=float(g('posterior_information_weight').value)
        self._posterior_pd=float(g('posterior_detection_probability').value)
        self._posterior_pf=float(g('posterior_false_alarm_probability').value)
        self._posterior_variance=float(g('posterior_measurement_variance').value)
        self._gp=GaussianProcessUCB(GPParams(max_samples=int(g('gp_max_samples').value),
            max_candidates=int(g('gp_max_candidates').value),length_scale_m=float(g('gp_length_scale_m').value),
            beta=float(g('gp_beta').value),signal_std_c=float(g('gp_signal_std_c').value),
            noise_std_c=float(g('gp_noise_std_c').value),travel_weight=float(g('gp_travel_weight').value)))
        self._surface_standoff=float(g('surface_standoff_m').value)
        self._surface_speed=float(g('surface_approach_speed').value)
        self._surface_hold=float(g('surface_confirm_hold_s').value)
        self._surface_max_age=float(g('surface_source_max_age_s').value)
        self._surface_robot_radius=float(g('surface_robot_radius_m').value)
        self._surface_waypoint_step=float(g('surface_waypoint_step_m').value)
        self._surface_blocked_wait=float(g('surface_blocked_wait_s').value)
        self._surface_blocked_since=None
        self._surface_seen_ids=set();self._surface_target_id=None;self._surface_hold_start=None
        self._surface_wp=None;self._surface_plan_t=-float('inf')
        self._surface_nav_goal=None;self._surface_deferred={}
        self._surface_approach_timeout=float(g('surface_approach_timeout_s').value)
        self._surface_retry_cooldown=float(g('surface_retry_cooldown_s').value)
        self._explore_timeout=float(g('exploration_goal_timeout_s').value)
        self._explore_arrival=float(g('exploration_arrival_m').value)
        self._explore_turn_weight=float(g('exploration_turn_weight').value)
        self._sweep_speed=min(self._max_ang,float(g('camera_sweep_speed_rad_s').value))
        self._camera_sweep=CoverageSweep(float(g('camera_sweep_distance_m').value),
                                         float(g('camera_sweep_interval_s').value))
        self._sweep_active=False
        self._explore_progress=ExplorationProgress(float(g('exploration_stall_s').value),
            float(g('exploration_retry_s').value),float(g('exploration_failed_radius_m').value))
        self._residual_planner_min_evidence = float(
            g('residual_planner_min_evidence').value)
        self._residual_planner_footprint_radius = float(
            g('residual_planner_footprint_radius').value)
        self._residual_planner_top_k = max(1, int(
            g('residual_planner_top_k').value))
        self._residual_waypoint_min_d = float(g('residual_waypoint_min_d').value)
        self._occ_view = None
        if self._random_seed > 0:
            random.seed(self._random_seed)
            np.random.seed(self._random_seed % (2**31))
        self._sample_min_trise    = float(g('sample_min_trise').value)
        self._pc_cooldown_s       = float(g('post_confirm_cooldown_s').value)
        self._adaptive_thresh     = bool(g('adaptive_thresholds_enabled').value)
        self._adapt_pk_ratio      = float(g('adapt_pk_ratio').value)
        self._adapt_samp_ratio    = float(g('adapt_sample_ratio').value)
        self._adapt_min_signal    = float(g('adapt_min_signal').value)
        self._departure_dist      = float(g('post_confirm_departure_dist').value)
        self._survey_pause_ivl    = float(g('survey_pause_interval').value)
        self._survey_sense_s      = float(g('survey_sense_s').value)
        self._survey_det_thresh   = float(g('survey_detect_thresh').value)
        self._coarse_transit_thr  = float(g('coarse_transit_detect_thresh').value)
        self._survey_wp_min_d     = float(g('survey_waypoint_min_d').value)
        self._survey_wp_max_d     = float(g('survey_waypoint_max_d').value)
        self._survey_safe_dist    = float(g('survey_novelty_safe_dist').value)
        self._survey_wp_timeout   = float(g('survey_waypoint_timeout_s').value)
        self._fine_entry_thresh   = float(g('fine_mode_entry_thresh').value)
        self._frontier_cold_to    = float(g('frontier_cold_timeout_s').value)
        self._departure_timeout   = float(g('departure_timeout_s').value)
        self._departure_speed     = float(g('departure_speed').value)
        self._departure_progress_timeout_s = float(g('departure_progress_timeout_s').value)
        self._departure_progress_min_delta = float(g('departure_progress_min_delta').value)
        self._planner_weights = PlannerWeights(
            information_gain=float(g('planner_information_gain_weight').value),
            source_probability=float(g('planner_source_probability_weight').value),
            coverage_gain=float(g('planner_coverage_gain_weight').value),
            travel_cost=float(g('planner_travel_cost_weight').value),
            duplicate_penalty=float(g('planner_duplicate_penalty_weight').value),
            risk_penalty=float(g('planner_risk_penalty_weight').value),
        )
        self._planner_map_stale_s = float(g('planner_map_stale_s').value)
        self._candidate_verify_radius = float(g('planner_candidate_verify_radius').value)
        self._tracker_verify_prob = float(g('tracker_verify_probability').value)
        self._tracker_verify_obs = int(g('tracker_verify_observations').value)
        self._tracker_verify_max_d = float(g('tracker_verify_max_dist').value)
        self._tracker_verify_arrival_r = float(g('tracker_verify_arrival_r').value)
        self._tracker_verify_min_elapsed_s = float(g('tracker_verify_min_elapsed_s').value)
        self._tracker_approach_lin = float(g('tracker_approach_lin_vel').value)
        self._pc_guard_near_dist = float(g('post_confirm_guard_near_dist').value)
        self._departure_directional_weight = float(g('departure_directional_weight').value)
        self._coverage_directional_weight = float(g('coverage_directional_weight').value)
        self._coverage_ring_min_d = float(g('coverage_ring_min_d').value)
        self._coverage_ring_max_d = float(g('coverage_ring_max_d').value)
        self._coverage_ring_fov_radius = float(g('coverage_ring_fov_radius').value)
        self._coverage_ring_angles = int(g('coverage_ring_angles').value)
        self._coverage_ring_rings = int(g('coverage_ring_rings').value)
        self._coverage_recent_yaw_penalty = float(g('coverage_recent_yaw_penalty').value)
        self._coverage_recent_yaw_window = max(1, int(g('coverage_recent_yaw_window').value))
        self._source_set_expansion_min_sources = max(2, int(g('source_set_expansion_min_sources').value))
        self._source_set_expansion_max_d = float(g('source_set_expansion_max_d').value)
        self._source_set_expansion_directional_weight = float(g('source_set_expansion_directional_weight').value)
        self._source_set_outward_directional_weight = float(g('source_set_outward_directional_weight').value)
        self._source_set_outward_bonus = float(g('source_set_outward_bonus').value)
        self._source_set_lateral_directional_weight = float(g('source_set_lateral_directional_weight').value)
        self._source_set_lateral_bonus = float(g('source_set_lateral_bonus').value)
        self._source_set_lateral_max_d = float(g('source_set_lateral_max_d').value)
        self._source_set_direct_first_s = float(g('source_set_direct_first_s').value)
        self._post_confirm_direct_first_s = float(g('post_confirm_direct_first_s').value)
        self._post_confirm_immediate_departure = bool(g('post_confirm_immediate_departure').value)
        self._nav2_progress_timeout_s = float(g('nav2_progress_timeout_s').value)
        self._nav2_progress_min_delta = float(g('nav2_progress_min_delta').value)
        self._nav2_stall_direct_s = float(g('nav2_stall_direct_s').value)
        self._nav2_goal_min_interval_s = float(g('nav2_goal_min_interval_s').value)
        self._direct_scan_stop_m = float(g('direct_scan_stop_m').value)
        self._direct_scan_stale_s = float(g('direct_scan_stale_s').value)
        self._direct_scan_half_angle = math.radians(float(
            g('direct_scan_half_angle_deg').value))

        self._source_seek_selector = SourceSeekTargetSelector(SourceSeekConfig(
            source_repulsion_k=self._rep_k,
            source_repulsion_min_dist=self._rep_min,
            source_exclusion_radius=self._excl_r,
            frontier_safe_buf=self._fr_safe_buf,
            survey_safe_dist=self._survey_safe_dist,
            pc_min_d=self._pc_min_d,
            pc_dist_sigma=self._pc_dist_sigma,
            coverage_directional_weight=self._coverage_directional_weight,
            departure_directional_weight=self._departure_directional_weight,
            coverage_ring_min_d=self._coverage_ring_min_d,
            coverage_ring_max_d=self._coverage_ring_max_d,
            coverage_ring_fov_radius=self._coverage_ring_fov_radius,
            coverage_ring_angles=self._coverage_ring_angles,
            coverage_ring_rings=self._coverage_ring_rings,
            coverage_recent_yaw_penalty=self._coverage_recent_yaw_penalty,
            coverage_recent_yaw_window=self._coverage_recent_yaw_window,
            source_set_expansion_min_sources=self._source_set_expansion_min_sources,
            source_set_expansion_max_d=self._source_set_expansion_max_d,
            source_set_expansion_directional_weight=self._source_set_expansion_directional_weight,
            source_set_outward_directional_weight=self._source_set_outward_directional_weight,
            source_set_outward_bonus=self._source_set_outward_bonus,
            source_set_lateral_directional_weight=self._source_set_lateral_directional_weight,
            source_set_lateral_bonus=self._source_set_lateral_bonus,
            source_set_lateral_max_d=self._source_set_lateral_max_d,
            departure_dist=self._departure_dist,
            planner_weights=self._planner_weights,
            planner_map_stale_s=self._planner_map_stale_s,
        ))


        # ── Ambient temperature calibration ───────────────────────────────
        if _ambient_param > 0:
            self._ambient_est = _ambient_param
            self._calib_done  = True
            self._calib_buf: list = []
        else:
            self._ambient_est = 25.0
            self._calib_done  = False
            self._calib_buf   = []
        self._ambient_ema_alpha = 0.02

        # ── Position state ────────────────────────────────────────────────
        # All world coords in Gazebo world frame.
        # FIX-1: SLAM TF gives map frame coords. world = map + spawn.
        # When TF unavailable, fallback to spawn + odom integration.
        self._odom_x   = 0.0
        self._odom_y   = 0.0
        self._odom_yaw = 0.0
        self._map_x    = 0.0
        self._map_y    = 0.0
        self._map_yaw  = 0.0
        self._wx       = self._spawn_x   # world X (updated from TF or odom)
        self._wy       = self._spawn_y   # world Y
        self._tf_ready = False
        self._prev_pos: Optional[Tuple[float,float]] = None
        self._path_length = 0.0

        # ── Perception ────────────────────────────────────────────────────
        self._last_ga: Optional[GradientArray] = None
        self._thermal_map: Optional[Dict] = None
        self._thermal_map_t: float = 0.0
        self._tracker_sources: List[Dict] = []
        self._tracker_sources_t: float = 0.0
        self._temp_max_seen = 25.0
        self._ang_smooth = 0.0
        self._lin_smooth = 0.0

        # ── Mission state ─────────────────────────────────────────────────
        self._found_sources: List[Tuple[float,float,float]] = []
        self._peak_cand_t: Optional[float] = None
        self._peak_armed   = True
        self._t0           = time.monotonic()

        # ── State machine ─────────────────────────────────────────────────
        self._state         = STATE_FRONTIER_NAV
        self._state_t       = time.monotonic()
        self._search_rounds = 0
        self._locked_yaw: Optional[float]               = None
        self._move_start:  Optional[Tuple[float,float]] = None
        self._escape_mode: str                          = EMODE_SOURCE_AVOID

        # ── CONVERGE ─────────────────────────────────────────────────────
        self._converge_best_T: float = 0.0
        self._converge_best_pos: Optional[Tuple[float,float]] = None
        self._conv_cold_t: Optional[float] = None
        self._conv_sticky_count:    int   = 0
        self._conv_best_global_T:   float = 0.0
        self._conv_best_global_pos: Optional[Tuple[float,float]] = None
        self._conv_returning:       bool  = False

        # ── SAMPLE ───────────────────────────────────────────────────────
        self._sample_t_start: Optional[float] = None
        self._sample_T_buf: List[float] = []

        # ── Post-confirm guard ────────────────────────────────────────────
        self._pc_guard_t: float = 0.0
        self._T_max_unconf: float = 25.0
        self._pc_rounds_left: int = 0

        # ── DEPARTURE ────────────────────────────────────────────────────
        self._departure_wp: Optional[Tuple[float,float]] = None
        self._departure_progress_goal = None
        self._departure_progress_best_d = float('inf')
        self._departure_progress_t = 0.0
        self._departure_direct_safe: bool = False

        # ── COARSE_SURVEY ─────────────────────────────────────────────────
        self._coarse_wp: Optional[Tuple[float,float]] = None
        self._coarse_wp_t: float = 0.0
        self._last_survey_pause_t: float = 0.0
        self._survey_buf: List[float] = []
        self._survey_t_start: Optional[float] = None
        self._coarse_wp_count: int = 0
        self._coarse_wp_execution_hint: str = 'nav2_preferred'
        self._coarse_wp_direct_safe: bool = False

        # ── Cold field detection ──────────────────────────────────────────
        self._frontier_cold_start: Optional[float] = None

        # ── Thermal tracking ─────────────────────────────────────────────
        self._last_warm_pos: Optional[Tuple[float,float]] = None
        self._thermal_lost_start: Optional[float]         = None

        # ── Stuck + ESCAPE loop detection ────────────────────────────────
        self._stuck_pos: Tuple[float,float] = (self._spawn_x, self._spawn_y)
        self._stuck_t   = time.monotonic()
        self._esc_loop_count: int    = 0
        self._esc_loop_last_t: float = 0.0

        # ── Belief map + frontier navigation ─────────────────────────────
        # Centered at spawn; all operations in world frame.
        self._bmap = ThermalBeliefMap(
            center_x=self._spawn_x, center_y=self._spawn_y,
            size_m=50.0, resolution=0.5)
        self._frontier_target: Optional[Tuple[float,float]] = None
        self._frontier_direct_safe: bool = False
        self._frontier_last_upd: float = 0.0

        # ── Local direct-motion guard ──────────────────────────────────────
        self._front_clearance_m: Optional[float] = None
        self._front_scan_t: float = 0.0
        self._direct_guard_mode: Dict[str, str] = {}
        self._direct_guard_log_t: Dict[str, float] = {}

        # ── [v30] TF2 for SLAM position ───────────────────────────────────
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ── [v30] Nav2 NavigateToPose Action Client ───────────────────────
        self._nav2_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._nav2_state: str                   = NAV2_IDLE
        self._nav2_goal_handle                  = None
        self._nav2_current_goal: Optional[Tuple[float,float]] = None
        self._nav2_last_send_t: float           = 0.0
        self._nav2_goal_generation: int         = 0
        self._nav2_ready: bool = False
        self._nav2_check_t: float = 0.0
        self._nav2_progress_goal = None
        self._nav2_progress_best_d = float('inf')
        self._nav2_progress_t = 0.0
        self._nav2_direct_until = 0.0

        # ── QoS ──────────────────────────────────────────────────────────
        be = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        history=QoSHistoryPolicy.KEEP_LAST, depth=5,
                        durability=QoSDurabilityPolicy.VOLATILE)
        rel = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=5,
                         durability=QoSDurabilityPolicy.VOLATILE)

        self.create_subscription(GradientArray, '/thermal/gradient', self._grad_cb, rel)
        self.create_subscription(ThermalMap,    '/thermal/map',      self._map_cb,  rel)
        self.create_subscription(SourceEstimateArray, '/thermal/sources', self._sources_cb, rel)
        self.create_subscription(Odometry,      '/odom',             self._odom_cb, be)
        self.create_subscription(LaserScan,     '/scan',             self._scan_cb, be)
        occ_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, '/map', self._occ_map_cb, occ_qos)
        self.create_subscription(BeliefState,'/thermal/belief',self._belief_cb,3)
        self._pub   = self.create_publisher(Twist, '/cmd_vel', 10)
        self._last_plan_ms=0.
        self._runtime_timer=self.create_timer(2.,self._report_runtime)
        self._timer = self.create_timer(1.0/self._rate, self._timer_cb)

        self.get_logger().info(
            f'controller_node v31 | FIX: SLAM coord offset + COARSE fallback + Nav2 frame | '
            f'COARSE→Nav2+fallback | FINE→direct /cmd_vel | '
            f'pose_source={self._pose_source} | '
            f'spawn=({self._spawn_x},{self._spawn_y}) | static inspection')
        self.get_logger().info(
            f'strategy={self._strategy_mode} random_seed={self._random_seed}')

    # ────────────────────────────────────────────────────────────────────────
    # Pose update: thermal world can use odom; Nav2 still gets map goals.
    # ────────────────────────────────────────────────────────────────────────

    def _update_world_pos_from_tf(self) -> bool:
        """
        Update robot pose in the thermal world frame.

        The simulated thermal image is generated from the Gazebo/odom pose, so
        the mapper and controller default to odom-aligned world coordinates.
        TF is still tracked for Nav2 map-goal conversion.
        """
        try:
            t = self._tf_buffer.lookup_transform(
                'map', 'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))

            self._map_x = t.transform.translation.x
            self._map_y = t.transform.translation.y

            q = t.transform.rotation
            siny = 2.0*(q.w*q.z + q.x*q.y)
            cosy = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
            self._map_yaw = math.atan2(siny, cosy)
            if self._pose_source == 'tf':
                self._wx = self._map_x + self._spawn_x
                self._wy = self._map_y + self._spawn_y
                self._odom_yaw = self._map_yaw
            else:
                self._wx = self._spawn_x + self._odom_x
                self._wy = self._spawn_y + self._odom_y

            if not self._tf_ready:
                self._tf_ready = True
                self.get_logger().info(
                    f'[TF_READY] SLAM TF (map→base_link) available '
                    f'pose_source={self._pose_source} '
                    f'world_pos=({self._wx:.2f},{self._wy:.2f}) '
                    f'[map=({self._map_x:.2f},{self._map_y:.2f}) '
                    f'odom=({self._odom_x:.2f},{self._odom_y:.2f})]')
            return True

        except (LookupException, ExtrapolationException, ConnectivityException):
            self._wx = self._spawn_x + self._odom_x
            self._wy = self._spawn_y + self._odom_y
            return False
        except Exception as e:
            if not hasattr(self, '_tf_warn_logged'):
                self._tf_warn_logged = True
                self.get_logger().warn(f'[TF] Unexpected error: {e}, falling back to odom')
            self._wx = self._spawn_x + self._odom_x
            self._wy = self._spawn_y + self._odom_y
            return False

    # ────────────────────────────────────────────────────────────────────────
    # [FIX-3] Nav2 Action Client: world frame → map frame conversion
    # ────────────────────────────────────────────────────────────────────────

    def _check_nav2_ready(self) -> bool:
        now = time.monotonic()
        if self._nav2_ready:
            return True
        if (now - self._nav2_check_t) < 3.0:
            return False
        self._nav2_check_t = now
        if self._nav2_client.wait_for_server(timeout_sec=0.1):
            self._nav2_ready = True
            self.get_logger().info('[NAV2_READY] NavigateToPose Action Server available')
        return self._nav2_ready

    def _send_nav2_goal(self, tx: float, ty: float) -> bool:
        """
        Send NavigateToPose goal.

        Nav2 expects goals in 'map' frame.  When thermal navigation uses the
        odom-aligned world frame, convert the local target displacement through
        the live map pose instead of assuming a fixed spawn offset.

        Rate limit: max one goal attempt per 3s to avoid rejection spam.
        Returns True only when Nav2 is actively navigating to this goal.
        """
        if not self._check_nav2_ready():
            return False

        now = time.monotonic()
        if now < self._nav2_direct_until:
            return False

        action = decide_goal_action(
            self._nav2_state,
            self._nav2_current_goal,
            (tx, ty),
            now,
            self._nav2_last_send_t,
            self._nav2_goal_min_interval_s,
        )
        if action == GOAL_KEEP:
            return True
        if action == GOAL_REPLACE:
            self.get_logger().info(
                f'[NAV2_REPLACE] old={self._nav2_current_goal} new=({tx:.1f},{ty:.1f})')
            self._cancel_nav2_goal()
            return False
        if action != GOAL_SEND:
            return False

        if self._pose_source == 'odom' and self._tf_ready:
            dx = tx - self._wx
            dy = ty - self._wy
            yaw_delta = self._map_yaw - self._odom_yaw
            c = math.cos(yaw_delta)
            s = math.sin(yaw_delta)
            goal_map_x = self._map_x + c * dx - s * dy
            goal_map_y = self._map_y + s * dx + c * dy
        else:
            goal_map_x = tx - self._spawn_x
            goal_map_y = ty - self._spawn_y

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        # Gazebo/SLAM transforms in this workspace are stamped on the simulation
        # clock even while use_sim_time defaults to false.  A wall-clock goal
        # stamp makes Nav2 ask TF for an impossible future transform.  Stamp zero
        # requests the latest available transform and keeps the default time
        # policy unchanged.
        goal_msg.pose.header.stamp    = rclpy.time.Time().to_msg()
        goal_msg.pose.pose.position.x = float(goal_map_x)
        goal_msg.pose.pose.position.y = float(goal_map_y)
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.w = 1.0

        self._nav2_state        = NAV2_SENDING
        self._nav2_current_goal = (tx, ty)
        self._nav2_last_send_t  = now
        self._nav2_goal_generation += 1
        generation = self._nav2_goal_generation

        send_future = self._nav2_client.send_goal_async(
            goal_msg, feedback_callback=self._nav2_feedback_cb)
        send_future.add_done_callback(
            lambda future, generation=generation:
            self._nav2_goal_response_cb(future, generation))

        self.get_logger().info(
            f'[NAV2→] Goal world=({tx:.1f},{ty:.1f}) '
            f'map=({goal_map_x:.1f},{goal_map_y:.1f}) state={self._state}')
        return False  # Not active yet; fallback handles movement this tick

    def _nav2_goal_response_cb(self, future, generation=None):
        try:
            goal_handle = future.result()
        except Exception as exc:
            if generation == self._nav2_goal_generation:
                self._nav2_state = NAV2_IDLE
                self._nav2_goal_handle = None
                self._nav2_current_goal = None
            self.get_logger().warn(f'[NAV2_RESPONSE_ERROR] {exc}')
            return
        if generation != self._nav2_goal_generation:
            if goal_handle.accepted:
                goal_handle.cancel_goal_async()
                self.get_logger().info(
                    f'[NAV2_STALE_CANCEL] generation={generation} '
                    f'current={self._nav2_goal_generation}')
            return
        if not goal_handle.accepted:
            # Goal rejected (e.g. map too small, target unreachable)
            self._nav2_state       = NAV2_IDLE
            self._nav2_goal_handle = None
            # Direct fallback will handle movement next tick
            return
        self._nav2_goal_handle = goal_handle
        self._nav2_state       = NAV2_ACTIVE
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda future, generation=generation:
            self._nav2_result_cb(future, generation))
        self.get_logger().info(f'[NAV2✓] Goal accepted, navigating...')

    def _nav2_result_cb(self, future, generation=None):
        if generation != self._nav2_goal_generation:
            return
        result = future.result()
        status_str = 'SUCCEEDED' if result.status == 4 else f'status={result.status}'
        self.get_logger().info(
            f'[NAV2] Navigation ended: {status_str} '
            f'target={self._nav2_current_goal}')
        self._nav2_state       = NAV2_DONE
        self._nav2_goal_handle = None

    def _nav2_feedback_cb(self, feedback_msg):
        pass  # State machine driven by thermal sensors, not Nav2 feedback

    def _cancel_nav2_goal(self):
        """Cancel Nav2 goal before entering FINE mode (direct /cmd_vel)."""
        self._nav2_goal_generation += 1
        if self._nav2_goal_handle is not None and self._nav2_state == NAV2_ACTIVE:
            self._nav2_goal_handle.cancel_goal_async()
            self.get_logger().info('[NAV2✗cancel] Cancelled → switching to direct /cmd_vel')
        self._nav2_state        = NAV2_IDLE
        self._nav2_goal_handle  = None
        self._nav2_current_goal = None
        self._nav2_progress_goal = None
        self._nav2_progress_best_d = float('inf')
        self._nav2_progress_t = 0.0

    def _nav2_progress_stalled(self, tx: float, ty: float, now: float, label: str) -> bool:
        """Return true after cancelling Nav2 if an active goal stops making progress."""
        if self._nav2_state != NAV2_ACTIVE:
            self._nav2_progress_goal = None
            self._nav2_progress_best_d = float('inf')
            self._nav2_progress_t = 0.0
            return False
        dist = math.hypot(tx - self._wx, ty - self._wy)
        goal_key = (label, round(tx, 1), round(ty, 1))
        if self._nav2_progress_goal != goal_key:
            self._nav2_progress_goal = goal_key
            self._nav2_progress_best_d = dist
            self._nav2_progress_t = now
            return False
        if dist <= self._nav2_progress_best_d - self._nav2_progress_min_delta:
            self._nav2_progress_best_d = dist
            self._nav2_progress_t = now
            return False
        if (now - self._nav2_progress_t) < self._nav2_progress_timeout_s:
            return False

        self._nav2_direct_until = now + self._nav2_stall_direct_s
        self.get_logger().warn(
            f'[NAV2_STALL/{label}] no progress for {now-self._nav2_progress_t:.1f}s '
            f'd={dist:.1f}m best={self._nav2_progress_best_d:.1f}m '
            f'→ direct fallback {self._nav2_stall_direct_s:.0f}s')
        self._cancel_nav2_goal()
        return True

    # ────────────────────────────────────────────────────────────────────────
    # Sensor callbacks (unchanged from v30)
    # ────────────────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        """Odom callback: always update yaw; update position only when SLAM TF unavailable.

        _odom_yaw is always kept fresh from odometry so that if SLAM TF temporarily
        fails after being initially available, steering calculations use a current
        heading estimate rather than a stale TF-derived value.  When TF IS available,
        _update_world_pos_from_tf (called every timer tick) overwrites _odom_yaw with
        the more accurate SLAM-derived heading, so there is no regression in the
        normal TF-available path.
        """
        self._odom_x = msg.pose.pose.position.x
        self._odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0*(q.w*q.z+q.x*q.y)
        cosy = 1.0-2.0*(q.y*q.y+q.z*q.z)
        # Always update yaw from odometry (fix: was gated on not self._tf_ready,
        # which left _odom_yaw stale when TF failed after initial lock).
        self._odom_yaw = math.atan2(siny, cosy)
        if not self._tf_ready:
            nx = self._spawn_x + self._odom_x
            ny = self._spawn_y + self._odom_y
            if self._prev_pos is not None:
                self._path_length += math.hypot(nx-self._prev_pos[0], ny-self._prev_pos[1])
            self._prev_pos = (nx, ny)
            self._wx, self._wy = nx, ny

    def _scan_cb(self, msg: LaserScan):
        """Cache only the forward clearance needed by direct-motion fallback."""
        self._front_clearance_m = forward_clearance(
            msg.ranges,
            angle_min=msg.angle_min,
            angle_increment=msg.angle_increment,
            range_min=msg.range_min,
            range_max=msg.range_max,
            half_angle_rad=self._direct_scan_half_angle,
        )
        self._front_scan_t = time.monotonic()

    def _grad_cb(self, msg: GradientArray):
        self._last_ga = msg
        T     = msg.peak_temperature_celsius
        trise = T - self._ambient_est if self._calib_done else 0.0

        if not self._calib_done:
            self._calib_buf.append(T)
            if len(self._calib_buf) >= _N_CALIB:
                arr    = sorted(self._calib_buf)
                cutoff = max(1, int(len(arr)*0.4))
                self._ambient_est = float(np.mean(arr[:cutoff]))
                self._calib_done  = True
                self._T_max_unconf = self._ambient_est
                self.get_logger().info(
                    f'[CALIB] ambient={self._ambient_est:.1f}C '
                    f'TF_ready={self._tf_ready}')
                if self._ambient_est > 30.0:
                    self.get_logger().warn(
                        f'[CALIB WARN] ambient={self._ambient_est:.1f}C > 30C, '
                        f'check if robot is already near heat source')
                current_trise = T - self._ambient_est
                if current_trise > 3.0 and self._thermal_map is None:
                    self._bmap.update(self._wx, self._wy,
                                      current_trise, heat_sigma=self._heat_sigma)
        else:
            if T < self._ambient_est + self._amb_margin:
                self._ambient_est = (self._ambient_ema_alpha*T
                                     + (1.0-self._ambient_ema_alpha)*self._ambient_est)

        if T > self._temp_max_seen:
            self._temp_max_seen = T
        if self._calib_done and trise > _WARM_TEMP_DELTA:
            self._last_warm_pos = (self._wx, self._wy)
        if self._calib_done and self._thermal_map is None:
            self._bmap.update(self._wx, self._wy, trise, heat_sigma=self._heat_sigma)
        if self._state == STATE_CONVERGE:
            if T > self._converge_best_T:
                self._converge_best_T   = T
                self._converge_best_pos = (self._wx, self._wy)
        if self._calib_done:
            away = (not self._found_sources or
                    self._nearest_known_dist() > self._excl_r * 1.5)
            not_conv = self._state not in (STATE_CONVERGE, STATE_SAMPLE,
                                            STATE_AT_PEAK, STATE_DEPARTURE)
            if away and not_conv and T > self._T_max_unconf:
                self._T_max_unconf = T
        if self._state == STATE_SURVEY_PAUSE and self._survey_t_start is not None:
            self._survey_buf.append(T)

    def _map_cb(self, msg: ThermalMap):
        if msg.width == 0 or msg.height == 0:
            return
        try:
            shape = (msg.height, msg.width)
            self._thermal_map = {
                'measurement_type': msg.measurement_type,
                'width': int(msg.width),
                'height': int(msg.height),
                'resolution': float(msg.resolution),
                'origin_x': float(msg.origin_x),
                'origin_y': float(msg.origin_y),
                'temperature_mean': np.asarray(msg.temperature_mean, dtype=np.float32).reshape(shape),
                'temperature_variance': np.asarray(msg.temperature_variance, dtype=np.float32).reshape(shape),
                'confidence': np.asarray(msg.confidence, dtype=np.float32).reshape(shape),
                'visit_count': np.asarray(msg.visit_count, dtype=np.float32).reshape(shape),
                'last_seen_age_s': np.asarray(msg.last_seen_age_s, dtype=np.float32).reshape(shape),
            }
            n_cells = int(msg.width) * int(msg.height)
            if len(msg.view_state) == n_cells:
                self._thermal_map['view_state'] = np.asarray(
                    msg.view_state, dtype=np.uint8).reshape(shape)
            self._thermal_map_t = time.monotonic()
        except ValueError as exc:
            if not hasattr(self, '_map_shape_warned'):
                self._map_shape_warned = True
                self.get_logger().warn(f'[THERMAL_MAP] bad shape: {exc}')

    def _occ_map_cb(self, msg: OccupancyGrid):
        if not fr_visibility.occupancy_grid_has_known_cells(
                msg.data, msg.info.width, msg.info.height):
            if not getattr(self, '_empty_occ_warned', False):
                self._empty_occ_warned = True
                self.get_logger().warn(
                    f'[OCC_MAP_SKIP] unusable size={msg.info.width}x{msg.info.height} '
                    f'data={len(msg.data)} known=0; retaining previous valid map')
            return
        q=msg.info.origin.orientation
        origin_yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        view=fr_visibility.from_flat(msg.data,msg.info.width,msg.info.height,
            msg.info.origin.position.x,msg.info.origin.position.y,msg.info.resolution,
            origin_yaw=origin_yaw)
        if self._pose_source=='odom':
            try:
                tf=self._tf_buffer.lookup_transform('odom',msg.header.frame_id,rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=.01))
                q=tf.transform.rotation;t=tf.transform.translation
                yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
                view=fr_visibility.transform_view(view,t.x+self._spawn_x,t.y+self._spawn_y,yaw)
            except (LookupException,ExtrapolationException,ConnectivityException):
                return
        else:
            view=fr_visibility.transform_view(view,self._spawn_x,self._spawn_y,0.)
        self._occ_view=view
        if not getattr(self, '_occ_map_diag_logged', False):
            self._occ_map_diag_logged = True
            values = np.asarray(msg.data, dtype=np.int16)
            free_count = int(np.count_nonzero(
                (values >= 0) & (values < self._occ_view.occupied_threshold)))
            occupied_count = int(np.count_nonzero(
                values >= self._occ_view.occupied_threshold))
            self.get_logger().info(
                f'[OCC_MAP] size={msg.info.width}x{msg.info.height} '
                f'res={msg.info.resolution:.3f} '
                f'world_origin=({self._occ_view.origin_x:.2f},'
                f'{self._occ_view.origin_y:.2f}) '
                f'free={free_count} occupied={occupied_count} '
                f'unknown={int(values.size-free_count-occupied_count)}')

    def _residual_snapshot(self):
        m = self._thermal_map
        if m is None or 'view_state' not in m or 'temperature_mean' not in m:
            return None
        sources = [(sx, sy, max(0.0, st - self._ambient_est), self._heat_sigma)
                   for sx, sy, st in self._found_sources]
        if self._strategy_mode in ('fast','dual','gp_ucb'):
            sources=[(t['x'],t['y'],t['strength'],t.get('sigma',self._heat_sigma))
                     for t in self._tracker_sources if t['status'] == 'confirmed'
                     and t['probability'] >= .75]
        predicted = fr_residual.predict_field(
            m['width'], m['height'], m['resolution'],
            m['origin_x'], m['origin_y'], self._ambient_est, sources)
        return fr_residual.residual_field(
            m['temperature_mean'], predicted, m['view_state'])

    def _residual_waypoint(self):
        m = self._thermal_map
        resid = self._residual_snapshot()
        if m is None or resid is None:
            return None
        effective_min_d = operational_waypoint_min_distance(
            configured_min_d=self._residual_waypoint_min_d,
            arrival_radius=(self._explore_arrival if self._strategy_mode in ('fast','dual','gp_ucb')
                            else self._frontier_r),
            resolution=m['resolution'],
        )
        if self._survey_wp_max_d <= effective_min_d:
            self.get_logger().warn(
                '[RESIDUAL_TARGET_REJECT/invalid_bounds] '
                f'min={effective_min_d:.2f} max={self._survey_wp_max_d:.2f}')
            return None
        reach_fn = None
        candidate_valid_mask = None
        candidate_count = -1
        annulus_candidate_count = -1
        yy, xx = np.mgrid[0:m['height'], 0:m['width']]
        cwx = m['origin_x']+(xx.astype(np.float32)+.5)*m['resolution']
        cwy = m['origin_y']+(yy.astype(np.float32)+.5)*m['resolution']
        if self._strategy_mode in ('fast','dual','gp_ucb'):
            candidate_valid_mask=self._explore_progress.allowed(cwx,cwy,time.monotonic())
        if self._occ_view is not None:
            free = fr_visibility.known_free_at(self._occ_view, cwx, cwy)
            candidate_valid_mask = free if candidate_valid_mask is None else candidate_valid_mask & free
            candidate_count = int(np.count_nonzero(candidate_valid_mask))
            annulus_dist = np.hypot(cwx - self._wx, cwy - self._wy)
            annulus_candidate_count = int(np.count_nonzero(
                candidate_valid_mask
                & (annulus_dist >= effective_min_d)
                & (annulus_dist <= self._survey_wp_max_d)))
            reach_fn = (lambda x0, y0, x1, y1:
                        fr_visibility.line_reachable_plannable(
                            self._occ_view, x0, y0, x1, y1))
        plan_t0 = time.monotonic()
        if self._strategy_mode=='gp_ucb':
            target=self._gp.select(m,(self._wx,self._wy),effective_min_d,self._survey_wp_max_d,
                reach_fn,candidate_valid_mask,self._ambient_est)
        else:
            target = select_residual_target(
            self._wx, self._wy,
            m['width'], m['height'], m['resolution'],
            m['origin_x'], m['origin_y'],
            resid, m['view_state'], m['last_seen_age_s'],
            known_sources=[(sx, sy) for sx, sy, _ in self._found_sources],
            w_residual=1.6 if self._residual_enabled else 0.,
            information_gain=self._posterior_gain(m),
            w_information=self._posterior_weight,
            heading_yaw=self._odom_yaw,
            w_turn=self._explore_turn_weight if self._strategy_mode in ('fast','dual') else 0.,
            min_d=effective_min_d,
            max_d=self._survey_wp_max_d,
            safe_dist=self._safe_dist(),
            residual_floor=self._residual_planner_min_evidence,
            footprint_radius=self._residual_planner_footprint_radius,
            top_k=self._residual_planner_top_k,
            candidate_valid_mask=candidate_valid_mask,
            line_reachable_fn=reach_fn)
        eval_ms = (time.monotonic() - plan_t0) * 1000.0
        self._last_plan_ms=eval_ms
        self.get_logger().info(
            f'[RESIDUAL_PLAN] result={"target" if target is not None else "none"} '
            f'min_d={effective_min_d:.2f} max_d={self._survey_wp_max_d:.2f} '
            f'known_free={candidate_count} annulus={annulus_candidate_count} '
            f'eval_ms={eval_ms:.1f}')
        if target is None:
            return None
        target_dist = math.hypot(target.x - self._wx, target.y - self._wy)
        if target_dist + 1e-6 < effective_min_d:
            self.get_logger().warn(
                '[RESIDUAL_TARGET_REJECT/too_close] '
                f'd={target_dist:.2f} min={effective_min_d:.2f} '
                f'arrival={self._frontier_r:.2f}')
            return None
        direct_safe = fr_visibility.line_reachable_known_free(
            self._occ_view, self._wx, self._wy, target.x, target.y)
        return StrategyTarget(
            x=target.x,
            y=target.y,
            score=target.score,
            reason=target.reason,
            execution_hint=EXECUTION_NAV2_PREFERRED,
            metadata={'label': 'residual',
                      'reachable': getattr(target, 'metadata_reachable', True),
                      'direct_safe': direct_safe},
        )


    def _belief_cb(self,msg):
        self._slow_msg=msg
        self._slow_t=time.monotonic()

    def _report_runtime(self):
        belief='active' if self._active_belief() is not None else 'fast_only'
        self.get_logger().info(f'[SOFTWARE_HEALTH] strategy={self._strategy_mode} '
            f'belief={belief} state={self._state} plan_ms={self._last_plan_ms:.2f} '
            f'nav={self._nav2_state} world=({self._wx:.2f},{self._wy:.2f})')

    def _active_belief(self):
        msg=self._slow_msg
        if msg is None: return None
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        if not slow_output_usable(self._strategy_mode,msg.mode,msg.health,stamp,
                self.get_clock().now().nanoseconds/1e9,time.monotonic()-self._slow_t,
                self._slow_timeout):
            return None
        return msg

    def _posterior_gain(self,m):
        msg=self._active_belief()
        if msg is None: return None
        yy,xx=np.indices((m['height'],m['width']))
        x=m['origin_x']+(xx+.5)*m['resolution'];y=m['origin_y']+(yy+.5)*m['resolution']
        points=np.stack((x,y),axis=-1)
        gain=np.zeros(x.shape)
        for c in msg.sources:
            covariance=np.array([[c.covariance_xx,c.covariance_xy],[c.covariance_xy,c.covariance_yy]])
            if not np.isfinite(covariance).all(): return None
            gain+=source_information_gain(points,(c.position.x,c.position.y),covariance,
                c.existence_probability,self._residual_planner_footprint_radius,
                self._posterior_pd,self._posterior_pf,self._posterior_variance)
        return gain

    def _surface_timer(self,now):
        if self._surface_nav_goal is not None:
            key,gx,gy,started=self._surface_nav_goal
            arrived=math.hypot(gx-self._wx,gy-self._wy)<.5
            expired=now-started>self._surface_approach_timeout
            # An ended action is not proof of arrival: aborted goals defer too.
            if arrived or expired or self._nav2_state==NAV2_DONE:
                self._cancel_nav2_goal();self._surface_nav_goal=None
                if not arrived:
                    self._surface_deferred[key]=now+self._surface_retry_cooldown
                    self.get_logger().info(f'[SURFACE_APPROACH_DEFERRED] {key} navigation unavailable or stalled')
            else:
                if not self._send_nav2_goal(gx,gy):self._pub.publish(Twist())
                return
        latency=now-self._tracker_sources_t
        sources=[t for t in self._tracker_sources if (t['status']=='confirmed'
                 or t.get('age_s',0)+latency<self._surface_max_age)
                 and t['probability']>.5 and t['id'] not in self._surface_seen_ids
                 and now>=self._surface_deferred.get(t['id'],-float('inf'))]
        # Finish an initial/in-flight sweep. Later full sweeps yield to known
        # actionable sources; coverage resumes when those sources are handled.
        if (self._sensor_model == 'b' and
                (self._sweep_active or self._camera_sweep.origin is None or not sources)):
            sweeping=self._camera_sweep.step((self._wx,self._wy),self._odom_yaw,now)
            if sweeping:
                if not self._sweep_active:
                    self._cancel_nav2_goal()
                    self.get_logger().info('[CAMERA_SWEEP] start')
                self._sweep_active=True
                self._pub.publish(self._make_cmd(0.,self._sweep_speed));return
            if self._sweep_active:
                self.get_logger().info('[CAMERA_SWEEP] complete')
                self._sweep_active=False
                self._surface_wp=None
        if sources:
            distance_to=lambda t:math.hypot(t['x']-self._wx,t['y']-self._wy)
            nearest=min(sources,key=distance_to)
            target=next((t for t in sources if t['id']==self._surface_target_id),nearest)
            # Ignore small estimate/ordering changes, but admit a substantially
            # closer new source instead of locking onto a long unnecessary trip.
            if distance_to(nearest)+self._surface_standoff<distance_to(target):target=nearest
            self._cancel_nav2_goal()
            self._state=STATE_CONVERGE
            dx,dy=target['x']-self._wx,target['y']-self._wy
            distance=math.hypot(dx,dy)
            yaw=math.atan2(dy,dx)
            if self._surface_target_id!=target['id']:
                self._surface_target_id=target['id'];self._surface_hold_start=None
                self._surface_blocked_since=None
                self.get_logger().info(f'[SURFACE_TARGET] {target["id"]}')
            error=math.atan2(math.sin(yaw-self._odom_yaw),math.cos(yaw-self._odom_yaw))
            if distance<=self._surface_standoff+.2 and abs(error)<.25:
                self._pub.publish(Twist())
                if self._surface_hold_start is None: self._surface_hold_start=now
                if now-self._surface_hold_start>=self._surface_hold and target['status']=='confirmed':
                    self._surface_seen_ids.add(target['id'])
                    self._surface_wp=None
                    self.get_logger().info(f'[SURFACE_CONFIRMED] {target["id"]} '
                        f'pos=({target["x"]:.2f},{target["y"]:.2f}) range={distance:.2f}')
                return
            self._surface_hold_start=None
            gx=target['x']-dx/max(distance,1e-6)*self._surface_standoff
            gy=target['y']-dy/max(distance,1e-6)*self._surface_standoff
            lin,ang=self._drive_toward_yaw(yaw,min(self._surface_speed,max(0.,distance-self._surface_standoff)))
            if abs(error)>.5: lin=0.
            guarded,_,_,_=self._direct_guarded_cmd(gx,gy,lin,now,'surface_approach')
            if lin>0. and guarded<=0. and abs(error)<.5:
                # Stop immediately, but do not abandon a source for one scan.
                if self._surface_blocked_since is None:self._surface_blocked_since=now
                if now-self._surface_blocked_since<self._surface_blocked_wait:
                    self._pub.publish(Twist());return
                self._surface_blocked_since=None
                waypoint=surface_approach_waypoint((self._wx,self._wy),(target['x'],target['y']),
                    self._surface_standoff,self._occ_view,self._surface_robot_radius,self._surface_waypoint_step)
                if waypoint is not None:
                    self._surface_nav_goal=(target['id'],*waypoint,now)
                    self.get_logger().info(f'[SURFACE_APPROACH_NAV2] {target["id"]} waypoint={waypoint}')
                else:
                    self._surface_deferred[target['id']]=now+self._surface_retry_cooldown
                    self._surface_wp=None
                    self.get_logger().info(f'[SURFACE_APPROACH_DEFERRED] {target["id"]} no known-free approach step')
                self._pub.publish(Twist());return
            self._surface_blocked_since=None
            lin=min(lin,guarded)
            self._pub.publish(self._make_cmd(lin,ang))
            return
        self._surface_target_id=None;self._surface_hold_start=None
        self._state=STATE_COARSE_SURVEY
        if self._surface_wp is not None and self._explore_progress.stalled((self._wx,self._wy),now):
            self._explore_progress.reject(self._surface_wp,now)
            self.get_logger().info(f'[EXPLORE_RETARGET] stalled waypoint={self._surface_wp}')
            self._surface_wp=None
        if (exploration_goal_due(self._surface_wp,(self._wx,self._wy),now,self._surface_plan_t,
                                 self._explore_arrival,self._explore_timeout)
                and (self._surface_wp is not None or now-self._surface_plan_t>=1.)):
            self._cancel_nav2_goal()
            if (self._surface_wp is not None
                    and math.dist(self._surface_wp,(self._wx,self._wy))>self._explore_arrival):
                self._explore_progress.reject(self._surface_wp,now)
            self._surface_wp=self._coarse_waypoint()
            self._surface_plan_t=now
            self._explore_progress.reset((self._wx,self._wy),now)
        if self._surface_wp is None:
            self._pub.publish(self._make_cmd(0.,self._max_ang*.25));return
        tx,ty=self._surface_wp
        if self._send_nav2_goal(tx,ty): return
        lin,ang=self._drive_toward_yaw(self._yaw_toward(tx,ty),self._surface_speed)
        lin,ang,_,_=self._direct_guarded_cmd(tx,ty,lin,now,'surface_explore')
        self._pub.publish(self._make_cmd(lin,ang))

    def _sources_cb(self, msg: SourceEstimateArray):
        now = time.monotonic()
        sources = []
        for src in msg.sources:
            sources.append({
                'id': src.id,
                'status': src.status,
                'x': float(src.position.x),
                'y': float(src.position.y),
                'strength': float(src.strength),
                'probability': float(src.existence_probability),
                'confidence': float(src.confidence),
                'observations': int(src.observations),
                'age_s':float(src.age_s),
                'covariance_xx':float(src.covariance_xx),'covariance_yy':float(src.covariance_yy),
                'sigma':float(src.sigma),
            })
        self._tracker_sources = sources
        self._tracker_sources_t = now
        if self._strategy_mode in ('fast','dual','gp_ucb'):
            self._found_sources=[(src['x'],src['y'],self._ambient_est+src['strength'])
                                 for src in sources if src['status']=='confirmed']

    # ────────────────────────────────────────────────────────────────────────
    # Adaptive thresholds (unchanged)
    # ────────────────────────────────────────────────────────────────────────

    def _adapt_pk_delta(self) -> float:
        if not self._adaptive_thresh:
            return self._pk_tdelta
        unconf_rise = self._T_max_unconf - self._ambient_est
        if unconf_rise < self._adapt_min_signal:
            return self._pk_tdelta
        return max(self._pk_tdelta, self._adapt_pk_ratio * unconf_rise)

    def _adapt_pre_pk_thresh(self) -> float:
        return self._pre_pk_ratio * self._adapt_pk_delta()

    def _adapt_sample_min(self) -> float:
        if not self._adaptive_thresh:
            return self._sample_min_trise
        return min(self._sample_min_trise, self._adapt_samp_ratio * self._adapt_pk_delta())

    # ────────────────────────────────────────────────────────────────────────
    # Internal utilities (unchanged)
    # ────────────────────────────────────────────────────────────────────────

    def _grad_mag(self):
        if not self._last_ga or not self._last_ga.gradients:
            return 0.0
        mags = sorted([g.magnitude for g in self._last_ga.gradients], reverse=True)
        top_n = max(1, len(mags) * 3 // 10)
        return float(sum(mags[:top_n]) / top_n)

    def _current_temp(self):
        return self._last_ga.peak_temperature_celsius if self._last_ga else self._ambient_est

    def _temp_rise(self):
        return self._current_temp() - self._ambient_est

    def _nearest_known_dist(self):
        if not self._found_sources:
            return float('inf')
        return min(math.hypot(self._wx-sx, self._wy-sy) for sx,sy,_ in self._found_sources)

    def _is_near_known(self):
        return self._nearest_known_dist() < self._rev_r

    def _sources_centroid(self):
        if not self._found_sources:
            return self._spawn_x, self._spawn_y
        cx=sum(s[0] for s in self._found_sources)/len(self._found_sources)
        cy=sum(s[1] for s in self._found_sources)/len(self._found_sources)
        return cx, cy

    def _dist_to_centroid(self):
        cx,cy=self._sources_centroid()
        return math.hypot(self._wx-cx, self._wy-cy)

    def _repulsion_vec(self):
        if not self._found_sources:
            return 0.0, 0.0
        rx=ry=0.0
        for sx,sy,_ in self._found_sources:
            dx=self._wx-sx; dy=self._wy-sy
            dist=max(math.hypot(dx,dy), self._rep_min)
            s=self._rep_k/(dist*dist)
            rx+=s*dx/dist; ry+=s*dy/dist
        mag=math.hypot(rx,ry)
        return (rx/mag, ry/mag) if mag>1e-6 else (0.0,0.0)

    def _known_source_positions(self):
        return [(s[0],s[1]) for s in self._found_sources]

    def _safe_dist(self):
        return self._excl_r + self._fr_safe_buf

    def _strategy_context(self, now: Optional[float] = None) -> SourceSeekContext:
        if now is None:
            now = time.monotonic()
        return SourceSeekContext(
            now=now,
            robot_wx=self._wx,
            robot_wy=self._wy,
            odom_yaw=self._odom_yaw,
            spawn_x=self._spawn_x,
            spawn_y=self._spawn_y,
            search_rounds=self._search_rounds,
            coarse_wp_count=self._coarse_wp_count,
            found_sources=tuple(self._found_sources),
            tracker_sources=tuple(self._tracker_sources),
            tracker_sources_t=self._tracker_sources_t,
            thermal_map=self._thermal_map,
            thermal_map_t=self._thermal_map_t,
            belief_map=self._bmap,
        )

    def _map_available(self):
        return (self._thermal_map is not None
                and (time.monotonic() - self._thermal_map_t) <= self._planner_map_stale_s)

    def _is_near_known_pos(self, x, y, radius=None):
        r = self._rev_r if radius is None else radius
        return any(math.hypot(x-sx, y-sy) < r for sx, sy, _ in self._found_sources)

    def _best_tracker_source_for_sample(self):
        usable = [
            s for s in self._tracker_sources
            if s['status'] in ('candidate', 'confirmed')
            and not self._is_near_known_pos(s['x'], s['y'], radius=self._excl_r)
        ]
        if not usable:
            return None
        usable.sort(
            key=lambda s: (
                0 if math.hypot(s['x']-self._wx, s['y']-self._wy) <= self._candidate_verify_radius else 1,
                -s['probability'],
                math.hypot(s['x']-self._wx, s['y']-self._wy),
            )
        )
        best = usable[0]
        if math.hypot(best['x']-self._wx, best['y']-self._wy) > max(self._candidate_verify_radius, self._excl_r * 2.0):
            return None
        return best

    def _map_region_novelty(self, radius=5.0):
        if not self._map_available():
            return self._bmap.region_novelty(self._wx, self._wy, radius=radius)
        m = self._thermal_map
        h, w = m['height'], m['width']
        yy, xx = np.mgrid[0:h, 0:w]
        wx = m['origin_x'] + (xx.astype(np.float32) + 0.5) * m['resolution']
        wy = m['origin_y'] + (yy.astype(np.float32) + 0.5) * m['resolution']
        dist = np.sqrt((wx - self._wx)**2 + (wy - self._wy)**2)
        mask = dist <= radius
        if not mask.any():
            return 0.0
        visits = m['visit_count'][mask]
        conf = m['confidence'][mask]
        return float((0.6 / (1.0 + visits) + 0.4 * (1.0 - np.clip(conf, 0.0, 1.0))).mean())

    def _novelty_at(self, wx, wy):
        if self._map_available():
            m = self._thermal_map
            ix = int((wx - m['origin_x']) / m['resolution'])
            iy = int((wy - m['origin_y']) / m['resolution'])
            if 0 <= ix < m['width'] and 0 <= iy < m['height']:
                visits = float(m['visit_count'][iy, ix])
                conf = float(m['confidence'][iy, ix])
                return 0.6 / (1.0 + visits) + 0.4 * (1.0 - max(0.0, min(1.0, conf)))
            return 0.0
        ci,cj=self._bmap._ci(wx,wy)
        return 1.0/(1.0+float(self._bmap.visit[ci,cj]))

    def _escape_yaw(self):
        return self._source_seek_selector.select_escape_yaw(
            self._strategy_context(),
            frontier_bias=self._esc_fr_bias,
        )

    def _yaw_toward(self, tx, ty):
        return math.atan2(ty-self._wy, tx-self._wx)

    def _drive_toward_yaw(self, yaw, lin=None):
        err=math.atan2(math.sin(yaw-self._odom_yaw),math.cos(yaw-self._odom_yaw))
        ang=float(np.clip(self._kp_ang*err,-self._max_ang,self._max_ang))
        v=lin if lin is not None else self._reloc_lin
        v=v if abs(err)<math.radians(35) else 0.05
        return v, ang

    def _target_path_known_free(self, tx: float, ty: float) -> bool:
        return fr_visibility.line_reachable_known_free(
            self._occ_view, self._wx, self._wy, tx, ty)

    def _direct_guarded_cmd(
            self, tx: float, ty: float, speed: float,
            now: float, context: str):
        """Return a target-following command with a fail-closed scan guard.

        A known-free map segment and locally guarded motion into unknown space
        remain distinct evidence classes.  Both require a fresh forward scan;
        without it, linear velocity is suppressed while angular alignment may
        continue.
        """
        path_known_free = self._target_path_known_free(tx, ty)
        scan_age = (now - self._front_scan_t
                    if self._front_scan_t > 0.0 else float('inf'))
        mode = decide_direct_motion(
            path_known_free=path_known_free,
            scan_clearance_m=self._front_clearance_m,
            scan_age_s=scan_age,
            stop_distance_m=self._direct_scan_stop_m,
            scan_stale_s=self._direct_scan_stale_s,
        )
        lin, ang = self._drive_toward_yaw(self._yaw_toward(tx, ty), speed)
        if mode == DIRECT_STOP:
            lin = 0.0

        previous = self._direct_guard_mode.get(context)
        last_log = self._direct_guard_log_t.get(context, 0.0)
        if mode != previous or (now - last_log) >= 5.0:
            clearance = ('none' if self._front_clearance_m is None
                         else f'{self._front_clearance_m:.2f}')
            age = 'none' if not math.isfinite(scan_age) else f'{scan_age:.2f}'
            self.get_logger().info(
                f'[DIRECT_GUARD/{context}] mode={mode} '
                f'direct_safe={path_known_free} scan_m={clearance} age_s={age}')
            self._direct_guard_mode[context] = mode
            self._direct_guard_log_t[context] = now
        return lin, ang, mode, path_known_free

    def _update_armed(self, T):
        if not self._peak_armed:
            if T < self._ambient_est + self._rearm_delta:
                self._peak_armed = True
                self.get_logger().info(f'[ARMED] T={T:.1f}C → peak detection re-enabled')

    def _is_peak_near_fov_center(self):
        if not self._last_ga:
            return False
        px = int(self._last_ga.peak_pixel_x)
        py = int(self._last_ga.peak_pixel_y)
        width=self._last_ga.width or int(self.get_parameter('thermal_image_width').value)
        height=self._last_ga.height or int(self.get_parameter('thermal_image_height').value)
        return math.hypot(px-width/2,py-height/2) <= self._sample_center_px_r*width/64.

    def _peak_fov_dist_m(self):
        if not self._last_ga:
            return 99.0
        if self._sensor_model!='a':
            return min((math.hypot(t['x']-self._wx,t['y']-self._wy) for t in self._tracker_sources
                        if t.get('age_s',0)<self._surface_max_age),default=float('inf'))
        width=self._last_ga.width or int(self.get_parameter('thermal_image_width').value)
        height=self._last_ga.height or int(self.get_parameter('thermal_image_height').value)
        dx=(self._last_ga.peak_pixel_x-width/2)*float(self.get_parameter('thermal_fov_x_m').value)/width
        dy=(self._last_ga.peak_pixel_y-height/2)*float(self.get_parameter('thermal_fov_y_m').value)/height
        return math.hypot(dx,dy)

    def _check_peak_sample_v22(self, now):
        if self._state != STATE_SAMPLE or self._sample_t_start is None:
            return False
        if (now - self._sample_t_start) < self._sample_hold:
            return False
        if not self._sample_T_buf:
            return False
        sorted_buf = sorted(self._sample_T_buf)
        cutoff = max(1, len(sorted_buf)*2//10)
        T_mean_top = float(sum(sorted_buf[cutoff:])/len(sorted_buf[cutoff:]))
        trise_mean = T_mean_top - self._ambient_est
        eff_min = self._adapt_sample_min()
        self.get_logger().info(
            f'[SAMPLE_DONE] T_mean80={T_mean_top:.1f}C trise={trise_mean:.1f}C '
            f'thr={eff_min:.1f}C → {"CONFIRM" if trise_mean >= eff_min else "REJECT"}')
        return trise_mean >= eff_min

    def _should_enter_converge(self):
        now = time.monotonic()
        if not self._calib_done or not self._peak_armed:
            return False
        if self._state in (STATE_CONVERGE, STATE_SAMPLE, STATE_AT_PEAK,
                           STATE_RELOCATE, STATE_ESCAPE,
                           STATE_COARSE_SURVEY, STATE_SURVEY_PAUSE,
                           STATE_DEPARTURE):
            return False
        if self._is_near_known():
            return False
        if self._found_sources and self._nearest_known_dist() < self._pc_guard_near_dist:
            if self._best_tracker_source_for_sample() is None:
                if not hasattr(self, '_last_known_residual_log_t') or (now-self._last_known_residual_log_t)>=30.0:
                    self._last_known_residual_log_t = now
                    self.get_logger().info(
                        f'[CONVERGE_KNOWN_RESIDUAL] d_known={self._nearest_known_dist():.1f}m '
                        'without new tracker candidate')
                return False
        if now < self._pc_guard_t and self._nearest_known_dist() < self._pc_guard_near_dist:
            if not hasattr(self, '_last_guard_log_t') or (now-self._last_guard_log_t)>=30.0:
                self._last_guard_log_t = now
                self.get_logger().info(f'[CONVERGE_GUARD] remaining {self._pc_guard_t-now:.0f}s')
            return False
        if self._found_sources and self._last_ga and self._last_ga.gradients:
            mags = [g.magnitude for g in self._last_ga.gradients]
            if mags:
                pg = self._last_ga.gradients[int(np.argmax(mags))]
                gwy = self._odom_yaw + float(pg.direction_rad)
                for sx, sy, _ in self._found_sources:
                    d_src = math.hypot(sx-self._wx, sy-self._wy)
                    if d_src < self._excl_r*3.0:
                        yts = math.atan2(sy-self._wy, sx-self._wx)
                        if abs(math.atan2(math.sin(gwy-yts), math.cos(gwy-yts))) < math.radians(60):
                            return False
        return self._temp_rise() >= self._adapt_pre_pk_thresh()

    def _gradient_cmd(self):
        if not self._last_ga or not self._last_ga.gradients:
            return self._max_lin*0.3, 0.0
        mags=[g.magnitude for g in self._last_ga.gradients]
        peak=self._last_ga.gradients[int(np.argmax(mags))]
        grd_err=math.atan2(math.sin(float(peak.direction_rad)),
                           math.cos(float(peak.direction_rad)))
        if self._found_sources:
            vx,vy=self._repulsion_vec()
            rep_err=math.atan2(math.sin(math.atan2(vy,vx)-self._odom_yaw),
                               math.cos(math.atan2(vy,vx)-self._odom_yaw))
            diff=abs(math.atan2(math.sin(rep_err-grd_err),math.cos(rep_err-grd_err)))
            d_near=self._nearest_known_dist()
            w_rep=0.4*math.exp(-d_near/max(self._excl_r,0.5))
            err=((1.0-w_rep)*grd_err+w_rep*rep_err if diff>math.radians(30) else grd_err)
        else:
            err=grd_err
        raw_ang=float(np.clip(self._kp_ang*err,-self._max_ang,self._max_ang))
        self._ang_smooth=self._alpha_ang*raw_ang+(1.0-self._alpha_ang)*self._ang_smooth
        raw_lin=float(np.clip(self._max_lin*max(0.0,math.cos(err))**2,0.05,self._max_lin))
        self._lin_smooth=self._alpha_lin*raw_lin+(1.0-self._alpha_lin)*self._lin_smooth
        return self._lin_smooth, self._ang_smooth

    def _should_ascent(self, gm):
        trise=self._temp_rise()
        if trise <= -self._neg_bail:
            return False
        if self._state==STATE_ASCENT:
            return gm>=self._min_gmag*_ASCENT_EXIT_GM_FACTOR or trise>=_ASCENT_EXIT_TRISE
        else:
            return gm>=self._min_gmag and trise>=self._min_asc_rise

    def _check_stuck(self, now):
        if (now-self._stuck_t)>=self._stuck_to:
            moved=math.hypot(self._wx-self._stuck_pos[0],self._wy-self._stuck_pos[1])
            self._stuck_t=now; self._stuck_pos=(self._wx,self._wy)
            return moved<self._stuck_thr
        return False

    def _check_thermal_lost(self, now):
        if self._last_warm_pos is None:
            self._thermal_lost_start = None
            return False
        if self._path_length < 3.0:
            self._thermal_lost_start = None
            return False
        if self._peak_cand_t is not None:
            self._thermal_lost_start=None; return False
        if self._temp_rise()<self._lost_t_thr:
            if self._thermal_lost_start is None:
                self._thermal_lost_start=now
            elif (now-self._thermal_lost_start)>=self._lost_timeout:
                if self._last_warm_pos is not None:
                    if math.hypot(self._wx-self._last_warm_pos[0],
                                  self._wy-self._last_warm_pos[1])>=3.0:
                        self._thermal_lost_start=None; return True
                else:
                    self._thermal_lost_start=None; return True
        else:
            self._thermal_lost_start=None
        return False

    def _check_radius_overflow(self):
        return self._dist_to_centroid()>self._max_srch_r

    def _levy_step(self):
        mu=self._levy_mu
        sn=math.gamma(1+mu)*math.sin(math.pi*mu/2)
        sd=math.gamma((1+mu)/2)*mu*(2**((mu-1)/2))
        su=(sn/sd)**(1.0/mu)
        u=random.gauss(0,su); v=abs(random.gauss(0,1))+1e-9
        return float(np.clip(abs(u/(v**(1.0/mu)))*self._levy_scale,
                             self._levy_min,self._levy_max))

    def _refresh_frontier(self, now, force=False):
        if self._strategy_mode == 'levy':
            if force or (now - self._frontier_last_upd) >= self._frontier_upd:
                self._do_levy_jump(now)
            return
        if self._strategy_mode in ('residual', 'fast', 'dual', 'gp_ucb'):
            if not force and (now - self._frontier_last_upd) < self._frontier_upd:
                return
            target = self._residual_waypoint()
            if target is not None:
                self._frontier_last_upd = now
                self._frontier_target = target.xy
                self._frontier_direct_safe = bool(
                    target.metadata.get('direct_safe', False))
                self.get_logger().info(
                    f'[FRONTIER/residual/{target.reason}] '
                    f'→({target.x:.1f},{target.y:.1f}) score={target.score:.3f} '
                    f'reachable={target.metadata.get("reachable", True)} '
                    f'direct_safe={self._frontier_direct_safe}')
                return
        if not force and (now-self._frontier_last_upd)<self._frontier_upd:
            return
        self._frontier_last_upd=now
        if self._pc_rounds_left>0:
            d_near=self._nearest_known_dist() if self._found_sources else 0.0
            min_d=max(self._pc_min_d,d_near+self._excl_r+1.5)
            d_sig=self._pc_dist_sigma; self._pc_rounds_left-=1
            mode_str=f'BOOST min_d={min_d:.1f}m'
        else:
            min_d=3.0; d_sig=8.0; mode_str='NORMAL'
        target = self._source_seek_selector.select_frontier(
            self._strategy_context(now),
            min_d=min_d,
            max_d=16.0,
            dist_sigma=d_sig,
        )
        if target is not None:
            fx, fy = target.xy
            self._frontier_target=(fx,fy)
            self._frontier_direct_safe = self._target_path_known_free(fx, fy)
            self.get_logger().info(
                f'[FRONTIER/{mode_str}/{target.reason}] →({fx:.1f},{fy:.1f}) score={target.score:.3f} '
                f'd={math.hypot(fx-self._wx,fy-self._wy):.1f}m')
        else:
            self._do_levy_jump(now)

    def _do_levy_jump(self, now):
        if self._strategy_mode == 'frontier':
            target = self._source_seek_selector.select_frontier(
                self._strategy_context(now), min_d=2.0, max_d=16.0, dist_sigma=8.0)
            if target is not None:
                fx, fy = target.xy
                self._frontier_target = (fx, fy)
                self._frontier_direct_safe = self._target_path_known_free(fx, fy)
                self._frontier_last_upd = now
                self._search_rounds = 0
                self.get_logger().info(
                    f'[FRONTIER/baseline-fallback] →({fx:.1f},{fy:.1f})')
                return
        step=self._levy_step()
        target = self._source_seek_selector.select_levy_jump(
            self._strategy_context(now),
            step=step,
        )
        lx, ly = target.xy
        direction = target.metadata.get('direction', self._yaw_toward(lx, ly))
        self._frontier_target=(lx,ly); self._frontier_last_upd=now; self._search_rounds=0
        self._frontier_direct_safe = self._target_path_known_free(lx, ly)
        self.get_logger().info(
            f'[LEVY] step={step:.1f}m dir={math.degrees(direction):.0f}deg '
            f'→({lx:.1f},{ly:.1f})')

    def _enter_escape(self, yaw, reason, mode=EMODE_SOURCE_AVOID):
        now=time.monotonic()
        self._cancel_nav2_goal()
        self._state=STATE_ESCAPE; self._state_t=now
        self._move_start=(self._wx,self._wy); self._escape_mode=mode
        self._peak_cand_t=None
        self._locked_yaw=yaw if mode==EMODE_SOURCE_AVOID else None
        if mode==EMODE_SOURCE_AVOID:
            if (now-self._esc_loop_last_t)<self._esc_loop_win: self._esc_loop_count+=1
            else: self._esc_loop_count=1
            self._esc_loop_last_t=now
        tag=(f'yaw={round(math.degrees(yaw))}deg'
             if mode==EMODE_SOURCE_AVOID else 'dynamic→centroid')
        self.get_logger().info(f'[→ESCAPE/{mode}] {reason} | {tag}')

    def _make_cmd(self, lin, ang):
        t=Twist()
        t.linear.x=float(np.clip(lin,-self._max_lin,self._max_lin))
        t.angular.z=float(np.clip(ang,-self._max_ang,self._max_ang))
        return t

    # ────────────────────────────────────────────────────────────────────────
    # DEPARTURE: always direct nav (target may be outside current map)
    # ────────────────────────────────────────────────────────────────────────

    def _compute_departure_wp(self) -> Tuple[float, float]:
        departure_min_d = max(4.5, self._excl_r + self._fr_safe_buf + 1.5)
        target = self._source_seek_selector.select_departure(
            self._strategy_context(),
            min_travel_d=departure_min_d,
        )
        tx, ty = target.xy
        self._departure_direct_safe = self._target_path_known_free(tx, ty)
        target.metadata['direct_safe'] = self._departure_direct_safe
        cx, cy = target.metadata.get('centroid', self._sources_centroid())
        yaw = target.metadata.get('yaw')
        label = target.metadata.get('label', 'selector')
        yaw_text = f' {label}={math.degrees(yaw):.0f}deg' if yaw is not None else f' {label}'
        self.get_logger().info(
            f'[DEPARTURE_WP/{target.reason}] strategy={target.strategy} '
            f'hint={target.execution_hint} centroid=({cx:.1f},{cy:.1f}){yaw_text} '
            f'→({tx:.1f},{ty:.1f}) d_robot={math.hypot(tx-self._wx,ty-self._wy):.1f}m '
            f'direct_safe={self._departure_direct_safe}')
        return (tx, ty)

    def _tracker_source_ready_for_sample(self, est: Optional[Dict], trise: float, elapsed: float) -> bool:
        if est is None:
            return False
        if est['probability'] < self._tracker_verify_prob:
            return False
        if est['observations'] < self._tracker_verify_obs:
            return False
        d_est = math.hypot(est['x'] - self._wx, est['y'] - self._wy)
        if d_est > self._tracker_verify_max_d:
            return False
        if trise < self._adapt_sample_min():
            return False
        return d_est <= self._tracker_verify_arrival_r or elapsed >= self._tracker_verify_min_elapsed_s

    def _enter_sample(self, now: float, T: float, reason: str):
        self._state = STATE_SAMPLE
        self._state_t = now
        self._sample_t_start = now
        self._sample_T_buf = [T]
        self._conv_cold_t = None
        self.get_logger().info(f'[CONVERGE→SAMPLE/{reason}]')
        self._pub.publish(Twist())

    def _start_post_confirm_departure(self, now: float, n_found: int):
        self._move_start = None
        self._locked_yaw = None
        self._peak_cand_t = None
        self._temp_max_seen = self._ambient_est
        self._search_rounds = 0
        self._esc_loop_count = 0
        self._departure_wp = self._compute_departure_wp()
        self._departure_progress_goal = None
        self._departure_progress_best_d = float('inf')
        self._departure_progress_t = now
        if n_found >= self._source_set_expansion_min_sources:
            self._enter_coarse_survey(
                initial_wp=self._departure_wp,
                reason='source_set_expansion')
            return
        self._state = STATE_DEPARTURE
        self._state_t = now
        self.get_logger().info(
            f'[AT_PEAK→DEPARTURE] {n_found} confirmed '
            f'→({self._departure_wp[0]:.1f},{self._departure_wp[1]:.1f})')

    def _exec_departure(self, now: float):
        """
        DEPARTURE: direct navigation toward exit waypoint.
        Nav2 NOT used here because target is typically outside current SLAM map.
        """
        if self._departure_wp is None:
            self._enter_coarse_survey(reason='departure_wp_none'); return

        tx, ty = self._departure_wp
        dist   = math.hypot(tx-self._wx, ty-self._wy)
        elapsed = now - self._state_t
        goal_key = (round(tx, 1), round(ty, 1))
        if self._departure_progress_goal != goal_key:
            self._departure_progress_goal = goal_key
            self._departure_progress_best_d = dist
            self._departure_progress_t = now
        elif dist <= self._departure_progress_best_d - self._departure_progress_min_delta:
            self._departure_progress_best_d = dist
            self._departure_progress_t = now

        if dist <= self._frontier_r:
            self.get_logger().info(f'[DEPARTURE→COARSE] arrived d={dist:.2f}m')
            self._enter_coarse_survey(reason='departure_arrived'); return
        if (now - self._departure_progress_t) >= self._departure_progress_timeout_s:
            self.get_logger().warn(
                f'[DEPARTURE→COARSE] stalled {now-self._departure_progress_t:.1f}s '
                f'd={dist:.1f}m best={self._departure_progress_best_d:.1f}m')
            self._enter_coarse_survey(reason='departure_stalled'); return
        if elapsed >= self._departure_timeout:
            self.get_logger().warn(f'[DEPARTURE→COARSE] timeout {elapsed:.0f}s')
            self._enter_coarse_survey(reason='departure_timeout'); return

        lin, ang, _, self._departure_direct_safe = self._direct_guarded_cmd(
            tx, ty, self._departure_speed, now, 'departure')
        self._pub.publish(self._make_cmd(lin, ang))

        if not hasattr(self,'_dep_log_t') or (now-self._dep_log_t)>=10.0:
            self._dep_log_t=now
            self.get_logger().info(
                f'[DEPARTURE] →({tx:.1f},{ty:.1f}) d={dist:.1f}m '
                f'elapsed={elapsed:.0f}s (direct nav, not using Nav2)')

    # ────────────────────────────────────────────────────────────────────────
    # [FIX-2] COARSE_SURVEY: Nav2 + guaranteed direct nav fallback
    # ────────────────────────────────────────────────────────────────────────

    def _coarse_waypoint(self):
        if self._strategy_mode == 'frontier':
            target = self._source_seek_selector.select_frontier(
                self._strategy_context(),
                min_d=self._survey_wp_min_d,
                max_d=self._survey_wp_max_d,
                dist_sigma=8.0,
            )
        elif self._strategy_mode == 'levy':
            target = self._source_seek_selector.select_levy_jump(
                self._strategy_context(),
                step=self._levy_step(),
            )
        elif self._strategy_mode in ('residual', 'fast', 'dual', 'gp_ucb'):
            target = self._residual_waypoint()
            # During mapper/SLAM startup a missing feasible target is not
            # evidence for a long random excursion. Wait/scan and retry.
            if target is None and self._strategy_mode == 'residual':
                target = self._source_seek_selector.select_coarse_waypoint(
                    self._strategy_context(),
                    min_d=self._survey_wp_min_d,
                    max_d=self._survey_wp_max_d,
                )
        else:
            target = self._source_seek_selector.select_coarse_waypoint(
                self._strategy_context(),
                min_d=self._survey_wp_min_d,
                max_d=self._survey_wp_max_d,
            )
        if target is None:
            self._coarse_wp_execution_hint = 'nav2_preferred'
            self._coarse_wp_direct_safe = False
            return None
        self._coarse_wp_execution_hint = target.execution_hint
        self._coarse_wp_direct_safe = bool(target.metadata.get(
            'direct_safe', self._target_path_known_free(target.x, target.y)))
        cx, cy = target.metadata.get('centroid', self._sources_centroid())
        yaw = target.metadata.get('yaw')
        label = target.metadata.get('label', 'selector')
        yaw_text = f' {label}={math.degrees(yaw):.0f}deg' if yaw is not None else f' {label}'
        self.get_logger().info(
            f'[COARSE_WP/{target.reason}] strategy={target.strategy} '
            f'hint={target.execution_hint} centroid=({cx:.1f},{cy:.1f}){yaw_text} '
            f'→({target.x:.1f},{target.y:.1f}) '
            f'direct_safe={self._coarse_wp_direct_safe}')
        return target.xy

    def _enter_coarse_survey(self, initial_wp=None, reason='unknown'):
        now=time.monotonic()
        self._state=STATE_COARSE_SURVEY; self._state_t=now
        self._survey_buf=[]; self._survey_t_start=None
        if initial_wp is not None:
            self._coarse_wp=initial_wp
            self._coarse_wp_direct_safe = self._target_path_known_free(*initial_wp)
            self._coarse_wp_execution_hint = (
                EXECUTION_DIRECT_FIRST if reason == 'source_set_expansion' else 'nav2_preferred'
            )
        else:
            wp=self._coarse_waypoint()
            self._coarse_wp=wp or (self._wx+5, self._wy)
            if wp is None:
                self._coarse_wp_direct_safe = self._target_path_known_free(
                    *self._coarse_wp)
        self._coarse_wp_t=now; self._coarse_wp_count=0
        self._last_survey_pause_t=now; self._frontier_cold_start=None
        self.get_logger().info(
            f'[→COARSE_SURVEY] reason={reason} '
            f'wp=({self._coarse_wp[0]:.1f},{self._coarse_wp[1]:.1f}) '
            f'd={math.hypot(self._coarse_wp[0]-self._wx,self._coarse_wp[1]-self._wy):.1f}m')
        direct_first_s = 0.0
        if reason == 'source_set_expansion' or self._coarse_wp_execution_hint == EXECUTION_DIRECT_FIRST:
            direct_first_s = self._source_set_direct_first_s
        elif reason.startswith('departure_') and self._found_sources:
            direct_first_s = self._post_confirm_direct_first_s
        if direct_first_s > 0.0:
            self._nav2_direct_until = now + direct_first_s
            self._last_survey_pause_t = now + direct_first_s
            self.get_logger().info(
                f'[COARSE_DIRECT_FIRST] reason={reason} {direct_first_s:.0f}s before Nav2 retry and survey pause')
        # Attempt Nav2 goal; guarded fallback can move if Nav2 fails.
        self._send_nav2_goal(self._coarse_wp[0], self._coarse_wp[1])

    def _exec_coarse_survey(self, now: float):
        """
        COARSE_SURVEY: systematic area coverage for heat source search.

        FIX-2: Added scan-guarded direct navigation fallback.
        When Nav2 is unavailable or rejects the goal, direct /cmd_vel is
        permitted only while fresh LaserScan evidence clears the forward path.

        Priority order:
          P1: Real-time thermal signal > threshold → enter FINE mode (ASCENT)
          P2: Periodic survey pause (stop+sense)
          P3: Waypoint management (arrival/timeout → new waypoint)
          P4: Nav2 goal maintenance (try to keep Nav2 active)
          P5: [FIX-2] Guarded direct fallback when Nav2 is not active
        """
        trise = self._temp_rise()

        if self._found_sources and self._dist_to_centroid() > self._max_srch_r * 1.15:
            cx, cy = self._sources_centroid()
            return_yaw = self._yaw_toward(cx, cy)
            overflow_d = self._dist_to_centroid()
            self._cancel_nav2_goal()
            self._enter_escape(
                return_yaw,
                f'coarse_radius_overflow d={overflow_d:.1f}m',
                EMODE_CENTROID_RET)
            self._pub.publish(self._make_cmd(*self._drive_toward_yaw(return_yaw, 0.22)))
            return

        # P1: Real-time thermal signal → switch to FINE gradient navigation
        near_known_residual = (
            self._found_sources
            and self._nearest_known_dist() < self._pc_guard_near_dist
            and self._best_tracker_source_for_sample() is None
        )
        if (self._calib_done and trise >= self._coarse_transit_thr
                and not self._is_near_known()
                and not near_known_residual):
            self._cancel_nav2_goal()
            self._state = STATE_FRONTIER_NAV; self._state_t = now
            self._search_rounds = 0; self._frontier_cold_start = None
            self._frontier_target = None; self._refresh_frontier(now, force=True)
            self.get_logger().info(
                f'[COARSE→FINE] trise={trise:.2f}C >= {self._coarse_transit_thr}C')
            return

        # P2: Periodic survey pause (stop and sense)
        if (now - self._last_survey_pause_t) >= self._survey_pause_ivl:
            self._cancel_nav2_goal()
            self._state = STATE_SURVEY_PAUSE; self._state_t = now
            self._survey_t_start = now; self._survey_buf = [self._current_temp()]
            self._pub.publish(Twist())
            self.get_logger().info(
                f'[COARSE→SURVEY_PAUSE] T={self._current_temp():.1f}C trise={trise:.2f}C')
            return

        # P3: Waypoint management
        tx, ty = self._coarse_wp
        dist_to_wp = math.hypot(tx - self._wx, ty - self._wy)
        elapsed_wp = now - self._coarse_wp_t

        if dist_to_wp <= self._frontier_r or elapsed_wp >= self._survey_wp_timeout:
            label = 'arrival' if dist_to_wp <= self._frontier_r else 'timeout'
            self._coarse_wp_count += 1
            wp = self._coarse_waypoint()
            if wp is not None:
                self._coarse_wp = wp; self._coarse_wp_t = now
                self.get_logger().info(
                    f'[COARSE_WP#{self._coarse_wp_count}] {label} '
                    f'→({wp[0]:.1f},{wp[1]:.1f})')
            else:
                # Lévy-style random step when no frontier found
                step = random.uniform(self._survey_wp_min_d, self._survey_wp_max_d)
                ang_rnd = random.uniform(-math.pi, math.pi)
                self._coarse_wp = (self._wx + step*math.cos(ang_rnd),
                                   self._wy + step*math.sin(ang_rnd))
                self._coarse_wp_direct_safe = self._target_path_known_free(
                    *self._coarse_wp)
                self._coarse_wp_t = now
            tx, ty = self._coarse_wp
            # Try Nav2; guarded fallback below handles unavailable Nav2.
            self._send_nav2_goal(tx, ty)

        elif self._nav2_state in (NAV2_IDLE, NAV2_DONE):
            # Nav2 finished/failed: retry
            self._send_nav2_goal(tx, ty)

        # ★ FIX-2: guarded direct navigation fallback ★
        # This evaluates every tick when Nav2 is not actively navigating;
        # linear velocity fails closed when scan evidence is absent or blocked.
        tx2, ty2 = self._coarse_wp
        dist2 = math.hypot(tx2 - self._wx, ty2 - self._wy)
        self._nav2_progress_stalled(tx2, ty2, now, 'coarse')

        if self._nav2_state != NAV2_ACTIVE:
            if dist2 > self._frontier_r:
                lin, ang, _, self._coarse_wp_direct_safe = self._direct_guarded_cmd(
                    tx2, ty2, self._frontier_lin, now, 'coarse')
                self._pub.publish(self._make_cmd(lin, ang))
            else:
                # At waypoint, slow rotation while computing next target
                self._pub.publish(self._make_cmd(0.0, self._max_ang * 0.2))

        # Periodic log
        if not hasattr(self, '_coarse_log_t') or (now - self._coarse_log_t) >= 10.0:
            self._coarse_log_t = now
            self.get_logger().info(
                f'[COARSE_SURVEY] wp=({tx2:.1f},{ty2:.1f}) d={dist2:.1f}m '
                f'T={self._current_temp():.1f}C trise={trise:.2f}C '
                f'nav2={self._nav2_state} wp#{self._coarse_wp_count}')

    def _exec_survey_pause(self, now: float):
        """Stop and sense: evaluate if thermal signal warrants FINE navigation."""
        self._pub.publish(Twist())
        elapsed_pause = (now - self._survey_t_start) if self._survey_t_start else 0.0
        if elapsed_pause < self._survey_sense_s:
            return
        if self._survey_buf:
            sorted_buf = sorted(self._survey_buf)
            trim_n = max(1, len(sorted_buf)//10)
            trimmed = sorted_buf[trim_n:max(trim_n+1, len(sorted_buf)-trim_n)]
            avg_T = float(np.mean(trimmed)) if trimmed else float(np.mean(sorted_buf))
            avg_trise = avg_T - self._ambient_est
            max_trise = sorted_buf[-1] - self._ambient_est
        else:
            avg_trise = max_trise = 0.0
        detected = (avg_trise >= self._survey_det_thresh
                    or max_trise >= 2.0 * self._survey_det_thresh)
        self.get_logger().info(
            f'[SURVEY_PAUSE_DONE] {elapsed_pause:.1f}s '
            f'avg_trise={avg_trise:.2f}C max_trise={max_trise:.2f}C '
            f'→ {"FINE" if detected else "COARSE"}')
        if detected and self._calib_done and self._peak_armed and not self._is_near_known():
            self._state = STATE_FRONTIER_NAV; self._state_t = now
            self._search_rounds = 0; self._frontier_cold_start = None
            self._frontier_target = None; self._refresh_frontier(now, force=True)
        else:
            self._state = STATE_COARSE_SURVEY; self._state_t = now
            self._last_survey_pause_t = now
            self._survey_buf = []; self._survey_t_start = None
            # Attempt Nav2 for new waypoint; FIX-2 fallback handles movement
            if self._coarse_wp is not None:
                self._send_nav2_goal(self._coarse_wp[0], self._coarse_wp[1])

    # ────────────────────────────────────────────────────────────────────────
    # CONVERGE: fine-grained gradient following (unchanged from v30)
    # ────────────────────────────────────────────────────────────────────────

    def _enter_converge(self):
        now=time.monotonic()
        self._cancel_nav2_goal()
        self._state=STATE_CONVERGE; self._state_t=now
        self._converge_best_T=self._current_temp()
        self._converge_best_pos=(self._wx,self._wy)
        self._conv_cold_t=None; self._conv_returning=False
        self._frontier_cold_start=None
        if self._conv_sticky_count==0:
            self._conv_best_global_T=self._converge_best_T
            self._conv_best_global_pos=(self._wx,self._wy)
        self.get_logger().info(
            f'[→CONVERGE] T={self._converge_best_T:.1f}C '
            f'trise={self._temp_rise():.1f}C@({self._wx:.1f},{self._wy:.1f}) '
            f'sticky={self._conv_sticky_count}/{self._conv_sticky_max}')

    def _exec_converge(self, now):
        T=self._current_temp(); trise=self._temp_rise(); elapsed=now-self._state_t
        if T>self._converge_best_T:
            self._converge_best_T=T; self._converge_best_pos=(self._wx,self._wy)
        if elapsed>self._conv_max_s:
            sample_trigger=self._adapt_pk_delta()*self._conv_sample_ratio
            if self._converge_best_T>self._conv_best_global_T:
                self._conv_best_global_T=self._converge_best_T
                self._conv_best_global_pos=self._converge_best_pos
            bon_thresh=self._adapt_pk_delta()*self._conv_bon_ratio
            if (self._conv_sticky_count>=self._conv_sticky_max
                    and self._conv_best_global_T-self._ambient_est>=bon_thresh
                    and not self._is_near_known()):
                self._state=STATE_SAMPLE; self._state_t=now
                self._sample_t_start=now; self._sample_T_buf=[self._current_temp()]
                self._conv_cold_t=None; self._conv_sticky_count=0
                self._pub.publish(Twist()); return
            if (self._conv_sticky_count<self._conv_sticky_max
                    and self._conv_best_global_T-self._ambient_est>=sample_trigger):
                self._conv_sticky_count+=1; self._conv_returning=True
                self._converge_best_T=0.0; self._converge_best_pos=None
                self._conv_cold_t=None; self._state_t=now
                self._pub.publish(Twist()); return
            self._state=STATE_FRONTIER_NAV; self._state_t=now
            self._conv_cold_t=None
            self._conv_sticky_count=0; self._conv_returning=False
            self._refresh_frontier(now,force=True); self._pub.publish(Twist()); return
        if trise<self._adapt_pre_pk_thresh()*0.3:
            if self._conv_cold_t is None: self._conv_cold_t=now
            elif (now-self._conv_cold_t)>=self._conv_cold_exit_s:
                if self._converge_best_pos is not None:
                    bx,by=self._converge_best_pos
                    if math.hypot(bx-self._wx,by-self._wy)>0.5:
                        lin,ang=self._drive_toward_yaw(self._yaw_toward(bx,by),0.12)
                        self._pub.publish(self._make_cmd(lin,ang)); return
                self._state=STATE_FRONTIER_NAV; self._state_t=now
                self._conv_cold_t=None
                self._refresh_frontier(now,force=True); self._pub.publish(Twist()); return
        else:
            self._conv_cold_t=None
        sample_trigger=self._adapt_pk_delta()*self._conv_sample_ratio
        near_center=self._is_peak_near_fov_center(); fov_dist_m=self._peak_fov_dist_m()
        tracker_est = self._best_tracker_source_for_sample()
        tracker_dist = (math.hypot(tracker_est['x']-self._wx, tracker_est['y']-self._wy)
                        if tracker_est is not None else float('inf'))
        if trise>=sample_trigger and not self._is_near_known():
            if self._tracker_source_ready_for_sample(tracker_est, trise, elapsed):
                self._enter_sample(
                    now, T,
                    f'tracker {tracker_est["id"]} p={tracker_est["probability"]:.2f} '
                    f'obs={tracker_est["observations"]} d={tracker_dist:.1f}m')
                return
            if near_center:
                self._enter_sample(now, T, f'fov_center dist={fov_dist_m:.2f}m')
                return
            else:
                if not hasattr(self,'_conv_approach_log_t') or (now-self._conv_approach_log_t)>=2.5:
                    self._conv_approach_log_t=now
                    self.get_logger().info(
                        f'[CONVERGE_APPROACH] trise={trise:.1f}C dist={fov_dist_m:.2f}m')
        if self._conv_returning and self._conv_best_global_pos is not None:
            bx,by=self._conv_best_global_pos
            if math.hypot(bx-self._wx,by-self._wy)>0.4:
                lin,ang=self._drive_toward_yaw(self._yaw_toward(bx,by),self._conv_sticky_ret_v)
                self._pub.publish(self._make_cmd(lin,ang)); return
            else:
                self._conv_returning=False
        if self._last_ga and self._last_ga.gradients:
            if (tracker_est is not None
                    and tracker_est['probability'] >= self._tracker_verify_prob
                    and tracker_dist <= self._tracker_verify_max_d
                    and not self._is_near_known_pos(tracker_est['x'], tracker_est['y'],
                                                    radius=self._excl_r)):
                yaw = self._yaw_toward(tracker_est['x'], tracker_est['y'])
                lin, ang = self._drive_toward_yaw(yaw, self._tracker_approach_lin)
                self._pub.publish(self._make_cmd(lin, ang))
                if T>self._conv_best_global_T:
                    self._conv_best_global_T=T; self._conv_best_global_pos=(self._wx,self._wy)
                if not hasattr(self,'_conv_tracker_log_t') or (now-self._conv_tracker_log_t)>=3.0:
                    self._conv_tracker_log_t=now
                    self.get_logger().info(
                        f'[CONVERGE_TRACKER] {tracker_est["id"]} '
                        f'p={tracker_est["probability"]:.2f} obs={tracker_est["observations"]} '
                        f'd={tracker_dist:.1f}m trise={trise:.1f}C')
                return
            lin,ang=self._gradient_cmd()
            speed_factor=max(0.3,1.0-trise/(self._adapt_pk_delta()*0.8))
            v_lin=max(0.03,min(self._conv_lin*speed_factor,self._conv_lin))
            self._pub.publish(self._make_cmd(v_lin,ang))
            if T>self._conv_best_global_T:
                self._conv_best_global_T=T; self._conv_best_global_pos=(self._wx,self._wy)
            if not hasattr(self,'_conv_px_log_t') or (now-self._conv_px_log_t)>=3.0:
                self._conv_px_log_t=now
                self.get_logger().info(
                    f'[CONVERGE] t_in={elapsed:.0f}s T={T:.1f}C trise={trise:.1f}C '
                    f'trigger>={sample_trigger:.1f}C dist={fov_dist_m:.2f}m '
                    f'{"✓NEAR" if near_center else "→APPROACH"}')
        else:
            self._pub.publish(self._make_cmd(0.0, self._max_ang*0.25))

    # ────────────────────────────────────────────────────────────────────────
    # SAMPLE: stop-and-confirm (unchanged from v30)
    # ────────────────────────────────────────────────────────────────────────

    def _exec_sample(self, now):
        T=self._current_temp(); self._sample_T_buf.append(T); self._pub.publish(Twist())
        if self._check_peak_sample_v22(now):
            est = self._best_tracker_source_for_sample()
            if est is not None:
                sx, sy = est['x'], est['y']
            else:
                sx = sy = None
            if est is not None and not self._is_near_known_pos(sx, sy, radius=self._excl_r):
                self._found_sources.append((sx,sy,T))
                self._esc_loop_count=0
                self._bmap.mark_excluded(sx,sy,self._excl_r)
                self._bmap.suppress_confirmed_source(sx,sy,self._excl_r+1.0)
                self._peak_armed=False; self._state=STATE_AT_PEAK; self._state_t=now
                kstr=', '.join(f'({s[0]:.1f},{s[1]:.1f})' for s in self._found_sources)
                elapsed=now-self._t0
                self.get_logger().info(
                    f'★ [SOURCE #{len(self._found_sources)}] '
                    f'est={est["id"]} pos=({sx:.2f},{sy:.2f}) '
                    f'p={est["probability"]:.2f} obs={est["observations"]} '
                    f'T={T:.1f}C robot=({self._wx:.2f},{self._wy:.2f}) '
                    f'path={self._path_length:.2f}m t={elapsed:.1f}s')
                self.get_logger().info(f'  Known sources: [{kstr}]')
                self._conv_sticky_count=0; self._conv_best_global_T=0.0
                self._conv_best_global_pos=None; self._conv_returning=False
                self._source_seek_selector.reset_source_sweeps()
                self._T_max_unconf=self._ambient_est
                self._pc_guard_t=now+self._pc_cooldown_s
                self.get_logger().info(
                    f'[CONVERGE_GUARD_SET] block CONVERGE for {self._pc_cooldown_s:.0f}s')
                self._pc_rounds_left=self._pc_rounds
            else:
                if est is None:
                    self.get_logger().warn(
                        '[SAMPLE→FRONTIER] thermal peak passed but no tracker source estimate '
                        'is close enough to confirm')
                self._state=STATE_FRONTIER_NAV; self._state_t=now
                self._refresh_frontier(now,force=True)
        elif self._sample_t_start and (now-self._sample_t_start)>=self._sample_hold:
            eff_min=self._adapt_sample_min()
            # Reproduce the same top-80% mean used inside _check_peak_sample_v22
            # so the log reflects the actual metric that caused rejection.
            if self._sample_T_buf:
                sorted_buf=sorted(self._sample_T_buf)
                cutoff=max(1, len(sorted_buf)*2//10)
                trise_mean=float(sum(sorted_buf[cutoff:])/len(sorted_buf[cutoff:]))-self._ambient_est
            else:
                trise_mean=0.0
            self.get_logger().info(
                f'[SAMPLE→FRONTIER] trise_mean80={trise_mean:.1f}C < thr={eff_min:.1f}C')
            self._state=STATE_FRONTIER_NAV; self._state_t=now
            self._sample_t_start=None; self._sample_T_buf=[]
            self._refresh_frontier(now,force=True)

    # ────────────────────────────────────────────────────────────────────────
    # Main control timer callback
    # ────────────────────────────────────────────────────────────────────────

    def _timer_cb(self):
        now=time.monotonic(); elapsed=now-self._t0

        # FIX-1: Update world position from SLAM TF every tick
        tf_ok = self._update_world_pos_from_tf()
        if tf_ok:
            if self._prev_pos is not None:
                self._path_length += math.hypot(
                    self._wx-self._prev_pos[0], self._wy-self._prev_pos[1])
            self._prev_pos = (self._wx, self._wy)

        if self._sensor_model!='a' or self._strategy_mode in ('fast','dual','gp_ucb'):
            self._surface_timer(now)
            return
        T=self._current_temp(); gm=self._grad_mag()
        n_found=len(self._found_sources); trise=self._temp_rise()
        self._update_armed(T)

        # ─ SAMPLE (highest priority) ──────────────────────────────────────
        if self._state == STATE_SAMPLE:
            self._exec_sample(now); return

        # ─ CONVERGE ───────────────────────────────────────────────────────
        if self._state == STATE_CONVERGE:
            self._exec_converge(now); return

        # ─ AT_PEAK ────────────────────────────────────────────────────────
        if self._state == STATE_AT_PEAK:
            self._pub.publish(Twist())
            if now - self._state_t >= self._peak_hold:
                if self._post_confirm_immediate_departure and self._found_sources:
                    self._start_post_confirm_departure(now, n_found)
                    return
                self._state = STATE_RELOCATE; self._state_t = now
                self._move_start = (self._wx, self._wy)
                self._locked_yaw = self._escape_yaw()
                self._peak_cand_t = None; self._temp_max_seen = self._ambient_est
                self._search_rounds = 0
                self._esc_loop_count = 0
                self.get_logger().info(
                    f'[→RELOCATE] yaw={math.degrees(self._locked_yaw):.0f}deg '
                    f'{n_found} confirmed')
            return

        # ─ SURVEY_PAUSE ───────────────────────────────────────────────────
        if self._state == STATE_SURVEY_PAUSE:
            self._exec_survey_pause(now); return

        # ─ COARSE_SURVEY (Nav2 + direct fallback) ────────────────────────
        if self._state == STATE_COARSE_SURVEY:
            self._exec_coarse_survey(now); return

        # ─ DEPARTURE (direct nav only) ────────────────────────────────────
        if self._state == STATE_DEPARTURE:
            self._exec_departure(now); return

        # ─ RELOCATE ───────────────────────────────────────────────────────
        if self._state == STATE_RELOCATE:
            if self._move_start is not None:
                moved=math.hypot(self._wx-self._move_start[0],self._wy-self._move_start[1])
                if moved >= self._reloc_dist:
                    self._departure_wp = self._compute_departure_wp()
                    if n_found >= self._source_set_expansion_min_sources:
                        self._enter_coarse_survey(
                            initial_wp=self._departure_wp,
                            reason='source_set_expansion')
                        return
                    self._state = STATE_DEPARTURE; self._state_t = now
                    self.get_logger().info(
                        f'[RELOCATE→DEPARTURE] moved={moved:.1f}m '
                        f'→({self._departure_wp[0]:.1f},{self._departure_wp[1]:.1f})')
                    return
            yaw=self._locked_yaw if self._locked_yaw is not None else self._escape_yaw()
            lin,ang=self._drive_toward_yaw(yaw,self._reloc_lin)
            self._pub.publish(self._make_cmd(lin,ang)); return

        # ─ ESCAPE ────────────────────────────────────────────────────────
        if self._state == STATE_ESCAPE:
            if self._escape_mode == EMODE_CENTROID_RET:
                cx,cy=self._sources_centroid()
                dyn_yaw=self._yaw_toward(cx,cy)
                lin,ang=self._drive_toward_yaw(dyn_yaw,0.22)
                self._pub.publish(self._make_cmd(lin,ang))
                if self._dist_to_centroid() <= self._centroid_arr:
                    self._state=STATE_FRONTIER_NAV; self._state_t=now
                    self._search_rounds=0; self._locked_yaw=None
                    self._move_start=None; self._escape_mode=EMODE_SOURCE_AVOID
                    self._thermal_lost_start=None
                    self._refresh_frontier(now,force=True)
            else:
                if self._esc_loop_count >= self._esc_loop_max:
                    self._esc_loop_count=0; self._do_levy_jump(now)
                    self._state=STATE_FRONTIER_NAV; self._state_t=now
                    self._locked_yaw=None; self._move_start=None
                    self._escape_mode=EMODE_SOURCE_AVOID
                    self._pub.publish(Twist()); return
                yaw=self._locked_yaw if self._locked_yaw is not None else self._escape_yaw()
                lin,ang=self._drive_toward_yaw(yaw,0.22)
                self._pub.publish(self._make_cmd(lin,ang))
                d_src=self._nearest_known_dist()
                moved=(math.hypot(self._wx-self._move_start[0],self._wy-self._move_start[1])
                       if self._move_start else 0.0)
                if d_src>=self._excl_r*_ESCAPE_EXIT_FACTOR and moved>=self._min_esc_dist:
                    self._state=STATE_FRONTIER_NAV; self._state_t=now
                    self._search_rounds=0; self._locked_yaw=None
                    self._move_start=None; self._escape_mode=EMODE_SOURCE_AVOID
                    self._refresh_frontier(now,force=True)
            return

        # ─ Active state safety checks (ASCENT/FRONTIER_NAV) ───────────────
        active = (STATE_ASCENT, STATE_FRONTIER_NAV)

        if self._state in active and self._peak_cand_t is None:
            if self._check_stuck(now):
                yaw=random.uniform(-math.pi,math.pi)
                self._enter_escape(yaw,'STUCK',EMODE_SOURCE_AVOID)
                self._pub.publish(self._make_cmd(*self._drive_toward_yaw(yaw,0.22))); return

        if (self._state in active and self._found_sources
                and self._nearest_known_dist() < self._excl_r
                and self._peak_cand_t is None):
            yaw=self._escape_yaw()
            self._enter_escape(yaw,f'entered exclusion zone dist={self._nearest_known_dist():.1f}m')
            self._pub.publish(self._make_cmd(*self._drive_toward_yaw(yaw,0.22))); return

        if self._state in active:
            if self._check_thermal_lost(now):
                cx,cy=self._sources_centroid(); yaw=self._yaw_toward(cx,cy)
                self._enter_escape(yaw,'thermal_lost',EMODE_CENTROID_RET)
                self._pub.publish(self._make_cmd(*self._drive_toward_yaw(yaw,0.22))); return

        if self._state in active and self._peak_cand_t is None:
            if self._check_radius_overflow():
                cx,cy=self._sources_centroid(); yaw=self._yaw_toward(cx,cy)
                self._enter_escape(yaw,'RADIUS_OVERFLOW',EMODE_CENTROID_RET)
                self._pub.publish(self._make_cmd(*self._drive_toward_yaw(yaw,0.22))); return

        # ─ Cold field detection: FRONTIER_NAV → COARSE_SURVEY ─────────────
        if self._state == STATE_FRONTIER_NAV and self._peak_cand_t is None:
            if trise < self._fine_entry_thresh:
                if self._frontier_cold_start is None:
                    self._frontier_cold_start = now
                elif (now - self._frontier_cold_start) >= self._frontier_cold_to:
                    self.get_logger().info(
                        f'[FRONTIER→COARSE] {now-self._frontier_cold_start:.0f}s cold '
                        f'trise={trise:.2f}C → COARSE_SURVEY')
                    self._frontier_cold_start = None
                    self._enter_coarse_survey(reason='frontier_cold'); return
            else:
                self._frontier_cold_start = None

        # ─ CONVERGE detection ─────────────────────────────────────────────
        if self._state in active and self._should_enter_converge():
            self._enter_converge()
            self._pub.publish(self._make_cmd(self._conv_lin, 0.0)); return

        # ─ ASCENT: gradient following ─────────────────────────────────────
        if self._should_ascent(gm):
            if self._state != STATE_ASCENT:
                if self._nav2_state != NAV2_IDLE:
                    self._cancel_nav2_goal()
                self._state = STATE_ASCENT; self._state_t = now
                self._search_rounds = 0; self._frontier_cold_start = None
            lin,ang=self._gradient_cmd()
            self._pub.publish(self._make_cmd(lin,ang)); return

        # ─ FRONTIER_NAV: large-scale exploration ──────────────────────────
        if self._state == STATE_ASCENT:
            self._state = STATE_FRONTIER_NAV; self._state_t = now
            self._search_rounds = 0; self._refresh_frontier(now, force=True)
        if self._state != STATE_FRONTIER_NAV:
            self._state = STATE_FRONTIER_NAV; self._state_t = now
            self._refresh_frontier(now, force=True)

        # Lévy flight when region is exhausted
        if self._search_rounds >= self._levy_rnd_thr and self._peak_cand_t is None:
            nov = self._map_region_novelty(radius=5.0)
            if nov < self._levy_nov_thr:
                self._do_levy_jump(now)

        # Frontier target management
        if self._frontier_target is not None:
            tx,ty=self._frontier_target
            if math.hypot(tx-self._wx,ty-self._wy) <= self._frontier_r:
                self._search_rounds += 1; self._refresh_frontier(now, force=True)
        else:
            self._refresh_frontier(now, force=True)
        self._refresh_frontier(now)

        # ★ FIX-4: FRONTIER_NAV guarded fallback ★
        # Try Nav2 first; evaluate guarded direct navigation this tick.
        # nav2_sent=False because _send_nav2_goal returns False until goal is accepted.
        if self._frontier_target is not None:
            tx, ty = self._frontier_target
            # Try Nav2 (rate-limited, non-blocking)
            self._send_nav2_goal(tx, ty)
            self._nav2_progress_stalled(tx, ty, now, 'frontier')
            # Use direct motion only under the local LaserScan guard.
            if self._nav2_state != NAV2_ACTIVE:
                lin, ang, _, self._frontier_direct_safe = self._direct_guarded_cmd(
                    tx, ty, self._frontier_lin, now, 'frontier')
                self._pub.publish(self._make_cmd(lin, ang))
        else:
            # No frontier yet: slow rotation to gather sensor data
            self._pub.publish(self._make_cmd(0.0, self._max_ang * 0.4))

        # Periodic status log
        if int(elapsed*self._rate) % 50 == 0:
            ft = (f'({self._frontier_target[0]:.1f},{self._frontier_target[1]:.1f})'
                  if self._frontier_target else 'None')
            nov = self._map_region_novelty()
            guard_r = max(0, self._pc_guard_t - now)
            cold_t = (now - self._frontier_cold_start) if self._frontier_cold_start else 0.0
            self.get_logger().info(
                f'[{self._state}] t={elapsed:.0f}s '
                f'world=({self._wx:.2f},{self._wy:.2f}) TF={tf_ok} '
                f'T={T:.1f}C rise={trise:.2f}C '
                f'adapt_pk={self._adapt_pk_delta():.1f}C guard={guard_r:.0f}s '
                f'cold={cold_t:.0f}s/{self._frontier_cold_to:.0f}s '
                f'nav2={self._nav2_state} '
                f'found={n_found} path={self._path_length:.2f}m '
                f'frontier={ft} nov={nov:.2f}')


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
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
