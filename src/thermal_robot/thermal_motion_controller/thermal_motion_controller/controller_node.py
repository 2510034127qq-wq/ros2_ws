#!/usr/bin/env python3
"""
controller_node.py — Thermal Gradient Navigation Controller v31
================================================================
Generalizable multi-source thermal navigation algorithm.
Core principle: robot navigates purely from sensor signals, never knows
source coordinates a priori.

v31 fixes vs v30 (targeted, algorithm logic unchanged):
  FIX-1: SLAM coordinate offset.
    SLAM map frame origin = robot spawn position in world.
    TF gives map coords → add spawn offset → world coords.
    world_x = slam_x + spawn_x  (slam_x≈0 at spawn → world_x = spawn_x ✓)

  FIX-2: COARSE_SURVEY direct navigation fallback.
    When Nav2 unavailable/failed, robot must still move toward waypoint.
    Added direct /cmd_vel fallback when nav2_state != NAV2_ACTIVE.
    This was the primary cause of robot stopping after ~40s.

  FIX-3: Nav2 goal frame correction.
    Nav2 expects goals in 'map' frame.
    world → map: map_x = world_x - spawn_x  (inverse of FIX-1).
    Previously goals were sent in world frame → Nav2 rejected all plans.

  FIX-4: FRONTIER_NAV direct fallback robustness.
    Guaranteed cmd_vel output every tick regardless of Nav2 state.

Generalizability design:
  - Works without SLAM (falls back to odom-based position)
  - Works without Nav2 (falls back to direct /cmd_vel everywhere)
  - Gradient ascent, Lévy flight, belief map logic unchanged
  - All state transitions driven by thermal sensor signals only

Architecture (layered):
  FINE states (direct /cmd_vel):
    ASCENT, CONVERGE, SAMPLE, AT_PEAK, RELOCATE, ESCAPE
  COARSE states (Nav2 preferred + direct fallback):
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
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from thermal_interfaces.msg import GradientArray

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
STATE_DONE          = 'DONE'

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
        self.declare_parameter('max_linear_vel',             0.25)
        self.declare_parameter('max_angular_vel',            0.5)
        self.declare_parameter('kp_angular',                 1.2)
        self.declare_parameter('ang_smooth_alpha',           0.55)
        self.declare_parameter('lin_smooth_alpha',           0.4)
        self.declare_parameter('min_gradient_mag',           0.15)
        self.declare_parameter('peak_window_s',              4.0)
        self.declare_parameter('peak_temp_delta',           15.0)
        self.declare_parameter('plateau_thresh',             2.0)
        self.declare_parameter('peak_confirm_s',             1.5)
        self.declare_parameter('peak_confirm_lin_vel',       0.0)
        self.declare_parameter('ambient_temp',              -1.0)
        self.declare_parameter('ambient_update_margin',      1.5)
        self.declare_parameter('rearm_cool_delta',           6.0)
        self.declare_parameter('relocate_dist',              3.0)
        self.declare_parameter('relocate_lin_vel',           0.2)
        self.declare_parameter('peak_hold_s',                1.5)
        self.declare_parameter('revisit_radius',             2.0)
        self.declare_parameter('source_repulsion_k',         0.8)
        self.declare_parameter('source_repulsion_min_dist',  0.5)
        self.declare_parameter('source_exclusion_radius',    2.0)
        self.declare_parameter('min_escape_dist',            2.0)
        self.declare_parameter('num_sources',                3)
        self.declare_parameter('no_new_source_timeout',     60.0)
        self.declare_parameter('spawn_x',                   -6.0)
        self.declare_parameter('spawn_y',                    0.0)
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
        self.declare_parameter('frontier_nav_lin_vel',       0.22)
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
        self.declare_parameter('converge_circle_radius',     1.5)
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
        self.declare_parameter('levy_post_confirm_step',     10.0)
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
        self.declare_parameter('frontier_cold_timeout_s',    20.0)
        self.declare_parameter('departure_timeout_s',       120.0)
        self.declare_parameter('departure_speed',             0.22)

        # ── Read parameters ───────────────────────────────────────────────
        g = self.get_parameter
        self._rate         = float(g('publish_rate').value)
        self._max_lin      = float(g('max_linear_vel').value)
        self._max_ang      = float(g('max_angular_vel').value)
        self._kp_ang       = float(g('kp_angular').value)
        self._alpha_ang    = float(g('ang_smooth_alpha').value)
        self._alpha_lin    = float(g('lin_smooth_alpha').value)
        self._min_gmag     = float(g('min_gradient_mag').value)
        self._win_n        = max(4, int(float(g('peak_window_s').value)*self._rate))
        self._pk_tdelta    = float(g('peak_temp_delta').value)
        self._plateau      = float(g('plateau_thresh').value)
        self._pk_conf_s    = float(g('peak_confirm_s').value)
        self._pk_conf_lin  = float(g('peak_confirm_lin_vel').value)
        _ambient_param     = float(g('ambient_temp').value)
        self._amb_margin   = float(g('ambient_update_margin').value)
        self._rearm_delta  = float(g('rearm_cool_delta').value)
        self._reloc_dist   = float(g('relocate_dist').value)
        self._reloc_lin    = float(g('relocate_lin_vel').value)
        self._peak_hold    = float(g('peak_hold_s').value)
        self._rev_r        = float(g('revisit_radius').value)
        self._rep_k        = float(g('source_repulsion_k').value)
        self._rep_min      = float(g('source_repulsion_min_dist').value)
        self._excl_r       = float(g('source_exclusion_radius').value)
        self._min_esc_dist = float(g('min_escape_dist').value)
        self._num_src      = int(g('num_sources').value)
        self._no_new_t     = float(g('no_new_source_timeout').value)
        self._spawn_x      = float(g('spawn_x').value)
        self._spawn_y      = float(g('spawn_y').value)
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
        self._conv_r       = float(g('converge_circle_radius').value)
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
        self._levy_pc_step        = float(g('levy_post_confirm_step').value)
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

        self._pre_pk_thresh = self._pre_pk_ratio * self._pk_tdelta

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
        self._wx       = self._spawn_x   # world X (updated from TF or odom)
        self._wy       = self._spawn_y   # world Y
        self._tf_ready = False
        self._prev_pos: Optional[Tuple[float,float]] = None
        self._path_length = 0.0

        # ── Perception ────────────────────────────────────────────────────
        self._last_ga: Optional[GradientArray] = None
        self._temp_win: deque = deque(maxlen=self._win_n)
        self._temp_max_seen = 25.0
        self._ang_smooth = 0.0
        self._lin_smooth = 0.0

        # ── Mission state ─────────────────────────────────────────────────
        self._found_sources: List[Tuple[float,float,float]] = []
        self._peak_cand_t: Optional[float] = None
        self._peak_armed   = True
        self._t0           = time.monotonic()
        self._last_found_t = time.monotonic()

        # ── State machine ─────────────────────────────────────────────────
        self._state         = STATE_FRONTIER_NAV
        self._state_t       = time.monotonic()
        self._search_rounds = 0
        self._locked_yaw: Optional[float]               = None
        self._move_start:  Optional[Tuple[float,float]] = None
        self._escape_mode: str                          = EMODE_SOURCE_AVOID

        # ── CONVERGE ─────────────────────────────────────────────────────
        self._converge_center: Optional[Tuple[float,float]] = None
        self._converge_best_T: float = 0.0
        self._converge_best_pos: Optional[Tuple[float,float]] = None
        self._converge_ang_vel: float = 0.0
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

        # ── COARSE_SURVEY ─────────────────────────────────────────────────
        self._coarse_wp: Optional[Tuple[float,float]] = None
        self._coarse_wp_t: float = 0.0
        self._last_survey_pause_t: float = 0.0
        self._survey_buf: List[float] = []
        self._survey_t_start: Optional[float] = None
        self._coarse_wp_count: int = 0

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
        self._frontier_last_upd: float = 0.0

        # ── [v30] TF2 for SLAM position ───────────────────────────────────
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ── [v30] Nav2 NavigateToPose Action Client ───────────────────────
        self._nav2_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._nav2_state: str                   = NAV2_IDLE
        self._nav2_goal_handle                  = None
        self._nav2_current_goal: Optional[Tuple[float,float]] = None
        self._nav2_last_send_t: float           = 0.0
        self._nav2_ready: bool = False
        self._nav2_check_t: float = 0.0

        # ── QoS ──────────────────────────────────────────────────────────
        be = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        history=QoSHistoryPolicy.KEEP_LAST, depth=5,
                        durability=QoSDurabilityPolicy.VOLATILE)
        rel = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=5,
                         durability=QoSDurabilityPolicy.VOLATILE)

        self.create_subscription(GradientArray, '/thermal/gradient', self._grad_cb, rel)
        self.create_subscription(Odometry,      '/odom',             self._odom_cb, be)
        self._pub   = self.create_publisher(Twist, '/cmd_vel', 10)
        self._timer = self.create_timer(1.0/self._rate, self._timer_cb)

        self.get_logger().info(
            f'controller_node v31 | FIX: SLAM coord offset + COARSE fallback + Nav2 frame | '
            f'COARSE→Nav2+fallback | FINE→direct /cmd_vel | '
            f'spawn=({self._spawn_x},{self._spawn_y}) | num_sources={self._num_src}')

    # ────────────────────────────────────────────────────────────────────────
    # [FIX-1] TF position update: SLAM map coords → world coords
    # ────────────────────────────────────────────────────────────────────────

    def _update_world_pos_from_tf(self) -> bool:
        """
        Get world position from SLAM TF (map→base_link).

        FIX-1: SLAM map frame origin = robot spawn position in world.
        Conversion: world = slam_map + spawn_offset
          - At robot spawn: SLAM says (0,0) → world is (spawn_x, spawn_y) ✓
          - Robot moves +1m in X: SLAM says (1,0) → world is (spawn_x+1, spawn_y) ✓

        Fallback to odom integration when SLAM TF unavailable (startup / SLAM failure).
        """
        try:
            t = self._tf_buffer.lookup_transform(
                'map', 'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))

            # FIX-1: Add spawn offset to convert SLAM map frame → world frame
            self._wx = t.transform.translation.x + self._spawn_x
            self._wy = t.transform.translation.y + self._spawn_y

            q = t.transform.rotation
            siny = 2.0*(q.w*q.z + q.x*q.y)
            cosy = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
            self._odom_yaw = math.atan2(siny, cosy)

            if not self._tf_ready:
                self._tf_ready = True
                self.get_logger().info(
                    f'[TF_READY] SLAM TF (map→base_link) available '
                    f'world_pos=({self._wx:.2f},{self._wy:.2f}) '
                    f'[slam=({t.transform.translation.x:.2f},{t.transform.translation.y:.2f}) '
                    f'+ spawn=({self._spawn_x},{self._spawn_y})]')
            return True

        except (LookupException, ExtrapolationException, ConnectivityException):
            # SLAM not ready: fallback to odom
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

        FIX-3: Nav2 expects goals in 'map' frame.
        tx, ty are in world frame. Convert: map = world - spawn.
        world = slam + spawn → slam(map) = world - spawn.

        Rate limit: max one goal attempt per 3s to avoid rejection spam.
        Returns True only when Nav2 is actively navigating to this goal.
        """
        if not self._check_nav2_ready():
            return False

        now = time.monotonic()

        # Already navigating to same goal
        if self._nav2_state == NAV2_ACTIVE and self._nav2_current_goal is not None:
            cx, cy = self._nav2_current_goal
            if math.hypot(tx-cx, ty-cy) < 0.5:
                return True

        # Waiting for server response
        if self._nav2_state == NAV2_SENDING:
            return False

        # Rate limit: max one attempt per 3s
        if self._nav2_state in (NAV2_IDLE, NAV2_DONE):
            if (now - self._nav2_last_send_t) < 3.0:
                return False

        # FIX-3: Convert world coords → map frame for Nav2
        # map_x = world_x - spawn_x  (inverse of FIX-1)
        goal_map_x = tx - self._spawn_x
        goal_map_y = ty - self._spawn_y

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp    = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(goal_map_x)
        goal_msg.pose.pose.position.y = float(goal_map_y)
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.w = 1.0

        self._nav2_state        = NAV2_SENDING
        self._nav2_current_goal = (tx, ty)
        self._nav2_last_send_t  = now

        send_future = self._nav2_client.send_goal_async(
            goal_msg, feedback_callback=self._nav2_feedback_cb)
        send_future.add_done_callback(self._nav2_goal_response_cb)

        self.get_logger().info(
            f'[NAV2→] Goal world=({tx:.1f},{ty:.1f}) '
            f'map=({goal_map_x:.1f},{goal_map_y:.1f}) state={self._state}')
        return False  # Not active yet; fallback handles movement this tick

    def _nav2_goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            # Goal rejected (e.g. map too small, target unreachable)
            self._nav2_state       = NAV2_IDLE
            self._nav2_goal_handle = None
            # Direct fallback will handle movement next tick
            return
        self._nav2_goal_handle = goal_handle
        self._nav2_state       = NAV2_ACTIVE
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._nav2_result_cb)
        self.get_logger().info(f'[NAV2✓] Goal accepted, navigating...')

    def _nav2_result_cb(self, future):
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
        if self._nav2_goal_handle is not None and self._nav2_state == NAV2_ACTIVE:
            self._nav2_goal_handle.cancel_goal_async()
            self.get_logger().info('[NAV2✗cancel] Cancelled → switching to direct /cmd_vel')
        self._nav2_state        = NAV2_IDLE
        self._nav2_goal_handle  = None
        self._nav2_current_goal = None

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
                if current_trise > 3.0:
                    self._bmap.update(self._wx, self._wy,
                                      current_trise, heat_sigma=self._heat_sigma)
        else:
            if T < self._ambient_est + self._amb_margin:
                self._ambient_est = (self._ambient_ema_alpha*T
                                     + (1.0-self._ambient_ema_alpha)*self._ambient_est)

        self._temp_win.append(T)
        if T > self._temp_max_seen:
            self._temp_max_seen = T
        if self._calib_done and trise > _WARM_TEMP_DELTA:
            self._last_warm_pos = (self._wx, self._wy)
        if self._calib_done:
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

    def _escape_yaw(self):
        if self._found_sources:
            vx,vy=self._repulsion_vec()
            rep_yaw=math.atan2(vy,vx)
        else:
            rep_yaw=math.atan2(-math.sin(self._odom_yaw),
                               -math.cos(self._odom_yaw))
        if self._esc_fr_bias>0.0:
            frontier=self._bmap.best_frontier(
                self._wx,self._wy,min_d=4.0,max_d=14.0,
                known_sources=self._known_source_positions(),
                safe_dist=self._safe_dist())
            if frontier is not None:
                fx,fy,_=frontier
                fr_yaw=math.atan2(fy-self._wy,fx-self._wx)
                w=self._esc_fr_bias
                rx=(1.0-w)*math.cos(rep_yaw)+w*math.cos(fr_yaw)
                ry=(1.0-w)*math.sin(rep_yaw)+w*math.sin(fr_yaw)
                return math.atan2(ry,rx)
        return rep_yaw

    def _yaw_toward(self, tx, ty):
        return math.atan2(ty-self._wy, tx-self._wx)

    def _drive_toward_yaw(self, yaw, lin=None):
        err=math.atan2(math.sin(yaw-self._odom_yaw),math.cos(yaw-self._odom_yaw))
        ang=float(np.clip(self._kp_ang*err,-self._max_ang,self._max_ang))
        v=lin if lin is not None else self._reloc_lin
        v=v if abs(err)<math.radians(35) else 0.05
        return v, ang

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
        return math.hypot(px-32, py-24) <= self._sample_center_px_r

    def _peak_fov_dist_m(self):
        if not self._last_ga:
            return 99.0
        return math.hypot(int(self._last_ga.peak_pixel_x)-32,
                          int(self._last_ga.peak_pixel_y)-24) * 0.0625

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
                           STATE_RELOCATE, STATE_ESCAPE, STATE_DONE,
                           STATE_COARSE_SURVEY, STATE_SURVEY_PAUSE,
                           STATE_DEPARTURE):
            return False
        if self._is_near_known():
            return False
        if now < self._pc_guard_t:
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
        if not force and (now-self._frontier_last_upd)<self._frontier_upd:
            return
        self._frontier_last_upd=now
        ksrc=self._known_source_positions()
        if self._pc_rounds_left>0:
            d_near=self._nearest_known_dist() if self._found_sources else 0.0
            min_d=max(self._pc_min_d,d_near+self._excl_r+1.5)
            d_sig=self._pc_dist_sigma; self._pc_rounds_left-=1
            mode_str=f'BOOST min_d={min_d:.1f}m'
        else:
            min_d=3.0; d_sig=8.0; mode_str='NORMAL'
        ft=self._bmap.best_frontier(self._wx,self._wy,min_d=min_d,max_d=16.0,
            dist_sigma=d_sig,known_sources=ksrc,safe_dist=self._safe_dist())
        if ft is not None:
            fx,fy,sc=ft; self._frontier_target=(fx,fy)
            self.get_logger().info(
                f'[FRONTIER/{mode_str}] →({fx:.1f},{fy:.1f}) score={sc:.3f} '
                f'd={math.hypot(fx-self._wx,fy-self._wy):.1f}m')
        else:
            self._do_levy_jump(now)

    def _do_levy_jump(self, now):
        step=self._levy_step(); ksrc=self._known_source_positions()
        ft=self._bmap.best_frontier(self._wx,self._wy,min_d=step*0.4,max_d=step*1.6,
            known_sources=ksrc,safe_dist=self._safe_dist())
        direction=(math.atan2(ft[1]-self._wy,ft[0]-self._wx)
                   if ft is not None else random.uniform(-math.pi,math.pi))
        lx=self._wx+step*math.cos(direction); ly=self._wy+step*math.sin(direction)
        self._frontier_target=(lx,ly); self._frontier_last_upd=now; self._search_rounds=0
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
        cx, cy = self._sources_centroid()
        best_yaw=random.uniform(-math.pi,math.pi); best_score=-1.0
        for i in range(24):
            yaw=-math.pi+(2.0*math.pi/24.0)*i; score=0.0
            for frac in (0.4,0.65,0.85,1.0):
                px=cx+self._departure_dist*frac*math.cos(yaw)
                py=cy+self._departure_dist*frac*math.sin(yaw)
                ci,cj=self._bmap._ci(px,py)
                nov=1.0/(1.0+float(self._bmap.visit[ci,cj]))
                src_ok=1.0
                if self._found_sources:
                    min_src_d=min(math.hypot(px-sx,py-sy) for sx,sy,_ in self._found_sources)
                    src_ok=1.0 if min_src_d>self._departure_dist*0.3 else 0.0
                score+=nov*src_ok
            if score>best_score: best_score=score; best_yaw=yaw
        tx=cx+self._departure_dist*math.cos(best_yaw)
        ty=cy+self._departure_dist*math.sin(best_yaw)
        self.get_logger().info(
            f'[DEPARTURE_WP] centroid=({cx:.1f},{cy:.1f}) '
            f'→({tx:.1f},{ty:.1f}) d_robot={math.hypot(tx-self._wx,ty-self._wy):.1f}m')
        return (tx, ty)

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

        if dist <= self._frontier_r * 2.0:
            self.get_logger().info(f'[DEPARTURE→COARSE] arrived d={dist:.2f}m')
            self._enter_coarse_survey(reason='departure_arrived'); return
        if elapsed >= self._departure_timeout:
            self.get_logger().warn(f'[DEPARTURE→COARSE] timeout {elapsed:.0f}s')
            self._enter_coarse_survey(reason='departure_timeout'); return

        nav_yaw = self._yaw_toward(tx, ty)
        lin, ang = self._drive_toward_yaw(nav_yaw, self._departure_speed)
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
        ksrc=self._known_source_positions()
        ft=self._bmap.best_frontier(self._wx,self._wy,
            min_d=self._survey_wp_min_d,max_d=self._survey_wp_max_d,
            dist_sigma=10.0,heat_prior=0.0,
            known_sources=ksrc,safe_dist=self._survey_safe_dist)
        return (ft[0],ft[1]) if ft is not None else None

    def _enter_coarse_survey(self, initial_wp=None, reason='unknown'):
        now=time.monotonic()
        self._state=STATE_COARSE_SURVEY; self._state_t=now
        self._survey_buf=[]; self._survey_t_start=None
        if initial_wp is not None:
            self._coarse_wp=initial_wp
        else:
            wp=self._coarse_waypoint()
            self._coarse_wp=wp or (self._wx+5, self._wy)
        self._coarse_wp_t=now; self._coarse_wp_count=0
        self._last_survey_pause_t=now; self._frontier_cold_start=None
        self.get_logger().info(
            f'[→COARSE_SURVEY] reason={reason} '
            f'wp=({self._coarse_wp[0]:.1f},{self._coarse_wp[1]:.1f}) '
            f'd={math.hypot(self._coarse_wp[0]-self._wx,self._coarse_wp[1]-self._wy):.1f}m')
        # Attempt Nav2 goal; direct fallback guarantees movement if Nav2 fails
        self._send_nav2_goal(self._coarse_wp[0], self._coarse_wp[1])

    def _exec_coarse_survey(self, now: float):
        """
        COARSE_SURVEY: systematic area coverage for heat source search.

        FIX-2: Added guaranteed direct navigation fallback.
        When Nav2 is unavailable or rejected the goal, robot MUST still move
        toward the waypoint using direct /cmd_vel.

        Priority order:
          P1: Real-time thermal signal > threshold → enter FINE mode (ASCENT)
          P2: Periodic survey pause (stop+sense)
          P3: Waypoint management (arrival/timeout → new waypoint)
          P4: Nav2 goal maintenance (try to keep Nav2 active)
          P5: [FIX-2] Direct nav fallback (always runs when Nav2 not active)
        """
        trise = self._temp_rise()

        # P1: Real-time thermal signal → switch to FINE gradient navigation
        if self._calib_done and trise >= self._coarse_transit_thr and not self._is_near_known():
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
                self._coarse_wp_t = now
            tx, ty = self._coarse_wp
            # Try Nav2; fallback below guarantees movement if it fails
            self._send_nav2_goal(tx, ty)

        elif self._nav2_state in (NAV2_IDLE, NAV2_DONE):
            # Nav2 finished/failed: retry
            self._send_nav2_goal(tx, ty)

        # ★ FIX-2: Direct navigation fallback ★
        # This runs EVERY tick when Nav2 is not actively navigating.
        # Ensures robot always moves, regardless of Nav2 availability.
        tx2, ty2 = self._coarse_wp
        dist2 = math.hypot(tx2 - self._wx, ty2 - self._wy)

        if self._nav2_state != NAV2_ACTIVE:
            if dist2 > self._frontier_r:
                # Drive directly toward waypoint
                nav_yaw = self._yaw_toward(tx2, ty2)
                lin, ang = self._drive_toward_yaw(nav_yaw, self._frontier_lin)
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
        self._converge_center=(self._wx,self._wy)
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
                self._converge_center=self._conv_best_global_pos
                self._converge_best_T=0.0; self._converge_best_pos=None
                self._conv_cold_t=None; self._state_t=now
                self._pub.publish(Twist()); return
            self._state=STATE_FRONTIER_NAV; self._state_t=now
            self._converge_center=None; self._conv_cold_t=None
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
                self._converge_center=None; self._conv_cold_t=None
                self._refresh_frontier(now,force=True); self._pub.publish(Twist()); return
        else:
            self._conv_cold_t=None
        sample_trigger=self._adapt_pk_delta()*self._conv_sample_ratio
        near_center=self._is_peak_near_fov_center(); fov_dist_m=self._peak_fov_dist_m()
        if trise>=sample_trigger and not self._is_near_known():
            if near_center:
                self._state=STATE_SAMPLE; self._state_t=now
                self._sample_t_start=now; self._sample_T_buf=[T]
                self._conv_cold_t=None; self._pub.publish(Twist()); return
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
            if not self._is_near_known():
                self._found_sources.append((self._wx,self._wy,T))
                self._last_found_t=now; self._esc_loop_count=0
                self._bmap.mark_excluded(self._wx,self._wy,self._excl_r)
                self._bmap.suppress_confirmed_source(self._wx,self._wy,self._excl_r+1.0)
                self._peak_armed=False; self._state=STATE_AT_PEAK; self._state_t=now
                self._converge_center=None
                kstr=', '.join(f'({s[0]:.1f},{s[1]:.1f})' for s in self._found_sources)
                elapsed=now-self._t0
                self.get_logger().info(
                    f'★ [SOURCE #{len(self._found_sources)}] '
                    f'pos=({self._wx:.2f},{self._wy:.2f}) T={T:.1f}C '
                    f'path={self._path_length:.2f}m t={elapsed:.1f}s')
                self.get_logger().info(f'  Known sources: [{kstr}]')
                self._conv_sticky_count=0; self._conv_best_global_T=0.0
                self._conv_best_global_pos=None; self._conv_returning=False
                self._T_max_unconf=self._ambient_est
                self._pc_guard_t=now+self._pc_cooldown_s
                self.get_logger().info(
                    f'[CONVERGE_GUARD_SET] block CONVERGE for {self._pc_cooldown_s:.0f}s')
                self._pc_rounds_left=self._pc_rounds
            else:
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
        T=self._current_temp(); gm=self._grad_mag()
        n_found=len(self._found_sources); trise=self._temp_rise()

        # FIX-1: Update world position from SLAM TF every tick
        tf_ok = self._update_world_pos_from_tf()
        if tf_ok:
            if self._prev_pos is not None:
                self._path_length += math.hypot(
                    self._wx-self._prev_pos[0], self._wy-self._prev_pos[1])
            self._prev_pos = (self._wx, self._wy)

        self._update_armed(T)

        # ─ DONE ──────────────────────────────────────────────────────────
        if self._state == STATE_DONE:
            self._pub.publish(Twist()); return
        if self._num_src < 0 and n_found > 0:
            if (now - self._last_found_t) > self._no_new_t:
                self._cancel_nav2_goal()
                self._state = STATE_DONE
                self.get_logger().info(
                    f'[ALL DONE] {n_found} sources found '
                    f'path={self._path_length:.2f}m t={elapsed:.1f}s')
                self._pub.publish(Twist()); return

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
                if self._num_src > 0 and n_found >= self._num_src:
                    self._cancel_nav2_goal()
                    self._state = STATE_DONE
                    self.get_logger().info(
                        f'[ALL DONE] {n_found} sources '
                        f'path={self._path_length:.2f}m t={elapsed:.1f}s')
                else:
                    self._state = STATE_RELOCATE; self._state_t = now
                    self._move_start = (self._wx, self._wy)
                    self._locked_yaw = self._escape_yaw()
                    self._peak_cand_t = None; self._temp_max_seen = self._ambient_est
                    self._temp_win.clear(); self._search_rounds = 0
                    self._esc_loop_count = 0
                    self.get_logger().info(
                        f'[→RELOCATE] yaw={math.degrees(self._locked_yaw):.0f}deg '
                        f'{n_found}/{self._num_src or "inf"}')
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
            nov = self._bmap.region_novelty(self._wx, self._wy, radius=5.0)
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

        # ★ FIX-4: FRONTIER_NAV guaranteed movement ★
        # Try Nav2 first; always fallback to direct nav this tick.
        # nav2_sent=False because _send_nav2_goal returns False until goal is accepted.
        if self._frontier_target is not None:
            tx, ty = self._frontier_target
            # Try Nav2 (rate-limited, non-blocking)
            self._send_nav2_goal(tx, ty)
            # Always ensure robot moves this tick
            if self._nav2_state != NAV2_ACTIVE:
                # Direct navigation fallback (works with or without Nav2)
                lin, ang = self._drive_toward_yaw(self._yaw_toward(tx, ty),
                                                   self._frontier_lin)
                self._pub.publish(self._make_cmd(lin, ang))
        else:
            # No frontier yet: slow rotation to gather sensor data
            self._pub.publish(self._make_cmd(0.05, self._max_ang * 0.4))

        # Periodic status log
        if int(elapsed*self._rate) % 50 == 0:
            ft = (f'({self._frontier_target[0]:.1f},{self._frontier_target[1]:.1f})'
                  if self._frontier_target else 'None')
            nov = self._bmap.region_novelty(self._wx, self._wy)
            guard_r = max(0, self._pc_guard_t - now)
            cold_t = (now - self._frontier_cold_start) if self._frontier_cold_start else 0.0
            self.get_logger().info(
                f'[{self._state}] t={elapsed:.0f}s '
                f'world=({self._wx:.2f},{self._wy:.2f}) TF={tf_ok} '
                f'T={T:.1f}C rise={trise:.2f}C '
                f'adapt_pk={self._adapt_pk_delta():.1f}C guard={guard_r:.0f}s '
                f'cold={cold_t:.0f}s/{self._frontier_cold_to:.0f}s '
                f'nav2={self._nav2_state} '
                f'found={n_found}/{self._num_src} path={self._path_length:.2f}m '
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
