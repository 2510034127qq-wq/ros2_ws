#!/usr/bin/env python3
"""
plot_slam_nav2.py — SLAM + Nav2 Integration Visualization (v2)
==============================================================
Generates 5 figures from data collected by collect_sim_data.py:

  01_slam_vs_odom.png      — SLAM trajectory (map frame) vs Odom trajectory
  02_scan_coverage.png     — LiDAR scan statistics (SLAM input quality)
  03_nav2_plans.png        — Nav2 global plan statistics
  04_slam_quality.png      — SLAM pose quality metrics
  05_slam_nav2_dashboard   — SLAM+Nav2 comprehensive dashboard

Usage:
  python3 plot_slam_nav2.py <data_dir>
  python3 plot_slam_nav2.py          # auto-uses latest directory

Fixes (v2):
  - Curvature calculation: dx/ds broadcast shape mismatch fixed
  - All Chinese text replaced with English (eliminates font warnings)
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── Global style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d0d17',
    'axes.facecolor':   '#0d0d17',
    'text.color':       '#e0e0e0',
    'axes.labelcolor':  '#e0e0e0',
    'axes.edgecolor':   '#444466',
    'xtick.color':      '#aaaacc',
    'ytick.color':      '#aaaacc',
    'grid.color':       '#222244',
    'grid.linewidth':   0.5,
    'axes.titlesize':   10,
    'axes.labelsize':   8,
    'font.size':        8,
    'lines.linewidth':  1.2,
})

SPAWN_X = -6.0
SPAWN_Y =  0.0
CONFIG_B_SOURCES = [
    {'name': 'SA_left', 'xy': [-1.0,  3.5], 'peak': 57.0, 'color': '#ff4444'},
    {'name': 'SB_far',  'xy': [ 6.0, -3.0], 'peak': 44.0, 'color': '#ff8844'},
    {'name': 'SC_weak', 'xy': [-5.0, -5.5], 'peak': 38.0, 'color': '#ffcc44'},
]


# ─── Utilities ───────────────────────────────────────────────────────────────

def load_csv(path: Path) -> dict:
    """Load CSV, return {col_name: np.array}. Returns {} if file missing."""
    if not path.exists():
        return {}
    import csv
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}
    out = {}
    for key in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[key]) if r[key] not in ('', 'None') else float('nan'))
            except (ValueError, TypeError):
                vals.append(float('nan'))
        out[key] = np.array(vals)
    return out


def draw_sources(ax, alpha=0.7):
    """Draw Config-B heat source markers on axis."""
    for src in CONFIG_B_SOURCES:
        x, y = src['xy']
        ax.scatter(x, y, s=80, marker='*', color=src['color'],
                   zorder=6, alpha=alpha, label=f"{src['name']} ({src['peak']:.0f}C)")


def thermal_field_bg(ax, xrange=(-12, 12), yrange=(-9, 9), resolution=150, ambient=22.0):
    """Draw Config-B thermal field contours in axis background."""
    xs = np.linspace(xrange[0], xrange[1], resolution)
    ys = np.linspace(yrange[0], yrange[1], resolution)
    XX, YY = np.meshgrid(xs, ys)
    T = np.full_like(XX, ambient)
    specs = [
        (-1.0,  3.5, 35.0, 1.1),
        ( 6.0, -3.0, 22.0, 0.9),
        (-5.0, -5.5, 16.0, 0.8),
    ]
    for sx, sy, amp, sigma in specs:
        T += amp * np.exp(-((XX - sx)**2 + (YY - sy)**2) / (2 * sigma**2))
    ax.contourf(XX, YY, T, levels=20, cmap='inferno', alpha=0.25)
    ax.contour( XX, YY, T, levels=10, colors='#ffffff', alpha=0.08, linewidths=0.4)


# ─── Figure 1: SLAM vs Odom trajectory ───────────────────────────────────────

def fig_slam_vs_odom(out_dir: Path, meta: dict):
    traj      = load_csv(out_dir / 'trajectory.csv')
    slam_traj = load_csv(out_dir / 'slam_trajectory.csv')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0d0d17')
    fig.suptitle(
        'SLAM Trajectory (map frame) vs Odometry Trajectory\n'
        '(Left: Odometry cumulative drift path  |  Right: SLAM-corrected path)',
        color='#e0e0e0', fontsize=11)

    for ax in axes:
        ax.set_facecolor('#0d0d17')
        ax.grid(True, alpha=0.2)
        thermal_field_bg(ax)
        draw_sources(ax)
        ax.scatter(SPAWN_X, SPAWN_Y, s=80, marker='^', color='#00ff88',
                   zorder=7, label='Spawn')

    # Left: Odom path (spawn + odom → world coords)
    ax0 = axes[0]
    if traj and 'wx' in traj:
        wx, wy = traj['wx'], traj['wy']
        t_arr  = traj.get('t', np.arange(len(wx)))
        sc = ax0.scatter(wx, wy, c=t_arr, cmap='cool', s=1.5, zorder=4)
        fig.colorbar(sc, ax=ax0, label='Time (s)', shrink=0.7)
        ax0.scatter(wx[0],  wy[0],  s=60, marker='^', color='#00ff88', zorder=8)
        ax0.scatter(wx[-1], wy[-1], s=60, marker='s', color='#ff6688', zorder=8)
    ax0.set_title('Odometry Path (spawn+odom, cumulative drift)', color='#e0e0e0')
    ax0.set_xlabel('World X (m)'); ax0.set_ylabel('World Y (m)')
    ax0.legend(fontsize=7, loc='upper left')

    # Right: SLAM map-frame path
    ax1 = axes[1]
    slam_available = bool(slam_traj and 'x' in slam_traj and len(slam_traj['x']) > 0)
    if slam_available:
        sx, sy = slam_traj['x'], slam_traj['y']
        st     = slam_traj.get('t', np.arange(len(sx)))
        sc2 = ax1.scatter(sx, sy, c=st, cmap='plasma', s=1.5, zorder=4)
        fig.colorbar(sc2, ax=ax1, label='Time (s)', shrink=0.7)
        ax1.scatter(sx[0],  sy[0],  s=60, marker='^', color='#00ff88', zorder=8)
        ax1.scatter(sx[-1], sy[-1], s=60, marker='s', color='#ff6688', zorder=8)

        if traj and 'wx' in traj and len(traj['wx']) > 0:
            n_common = min(len(sx), len(traj['wx']))
            drift = np.sqrt((sx[:n_common] - traj['wx'][:n_common])**2 +
                            (sy[:n_common] - traj['wy'][:n_common])**2)
            ax1.set_title(
                f'SLAM-corrected Path (map frame) | mean drift correction={drift.mean():.2f}m',
                color='#e0e0e0')
        else:
            ax1.set_title('SLAM-corrected Path (map frame)', color='#e0e0e0')
    else:
        ax1.text(0.5, 0.5,
                 'SLAM data unavailable\n(slam_trajectory.csv is empty)',
                 ha='center', va='center', transform=ax1.transAxes,
                 color='#ffaa44', fontsize=11)
        ax1.set_title('SLAM Path (unavailable)', color='#ffaa44')

    ax1.set_xlabel('Map X (m)'); ax1.set_ylabel('Map Y (m)')
    ax1.legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    out_path = out_dir / '01_slam_vs_odom.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print('  ✓ 01_slam_vs_odom.png')
    return out_path


# ─── Figure 2: LiDAR scan coverage statistics ────────────────────────────────

def fig_scan_coverage(out_dir: Path, meta: dict):
    scan = load_csv(out_dir / 'scan_stats.csv')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor('#0d0d17')
    fig.suptitle('/scan LiDAR Statistics (SLAM Input Quality Assessment)',
                 color='#e0e0e0', fontsize=11)
    for ax in axes.flat:
        ax.set_facecolor('#0d0d17')
        ax.grid(True, alpha=0.2)

    if not scan or 't' not in scan:
        for ax in axes.flat:
            ax.text(0.5, 0.5,
                    'No LiDAR scan data\n(/scan not published during collection)',
                    ha='center', va='center', transform=ax.transAxes,
                    color='#ffaa44', fontsize=11)
    else:
        t  = scan['t']
        vr = scan.get('valid_ratio', np.ones_like(t))
        rm = scan.get('range_mean',  np.zeros_like(t))
        rx = scan.get('range_max',   np.zeros_like(t))
        nv = scan.get('n_valid',     np.zeros_like(t))

        # Valid ray ratio
        axes[0, 0].plot(t, vr * 100, color='#44ddff', linewidth=1.0)
        axes[0, 0].axhline(y=95, color='#44ff88', linestyle='--',
                           alpha=0.6, label='95% target')
        axes[0, 0].set_title('Valid Ray Ratio (%)')
        axes[0, 0].set_xlabel('Time (s)'); axes[0, 0].set_ylabel('%')
        axes[0, 0].legend(fontsize=7)
        axes[0, 0].set_ylim(0, 105)

        # Range statistics
        axes[0, 1].plot(t, rm, color='#ffaa44', linewidth=1.0, label='Mean')
        axes[0, 1].plot(t, rx, color='#ff4444', linewidth=0.8, alpha=0.6, label='Max')
        axes[0, 1].set_title('Range Statistics (m)')
        axes[0, 1].set_xlabel('Time (s)'); axes[0, 1].set_ylabel('Range (m)')
        axes[0, 1].legend(fontsize=7)

        # Valid ray count distribution
        axes[1, 0].hist(nv, bins=30, color='#44aaff', edgecolor='#2266aa', alpha=0.8)
        axes[1, 0].set_title('Valid Ray Count Distribution')
        axes[1, 0].set_xlabel('N valid rays'); axes[1, 0].set_ylabel('Count')
        axes[1, 0].axvline(x=360 * 0.9, color='#44ff88', linestyle='--',
                           label=f'90%×360={int(360*0.9)}')
        axes[1, 0].legend(fontsize=7)

        # Text summary
        axes[1, 1].axis('off')
        vr_mean = float(np.nanmean(vr)) if len(vr) else 0.0
        rm_mean = float(np.nanmean(rm)) if len(rm) else 0.0
        rx_max  = float(np.nanmax(rx))  if len(rx) and not np.all(np.isnan(rx)) else 0.0
        quality = ('Excellent' if vr_mean > 0.9 else
                   'Good'      if vr_mean > 0.7 else 'Poor')
        summary = (
            f'LiDAR Scan Statistics Summary\n'
            f'─────────────────\n'
            f'Frames recorded: {len(t):>8d}\n'
            f'Mean valid ratio: {vr_mean*100:>6.1f}%\n'
            f'Mean range:      {rm_mean:>7.2f} m\n'
            f'Max range:       {rx_max:>7.2f} m\n'
            f'Coverage period: {float(t[-1]-t[0]):>7.1f} s\n'
            f'─────────────────\n'
            f'SLAM input quality: {quality}'
        )
        axes[1, 1].text(0.1, 0.9, summary, transform=axes[1, 1].transAxes,
                        fontsize=9, verticalalignment='top',
                        fontfamily='monospace', color='#aaddff',
                        bbox=dict(boxstyle='round', facecolor='#111133', alpha=0.8))

    plt.tight_layout()
    out_path = out_dir / '02_scan_coverage.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print('  ✓ 02_scan_coverage.png')
    return out_path


# ─── Figure 3: Nav2 plan statistics ──────────────────────────────────────────

def fig_nav2_plans(out_dir: Path, meta: dict):
    plans = load_csv(out_dir / 'nav2_plan_stats.csv')
    cmdv  = load_csv(out_dir / 'cmd_vel.csv')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor('#0d0d17')
    fig.suptitle('Nav2 Global Plan Statistics (Path Planning Quality Analysis)',
                 color='#e0e0e0', fontsize=11)
    for ax in axes.flat:
        ax.set_facecolor('#0d0d17')
        ax.grid(True, alpha=0.2)

    nav2_avail = bool(plans and 't' in plans and len(plans['t']) > 0)

    if not nav2_avail:
        for ax in axes.flat:
            ax.text(0.5, 0.5,
                    'No Nav2 plan data\n'
                    'Reason: bt_navigator crash (RateController BT node missing)\n'
                    'Fix: remove navigate_through_poses plugin from\n'
                    '     nav2_params.yaml bt_navigator section',
                    ha='center', va='center', transform=ax.transAxes,
                    color='#ffaa44', fontsize=10, linespacing=1.8,
                    bbox=dict(boxstyle='round', facecolor='#221100', alpha=0.7))
        plt.tight_layout()
        out_path = out_dir / '03_nav2_plans.png'
        plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        print('  ✓ 03_nav2_plans.png (Nav2 unavailable)')
        return out_path

    t    = plans['t']
    lens = plans.get('path_length', np.zeros_like(t))
    nwp  = plans.get('n_waypoints', np.zeros_like(t))

    # Plan path length timeline
    axes[0, 0].plot(t, lens, color='#44ff88', linewidth=1.2,
                    marker='o', markersize=3, label='Plan path length')
    axes[0, 0].set_title('Nav2 Plan Path Length vs Time')
    axes[0, 0].set_xlabel('Time (s)'); axes[0, 0].set_ylabel('Path Length (m)')
    axes[0, 0].legend(fontsize=7)

    # Plan trigger interval distribution
    if len(t) > 1:
        intervals = np.diff(t)
        axes[0, 1].hist(intervals, bins=20, color='#4488ff',
                        edgecolor='#2244aa', alpha=0.8)
        axes[0, 1].set_title(f'Plan Trigger Interval Distribution (mean={intervals.mean():.1f}s)')
        axes[0, 1].set_xlabel('Interval (s)'); axes[0, 1].set_ylabel('Count')
        axes[0, 1].axvline(x=intervals.mean(), color='#ff6644', linestyle='--', label='Mean')
        axes[0, 1].legend(fontsize=7)

    # Waypoint count
    axes[1, 0].plot(t, nwp, color='#ffaa44', linewidth=1.0)
    axes[1, 0].set_title('Plan Waypoint Count vs Time')
    axes[1, 0].set_xlabel('Time (s)'); axes[1, 0].set_ylabel('N Waypoints')

    # cmd_vel + Nav2 plan trigger overlay
    if cmdv and 't' in cmdv:
        ct = cmdv['t']
        cv = cmdv.get('lin_x', np.zeros_like(ct))
        axes[1, 1].plot(ct, cv, color='#88ccff', linewidth=0.8, alpha=0.7, label='lin_x')
        for pt in t:
            axes[1, 1].axvline(x=pt, color='#44ff88', alpha=0.4, linewidth=0.6)
        axes[1, 1].set_title('Velocity + Nav2 Plan Triggers (green lines)')
        axes[1, 1].set_xlabel('Time (s)'); axes[1, 1].set_ylabel('v (m/s)')
        axes[1, 1].legend(fontsize=7)

    plt.tight_layout()
    out_path = out_dir / '03_nav2_plans.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print('  ✓ 03_nav2_plans.png')
    return out_path


# ─── Figure 4: SLAM quality metrics ──────────────────────────────────────────

def fig_slam_quality(out_dir: Path, meta: dict):
    slam = load_csv(out_dir / 'slam_trajectory.csv')
    odom = load_csv(out_dir / 'trajectory.csv')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor('#0d0d17')
    fig.suptitle('SLAM Pose Quality Metrics', color='#e0e0e0', fontsize=11)
    for ax in axes.flat:
        ax.set_facecolor('#0d0d17')
        ax.grid(True, alpha=0.2)

    slam_avail = bool(slam and 'x' in slam and len(slam['x']) > 0)
    odom_avail = bool(odom and 'wx' in odom and len(odom['wx']) > 0)

    if not slam_avail:
        for ax in axes.flat:
            ax.text(0.5, 0.5,
                    'SLAM data unavailable\n(slam_trajectory.csv is empty)',
                    ha='center', va='center', transform=ax.transAxes,
                    color='#ffaa44', fontsize=11)
        plt.tight_layout()
        out_path = out_dir / '04_slam_quality.png'
        plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        print('  ✓ 04_slam_quality.png (SLAM unavailable)')
        return out_path

    t_slam = slam['t']

    # --- Sub-plot 1: SLAM pose update rate ---
    if len(t_slam) > 1:
        slam_rates = 1.0 / np.diff(t_slam)
        axes[0, 0].plot(t_slam[1:], slam_rates, color='#44ddff', linewidth=0.8)
        axes[0, 0].axhline(y=10, color='#44ff88', linestyle='--', label='Target 10 Hz')
        axes[0, 0].set_title(f'SLAM Pose Update Rate (mean={slam_rates.mean():.1f} Hz)')
        axes[0, 0].set_xlabel('Time (s)'); axes[0, 0].set_ylabel('Hz')
        axes[0, 0].legend(fontsize=7)

    # --- Sub-plot 2: SLAM speed estimate vs Odom ---
    if odom_avail and len(t_slam) > 10:
        n_common = min(len(t_slam), len(odom['t']))
        if n_common > 10:
            dx_slam = np.diff(slam['x'][:n_common])
            dy_slam = np.diff(slam['y'][:n_common])
            dt_slam = np.diff(t_slam[:n_common])
            v_slam  = np.sqrt(dx_slam**2 + dy_slam**2) / (dt_slam + 1e-9)
            axes[0, 1].plot(t_slam[1:n_common], v_slam, color='#4488ff',
                            linewidth=0.8, label='SLAM estimated speed', alpha=0.8)
            odom_vx = odom.get('vx', np.zeros(len(odom['t'])))
            axes[0, 1].plot(odom['t'][:n_common], np.abs(odom_vx[:n_common]),
                            color='#ff8844', linewidth=0.8, label='Odom speed', alpha=0.8)
            axes[0, 1].set_title('SLAM vs Odom Speed Comparison')
            axes[0, 1].set_xlabel('Time (s)'); axes[0, 1].set_ylabel('Speed (m/s)')
            axes[0, 1].legend(fontsize=7)

    # --- Sub-plot 3: SLAM path curvature ---
    # FIX v2: compute unit tangent first, then diff to get curvature
    if len(t_slam) > 3:
        dx = np.diff(slam['x'])           # shape (N-1,)
        dy = np.diff(slam['y'])           # shape (N-1,)
        ds = np.sqrt(dx**2 + dy**2) + 1e-9  # shape (N-1,)
        tx = dx / ds                      # unit tangent x, shape (N-1,)
        ty = dy / ds                      # unit tangent y, shape (N-1,)
        dtx = np.diff(tx)                 # shape (N-2,)
        dty = np.diff(ty)                 # shape (N-2,)
        curvature = np.sqrt(dtx**2 + dty**2)  # shape (N-2,)
        axes[1, 0].plot(t_slam[2:], curvature, color='#ff88cc', linewidth=0.7)
        axes[1, 0].set_title('SLAM Path Curvature (low=straight, high=turning)')
        axes[1, 0].set_xlabel('Time (s)'); axes[1, 0].set_ylabel('Curvature')
        p99 = float(np.percentile(curvature, 99)) if len(curvature) > 0 else 5.0
        axes[1, 0].set_ylim(0, min(p99, 5.0))

    # --- Sub-plot 4: Text summary ---
    slam_first_t = meta.get('slam_first_t')
    slam_count   = meta.get('slam_pos_count', len(t_slam))
    axes[1, 1].axis('off')
    total_dist = 0.0
    if len(slam['x']) > 1:
        dx2 = np.diff(slam['x'])
        dy2 = np.diff(slam['y'])
        total_dist = float(np.sqrt(dx2**2 + dy2**2).sum())
    duration = float(t_slam[-1] - t_slam[0]) if len(t_slam) > 1 else 0.0
    avg_hz   = slam_count / (duration + 1e-9)
    summary = (
        f'SLAM Quality Summary\n'
        f'─────────────────────\n'
        f'First ready:   {slam_first_t:.1f} s\n'
        f'Total poses:   {slam_count:>8d}\n'
        f'Record period: {duration:.1f} s\n'
        f'Avg rate:      {avg_hz:.1f} Hz\n'
        f'Total path:    {total_dist:.1f} m\n'
        f'─────────────────────\n'
        f'TF data: {"Available" if slam_avail else "Unavailable"}'
    )
    axes[1, 1].text(0.1, 0.9, summary, transform=axes[1, 1].transAxes,
                    fontsize=9, verticalalignment='top',
                    fontfamily='monospace', color='#aaddff',
                    bbox=dict(boxstyle='round', facecolor='#111133', alpha=0.8))

    plt.tight_layout()
    out_path = out_dir / '04_slam_quality.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print('  ✓ 04_slam_quality.png')
    return out_path


# ─── Figure 5: SLAM + Nav2 comprehensive dashboard ───────────────────────────

def fig_slam_nav2_dashboard(out_dir: Path, meta: dict):
    slam  = load_csv(out_dir / 'slam_trajectory.csv')
    odom  = load_csv(out_dir / 'trajectory.csv')
    grad  = load_csv(out_dir / 'gradient_stats.csv')
    cmdv  = load_csv(out_dir / 'cmd_vel.csv')
    scan  = load_csv(out_dir / 'scan_stats.csv')
    plans = load_csv(out_dir / 'nav2_plan_stats.csv')

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0d0d17')
    fig.suptitle(
        'SLAM + Nav2 Thermal Navigation System — Comprehensive Dashboard\n'
        'Route A: Thermal Source Seeking | ROS2 Humble + Gazebo Classic + slam_toolbox + Nav2',
        color='#e0e0e0', fontsize=11, y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)
    ax_traj = fig.add_subplot(gs[0:2, 0:2])   # Large trajectory
    ax_temp = fig.add_subplot(gs[0, 2:4])     # Temperature
    ax_grad = fig.add_subplot(gs[1, 2:4])     # Gradient
    ax_dist = fig.add_subplot(gs[2, 0:2])     # Distance
    ax_vel  = fig.add_subplot(gs[2, 2:3])     # Velocity
    ax_info = fig.add_subplot(gs[2, 3])       # Info box

    for ax in [ax_traj, ax_temp, ax_grad, ax_dist, ax_vel, ax_info]:
        ax.set_facecolor('#0d0d17')
        ax.grid(True, alpha=0.2)

    # ── Trajectory (SLAM preferred, odom fallback) ──────────────────────────
    ax_traj.set_title('Navigation Trajectory (SLAM map frame preferred)', color='#e0e0e0')
    thermal_field_bg(ax_traj)
    draw_sources(ax_traj)

    slam_avail = bool(slam and 'x' in slam and len(slam['x']) > 0)
    odom_avail = bool(odom and 'wx' in odom and len(odom['wx']) > 0)

    if slam_avail:
        sx, sy = slam['x'], slam['y']
        st     = slam.get('t', np.arange(len(sx)))
        sc = ax_traj.scatter(sx, sy, c=st, cmap='cool', s=2, zorder=4, alpha=0.9)
        ax_traj.scatter(sx[0],  sy[0],  s=80, marker='^', color='#00ff88',
                        zorder=8, label='Start (SLAM)')
        ax_traj.scatter(sx[-1], sy[-1], s=60, marker='s', color='#ff6688',
                        zorder=8, label='End (SLAM)')
        plt.colorbar(sc, ax=ax_traj, label='Time (s)', shrink=0.8)
        coord_src = 'SLAM (map frame)'
    elif odom_avail:
        wx, wy = odom['wx'], odom['wy']
        ot = odom.get('t', np.arange(len(wx)))
        sc = ax_traj.scatter(wx, wy, c=ot, cmap='cool', s=2, zorder=4, alpha=0.9)
        ax_traj.scatter(wx[0], wy[0], s=80, marker='^', color='#00ff88',
                        zorder=8, label='Start (odom)')
        plt.colorbar(sc, ax=ax_traj, label='Time (s)', shrink=0.8)
        coord_src = 'Odometry (SLAM unavailable)'
    else:
        ax_traj.text(0.5, 0.5, 'No trajectory data', ha='center', va='center',
                     transform=ax_traj.transAxes, color='#ffaa44', fontsize=12)
        coord_src = 'No data'

    ax_traj.set_xlabel('X (m)'); ax_traj.set_ylabel('Y (m)')
    ax_traj.legend(fontsize=7, loc='upper left')
    ax_traj.text(0.02, 0.02, f'Frame: {coord_src}', transform=ax_traj.transAxes,
                 fontsize=7, color='#88aacc',
                 bbox=dict(boxstyle='round', facecolor='#111133', alpha=0.7))

    # ── Temperature timeline ────────────────────────────────────────────────
    th_stats = load_csv(out_dir / 'thermal_stats.csv')
    if th_stats and 't' in th_stats:
        t_th   = th_stats['t']
        raw_mx = th_stats.get('raw_max',  np.full_like(t_th, np.nan))
        flt_mx = th_stats.get('filt_max', np.full_like(t_th, np.nan))
        ax_temp.plot(t_th, raw_mx,  color='#ff4444', linewidth=0.9, label='Raw T_max')
        ax_temp.plot(t_th, flt_mx,  color='#44aaff', linewidth=0.9, label='Filt T_max', alpha=0.8)
        ax_temp.axhline(y=22, color='#888888', linestyle=':', alpha=0.5, label='Ambient')
    ax_temp.set_title('T_max in FOV (C)'); ax_temp.set_ylabel('C')
    ax_temp.legend(fontsize=6, loc='upper left')
    ax_temp.set_xlabel('t (s)')

    # ── Gradient timeline ───────────────────────────────────────────────────
    if grad and 't' in grad:
        ax_grad.plot(grad['t'], grad.get('mean_mag', np.zeros(len(grad['t']))),
                     color='#44ddff', linewidth=0.9, label='Mean |grad T|')
    ax_grad.set_title('|grad T| Mean (C/px)'); ax_grad.set_ylabel('C/px')
    ax_grad.legend(fontsize=6); ax_grad.set_xlabel('t (s)')

    # ── Distance to heat sources (SLAM preferred) ───────────────────────────
    traj_data = slam if slam_avail else odom
    x_key = 'x' if slam_avail else 'wx'
    y_key = 'y' if slam_avail else 'wy'
    if traj_data and x_key in traj_data:
        tx_arr = traj_data[x_key]
        ty_arr = traj_data[y_key]
        t_arr  = traj_data.get('t', np.arange(len(tx_arr)))
        colors_src = ['#ff4444', '#ff8844', '#ffcc44']
        for src, c in zip(CONFIG_B_SOURCES, colors_src):
            sx2, sy2 = src['xy']
            dist = np.sqrt((tx_arr - sx2)**2 + (ty_arr - sy2)**2)
            ax_dist.plot(t_arr, dist, color=c, linewidth=0.9, label=src['name'])
    ax_dist.axhline(y=0.5, color='#44ff88', linestyle='--',
                    linewidth=0.8, label='Arrival r=0.5m')
    ax_dist.set_title('Distance to Heat Sources')
    ax_dist.set_xlabel('t (s)'); ax_dist.set_ylabel('m')
    ax_dist.legend(fontsize=6, loc='upper right')

    # ── Velocity commands ───────────────────────────────────────────────────
    if cmdv and 't' in cmdv:
        ct = cmdv['t']
        cv = cmdv.get('lin_x', np.zeros_like(ct))
        cw = cmdv.get('ang_z', np.zeros_like(ct))
        ax_vel.plot(ct, cv,     color='#44ff88', linewidth=0.9, label='v (m/s)')
        ax_vel.plot(ct, cw*0.2, color='#ffaa44', linewidth=0.7, alpha=0.7, label='w*0.2')
    ax_vel.set_title('Velocity Commands')
    ax_vel.set_xlabel('t (s)'); ax_vel.legend(fontsize=6)

    # ── Info box ────────────────────────────────────────────────────────────
    ax_info.axis('off')
    duration     = meta.get('duration_s', 0.0)
    n_plans      = meta.get('nav2_plan_count', 0)
    slam_first   = meta.get('slam_first_t', None)
    slam_cnt     = meta.get('slam_pos_count', 0)
    nav2_avail   = meta.get('nav2_available', False)
    slam_avail_m = meta.get('slam_available', False)

    path_len = 0.0
    if traj_data and x_key in traj_data and len(traj_data[x_key]) > 1:
        dx3 = np.diff(traj_data[x_key])
        dy3 = np.diff(traj_data[y_key])
        path_len = float(np.sqrt(dx3**2 + dy3**2).sum())

    scan_rate = 0.0
    if scan and 't' in scan and len(scan['t']) > 1:
        dt_scan   = np.diff(scan['t'])
        scan_rate = float(1.0 / dt_scan.mean())

    slam_first_str = f'{slam_first:.1f}s' if slam_first is not None else 'N/A'
    info_text = (
        f'SLAM + Nav2 Status\n'
        f'────────────────────\n'
        f'SLAM: {"Available" if slam_avail_m else "Unavailable"}\n'
        f'  First ready: {slam_first_str}\n'
        f'  Poses:       {slam_cnt}\n'
        f'Nav2: {"Available" if nav2_avail else "Unavailable"}\n'
        f'  Plans sent:  {n_plans}\n'
        f'/scan rate:   {scan_rate:.1f} Hz\n'
        f'────────────────────\n'
        f'Path length:  {path_len:.1f} m\n'
        f'Frame: {coord_src[:16]}\n'
    )
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                 fontsize=8, verticalalignment='top',
                 fontfamily='monospace', color='#aaddff',
                 bbox=dict(boxstyle='round', facecolor='#111133', alpha=0.8))

    out_path = out_dir / '05_slam_nav2_dashboard.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print('  ✓ 05_slam_nav2_dashboard.png')
    return out_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        base = Path.home() / 'ros2_ws' / 'bags' / 'collected'
        if not base.exists():
            print(f'[plot_slam_nav2] Error: data directory does not exist: {base}')
            sys.exit(1)
        dirs = sorted(base.iterdir())
        if not dirs:
            print(f'[plot_slam_nav2] Error: no data in {base}')
            sys.exit(1)
        data_dir = dirs[-1]

    if not data_dir.exists():
        print(f'[plot_slam_nav2] Error: {data_dir} does not exist')
        sys.exit(1)

    print(f'[plot_slam_nav2] Reading data: {data_dir}')

    meta = {}
    meta_path = data_dir / 'metadata.json'
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    print('[plot_slam_nav2] Generating figures...')
    fig_slam_vs_odom(data_dir, meta)
    fig_scan_coverage(data_dir, meta)
    fig_nav2_plans(data_dir, meta)
    fig_slam_quality(data_dir, meta)
    fig_slam_nav2_dashboard(data_dir, meta)

    print(f'\n[plot_slam_nav2] Done! 5 figures saved to: {data_dir}')
    print('  01_slam_vs_odom.png')
    print('  02_scan_coverage.png')
    print('  03_nav2_plans.png')
    print('  04_slam_quality.png')
    print('  05_slam_nav2_dashboard.png')


if __name__ == '__main__':
    main()
