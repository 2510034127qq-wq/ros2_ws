#!/usr/bin/env python3
"""
plot_all_figures.py — 热导航仿真全量可视化生成器 v3
====================================================
读取 collect_sim_data.py 收集的结构化数据 + ROS2 日志，
生成适用于中期答辩的 13 张高质量图表。

用法：
  # 指定采集目录（推荐）
  python3 plot_all_figures.py ~/ros2_ws/bags/collected/20250315_143022

  # 自动找最新采集目录
  python3 plot_all_figures.py

  # 仅用日志（无采集数据时的降级模式）
  python3 plot_all_figures.py --log-only

输出目录：<采集目录>/figures/
  01_trajectory.png          — 轨迹 + 热源 + 状态分段
  02_temperature_series.png  — 温度时序（raw/filtered 双轴）
  03_gradient_analysis.png   — 梯度幅值 + 方向分布
  04_convergence.png         — 热源收敛距离
  05_state_timeline.png      — 状态机甘特图
  06_thermal_snapshots.png   — 热图快照对比（raw vs filtered）
  07_gradient_field.png      — 梯度向量场叠加
  08_velocity_profile.png    — 速度指令时序
  09_belief_map.png          — ThermalBeliefMap 重构
  10_topic_rates.png         — 话题频率验证
  11_source_detection.png    — 热源发现时间线
  12_performance_metrics.png — 性能雷达图
  13_dashboard.png           — 综合面板（答辩主图）

依赖：numpy, matplotlib, scipy (可选，无 scipy 自动降级)
"""

import csv
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except Exception:          # 兼容 NumPy 2.x 下 scipy ABI 不匹配
    HAS_SCIPY = False
    def gaussian_filter(arr, sigma=1.0):
        """scipy 不可用时的简易均值滤波降级实现。"""
        import numpy as _np
        from numpy.lib.stride_tricks import sliding_window_view
        k = max(1, int(sigma * 2) | 1)   # 奇数窗口
        pad = k // 2
        padded = _np.pad(arr, pad, mode='edge')
        # 2D 均值滤波
        out = _np.zeros_like(arr, dtype=_np.float32)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                out[i, j] = padded[i:i+k, j:j+k].mean()
        return out

# ─── 全局常量（与 sensor_node.py 保持一致）──────────────────────────────────
# ── Config-B: 通用性验证配置（与 sensor_node.py v13 保持一致）──────────────
SOURCES = [
    {'name': 'SA_left',  'xy': (-1.0,  3.5), 'amp': 35.0, 'sigma': 1.1,
     'color': '#e74c3c', 'peak': 57.0},
    {'name': 'SB_far',   'xy': ( 6.0, -3.0), 'amp': 22.0, 'sigma': 0.9,
     'color': '#e67e22', 'peak': 44.0},
    {'name': 'SC_weak',  'xy': (-5.0, -5.5), 'amp': 16.0, 'sigma': 0.8,
     'color': '#f1c40f', 'peak': 38.0},
]
SPAWN_X, SPAWN_Y = -6.0, 0.0
AMBIENT_T   = 22.0
ARRIVAL_R   = 0.5
SNAPSHOT_INTERVAL_S = 5.0   # 与 collect_sim_data.py 保持一致
STATE_COLORS = {
    'ASCENT':       '#00d4ff',
    'CONVERGE':     '#9b59b6',
    'SAMPLE':       '#2ecc71',
    'AT_PEAK':      '#27ae60',
    'FRONTIER_NAV': '#888888',
    'RELOCATE':     '#e67e22',
    'ESCAPE':       '#e74c3c',
    'DONE':         '#1abc9c',
}
DARK_BG  = '#0d0d1a'
DARK_AX  = '#111122'
GRID_C   = '#2a2a3a'
TEXT_C   = '#e8e8e8'
ACCENT   = '#00d4ff'
THERMAL_CMAP = LinearSegmentedColormap.from_list(
    'thermal', ['#000080', '#0000ff', '#00ffff', '#00ff00',
                '#ffff00', '#ff8000', '#ff0000', '#ffffff'])


# ═══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════════════

def find_latest_collection() -> Optional[Path]:
    base = Path.home() / 'ros2_ws' / 'bags' / 'collected'
    if not base.exists():
        return None
    dirs = [d for d in base.iterdir() if d.is_dir()]
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


def load_csv(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # 自动转换数值
            converted = {}
            for k, v in row.items():
                if v in ('', 'None', 'nan', 'null'):
                    converted[k] = None
                else:
                    try:
                        converted[k] = float(v)
                    except ValueError:
                        converted[k] = v
            rows.append(converted)
    return rows


def load_collection(data_dir: Path) -> Dict:
    meta_path = data_dir / 'metadata.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    traj   = load_csv(data_dir / 'trajectory.csv')
    th     = load_csv(data_dir / 'thermal_stats.csv')
    field  = load_csv(data_dir / 'field_stats.csv')
    grad   = load_csv(data_dir / 'gradient_stats.csv')
    cmd    = load_csv(data_dir / 'cmd_vel.csv')

    # 快照索引
    snap_index = []
    snap_idx_path = data_dir / 'snapshots' / 'index.jsonl'
    if snap_idx_path.exists():
        for line in snap_idx_path.read_text().splitlines():
            try:
                snap_index.append(json.loads(line))
            except Exception:
                pass

    return {
        'meta': meta, 'traj': traj, 'thermal': th,
        'field': field, 'grad': grad, 'cmd': cmd,
        'snap_index': snap_index, 'data_dir': data_dir,
    }


# ─── ROS2 日志解析 ───────────────────────────────────────────────────────────

def find_latest_ros_session() -> Optional[Path]:
    log_root = Path.home() / '.ros' / 'log'
    if not log_root.exists():
        return None
    sessions = [p for p in log_root.iterdir()
                if p.is_dir() and not p.name.startswith('latest')]
    return max(sessions, key=lambda p: p.stat().st_mtime) if sessions else None


def read_session_text(session_dir: Path) -> str:
    text = ''
    for f in session_dir.rglob('*'):
        if f.is_file() and f.suffix in ('', '.log', '.txt'):
            try:
                text += f.read_text(errors='ignore') + '\n'
            except Exception:
                pass
    return text


def parse_controller_logs(text: str) -> List[Dict]:
    """解析 controller_node v20 格式日志，兼容 v7–v20 所有格式。"""
    records = []

    # v20 格式（最新）：[STATE] t=Xs pos=(x,y) T=X°C rise=X°C gm=X found=X/Y path=Xm
    pat_v20 = re.compile(
        r'\[(ASCENT|CONVERGE|SAMPLE|AT_PEAK|FRONTIER_NAV|RELOCATE|ESCAPE|DONE)\]'
        r'\s+t=(\d+)s\s+pos=\(([+-]?\d+\.?\d*),([+-]?\d+\.?\d*)\)'
        r'\s+T=([+-]?\d+\.?\d*).C\s+rise=([+-]?\d+\.?\d*).C'
        r'\s+gm=([+-]?\d+\.?\d*)\s+found=(\d+)/\S+\s+path=([+-]?\d+\.?\d*)m')
    for m in pat_v20.finditer(text):
        records.append({
            'state':   m.group(1),
            't':       float(m.group(2)),
            'wx':      float(m.group(3)),
            'wy':      float(m.group(4)),
            'T':       float(m.group(5)),
            'rise':    float(m.group(6)),
            'gm':      float(m.group(7)),
            'found':   int(m.group(8)),
            'path':    float(m.group(9)),
        })

    # v8-v10 格式：[STATE] t=Xs world=(x,y) T_fov=X°C T_win_max=X°C plateau=X°C grad=X
    pat_v10 = re.compile(
        r'\[(ASCENT|SEARCH|RELOCATE|NAVIGATE|FRONTIER_NAV)\]'
        r'\s+t=(\d+)s\s+world=\(([+-]?\d+\.?\d*),([+-]?\d+\.?\d*)\)'
        r'\s+T(?:_fov)?=([+-]?\d+\.?\d*)..C\s+(?:T_win_max=([+-]?\d+\.?\d*)..C\s+)?'
        r'(?:plateau=([+-]?\d+\.?\d*)..C\s+)?grad=([+-]?\d+\.?\d*)')
    for m in pat_v10.finditer(text):
        if not any(r['t'] == float(m.group(2)) and r['state'] == m.group(1)
                   for r in records):
            records.append({
                'state':   m.group(1),
                't':       float(m.group(2)),
                'wx':      float(m.group(3)),
                'wy':      float(m.group(4)),
                'T':       float(m.group(5)),
                'rise':    0.0,
                'gm':      float(m.group(8)),
                'found':   0,
                'path':    0.0,
            })

    # SOURCE FOUND 事件（兼容 v20/v21/v22 格式）
    found_events = []
    # v22 格式：★ [SOURCE #N] pos=(x,y) T=X°C path=Xm t=Xs
    for m in re.finditer(
            r'SOURCE #(\d+)\].*?'
            r'pos=\(([+-]?\d+\.?\d*),([+-]?\d+\.?\d*)\)'
            r'.*?T=([+-]?\d+\.?\d*).*?t=([+-]?\d+\.?\d*)s', text):
        found_events.append({
            'idx':  int(m.group(1)),
            'wx':   float(m.group(2)),
            'wy':   float(m.group(3)),
            'T':    float(m.group(4)),
            't':    float(m.group(5)),
        })
    # v8-v19 备用格式
    for m in re.finditer(
            r'SOURCE FOUND #(\d+).*?'
            r'(?:world|pos)=\(([+-]?\d+\.?\d*),([+-]?\d+\.?\d*)\)'
            r'.*?[Tt]=([+-]?\d+\.?\d*).*?[Tt](?:ime)?=([+-]?\d+\.?\d*)s', text):
        t_val = float(m.group(5))
        if not any(abs(e['t'] - t_val) < 2.0 for e in found_events):
            found_events.append({
                'idx':  int(m.group(1)),
                'wx':   float(m.group(2)),
                'wy':   float(m.group(3)),
                'T':    float(m.group(4)),
                't':    t_val,
            })
    # AT_PEAK 确认事件（v22 SAMPLE_DONE_v22 后）
    for m in re.finditer(
            r'SAMPLE_DONE_v22.*?CONFIRM.*?\n.*?\[AT_PEAK\].*?t=([+-]?\d+\.?\d*)s',
            text, re.DOTALL):
        pass  # 已通过 SOURCE # 捕获
    # 备用：GOAL REACHED
    for m in re.finditer(
            r'(?:GOAL REACHED|ARRIVED).*?t=([+-]?\d+\.?\d*)s', text):
        t_val = float(m.group(1))
        if not any(abs(e['t'] - t_val) < 2.0 for e in found_events):
            found_events.append({'idx': len(found_events), 'wx': 0, 'wy': 0,
                                  'T': 0, 't': t_val})

    records.sort(key=lambda r: r['t'])
    return records, found_events


# ─── 坐标转换（odom → world）──────────────────────────────────────────────────

def odom_to_world(traj: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not traj:
        return np.array([]), np.array([]), np.array([]), np.array([])
    ts  = np.array([r['t'] for r in traj])
    xs  = np.array([SPAWN_X + r['x'] for r in traj])
    ys  = np.array([SPAWN_Y + r['y'] for r in traj])
    yaw = np.array([r['yaw'] for r in traj])
    return ts, xs, ys, yaw


def traj_from_logs(log_recs: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not log_recs:
        return np.array([]), np.array([]), np.array([])
    ts  = np.array([r['t']  for r in log_recs])
    wxs = np.array([r['wx'] for r in log_recs])
    wys = np.array([r['wy'] for r in log_recs])
    return ts, wxs, wys


# ─── 地图背景：高斯热场 ────────────────────────────────────────────────────────

def make_thermal_bg(extent=(-10, 10, -8, 8), res=200):
    xmin, xmax, ymin, ymax = extent
    xs = np.linspace(xmin, xmax, res)
    ys = np.linspace(ymin, ymax, res)
    XX, YY = np.meshgrid(xs, ys)
    field = np.full_like(XX, AMBIENT_T)
    for s in SOURCES:
        sx, sy = s['xy']
        field += s['amp'] * np.exp(
            -((XX - sx)**2 + (YY - sy)**2) / (2 * s['sigma']**2))
    return XX, YY, field


# ═══════════════════════════════════════════════════════════════════════════════
# 样式辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(DARK_AX)
    for sp in ax.spines.values():
        sp.set_color('#3a3a4a')
    ax.tick_params(colors=TEXT_C, which='both')
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)
    ax.title.set_color(TEXT_C)
    ax.grid(color=GRID_C, alpha=0.6, linewidth=0.7)
    if title:   ax.set_title(title, fontsize=11, fontweight='bold', color=TEXT_C)
    if xlabel:  ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:  ax.set_ylabel(ylabel, fontsize=9)


def dark_fig(figsize=(12, 6)):
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)
    return fig


def source_legend_handles():
    return [Line2D([0], [0], marker='*', color='w', markerfacecolor=s['color'],
                   markersize=12, lw=0, label=s['name']) for s in SOURCES]


# ═══════════════════════════════════════════════════════════════════════════════
# 图 01 — 轨迹图
# ═══════════════════════════════════════════════════════════════════════════════

def plot_trajectory(data: Dict, log_recs: List[Dict], found_events: List[Dict], out: Path):
    fig = dark_fig((13, 9))
    ax = fig.add_subplot(111)
    ax.set_facecolor(DARK_AX)

    # 背景热场
    XX, YY, bg = make_thermal_bg()
    cf = ax.contourf(XX, YY, bg, levels=30, cmap=THERMAL_CMAP, alpha=0.35)

    # 等温线
    cs = ax.contour(XX, YY, bg, levels=[30, 40, 50, 60],
                    colors=['cyan'], linewidths=0.6, alpha=0.4)
    ax.clabel(cs, fmt='%d°C', fontsize=7, colors='cyan')

    # 轨迹（优先使用 /odom 数据）
    plotted = False
    if data['traj']:
        ts, wxs, wys, _ = odom_to_world(data['traj'])
        if len(ts) > 2:
            # 按时间着色
            sc = ax.scatter(wxs, wys, c=ts, cmap='cool', s=3, alpha=0.7,
                            zorder=4, linewidths=0)
            cb = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
            cb.set_label('Time (s)', color=TEXT_C, fontsize=9)
            cb.ax.yaxis.set_tick_params(color=TEXT_C)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_C)
            ax.plot(wxs, wys, '-', color='white', lw=0.8, alpha=0.35, zorder=3)
            ax.plot(wxs[0], wys[0], '^', color='#2ecc71', ms=14, zorder=8,
                    markeredgecolor='white', markeredgewidth=1.5, label='Start')
            ax.plot(wxs[-1], wys[-1], 's', color='#9b59b6', ms=12, zorder=8,
                    markeredgecolor='white', markeredgewidth=1.5, label='End')
            plotted = True

    # 降级到日志轨迹
    if not plotted and log_recs:
        ts, wxs, wys = traj_from_logs(log_recs)
        if len(ts) > 2:
            sc = ax.scatter(wxs, wys, c=ts, cmap='cool', s=8, alpha=0.8,
                            zorder=4, linewidths=0)
            ax.plot(wxs, wys, '-', color='white', lw=1.0, alpha=0.4, zorder=3)
            ax.plot(wxs[0], wys[0], '^', color='#2ecc71', ms=14, zorder=8,
                    markeredgecolor='white', markeredgewidth=1.5, label='Start')

    # 热源
    for s in SOURCES:
        sx, sy = s['xy']
        ax.plot(sx, sy, '*', color=s['color'], ms=20, zorder=9,
                markeredgecolor='white', markeredgewidth=1.0)
        ax.add_patch(plt.Circle((sx, sy), ARRIVAL_R, color=s['color'],
                                fill=False, lw=2, ls='--', alpha=0.9, zorder=8))
        ax.annotate(f"{s['name']}\n{s['peak']}°C", (sx, sy),
                    xytext=(10, 10), textcoords='offset points',
                    color=s['color'], fontsize=9, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    # 发现事件标记
    for ev in found_events:
        ax.plot(ev['wx'], ev['wy'], 'D', color='lime', ms=12, zorder=10,
                markeredgecolor='white', markeredgewidth=1.5)
        ax.annotate(f"Found t={ev['t']:.0f}s", (ev['wx'], ev['wy']),
                    xytext=(6, -14), textcoords='offset points',
                    color='lime', fontsize=8,
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    # 起点标记
    ax.plot(SPAWN_X, SPAWN_Y, '^', color='#2ecc71', ms=14, zorder=9,
            markeredgecolor='white', markeredgewidth=1.5)

    style_ax(ax, 'Thermal Navigation Trajectory\n(Background: Ground-truth Thermal Field)',
             'World X (m)', 'World Y (m)')
    ax.set_aspect('equal')

    handles = [
        Line2D([0],[0], marker='^', color='w', mfc='#2ecc71', ms=10, lw=0, label='Start'),
        Line2D([0],[0], marker='D', color='w', mfc='lime',    ms=10, lw=0, label='Source Found'),
        Line2D([0],[0], marker='*', color='w', mfc='#e74c3c', ms=14, lw=0, label='Heat Source'),
    ] + source_legend_handles()
    ax.legend(handles=handles, loc='lower right', fontsize=9,
              facecolor='#1a1a2e', labelcolor='white', framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out / '01_trajectory.png', dpi=160, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.close(fig)
    print('  ✓ 01_trajectory.png')


# ═══════════════════════════════════════════════════════════════════════════════
# 图 02 — 温度时序
# ═══════════════════════════════════════════════════════════════════════════════

def plot_temperature_series(data: Dict, log_recs: List[Dict],
                            found_events: List[Dict], out: Path):
    fig = dark_fig((13, 7))
    gs = GridSpec(2, 1, figure=fig, hspace=0.08, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax1.set_facecolor(DARK_AX); ax2.set_facecolor(DARK_AX)

    th = data['thermal']
    if th:
        ts_r  = np.array([r['t']        for r in th if r.get('t')        is not None])
        raw_m = np.array([r.get('raw_max',  22) or 22 for r in th if r.get('t') is not None])
        raw_n = np.array([r.get('raw_mean', 22) or 22 for r in th if r.get('t') is not None])
        filt_m= np.array([r.get('filt_max', None) for r in th if r.get('t') is not None],
                         dtype=object)

        ax1.plot(ts_r, raw_m,  color='#e74c3c', lw=2.0, label='Raw T_max',  alpha=0.9)
        ax1.plot(ts_r, raw_n,  color='#ff8888', lw=1.2, label='Raw T_mean', alpha=0.7, ls='--')
        # filtered（仅有数据的部分）
        valid = np.array([v is not None for v in filt_m])
        if valid.any():
            ts_f = ts_r[valid]
            fm   = np.array(filt_m[valid], dtype=float)
            ax1.plot(ts_f, fm, color='#3498db', lw=2.0, label='Filtered T_max', alpha=0.9)

        ax1.axhline(AMBIENT_T, color='gray', ls=':', lw=1.0, alpha=0.6,
                    label=f'Ambient {AMBIENT_T}°C')
        ax1.fill_between(ts_r, AMBIENT_T, raw_m, alpha=0.10, color='#e74c3c')
        ax1.set_ylim(AMBIENT_T - 3, raw_m.max() + 8)

    elif log_recs:
        ts_l  = np.array([r['t'] for r in log_recs])
        T_l   = np.array([r['T'] for r in log_recs])
        ax1.plot(ts_l, T_l, color='#e74c3c', lw=2, label='T_fov (log)')
        ax1.axhline(AMBIENT_T, color='gray', ls=':', lw=1.0)

    # 热源发现事件竖线
    for ev in found_events:
        ax1.axvline(ev['t'], color='lime', ls='--', lw=1.5, alpha=0.8)
        ax1.text(ev['t'] + 1, ax1.get_ylim()[0] + 2,
                 f"✓S{ev['idx']}", color='lime', fontsize=8)

    style_ax(ax1, 'Thermal Camera Temperature over Time', '', 'Temperature (°C)')
    ax1.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white', loc='upper left')
    plt.setp(ax1.get_xticklabels(), visible=False)

    # 下面板：T_max 滑动标准差（信号质量）
    if th and len(ts_r) > 20:
        win = 20
        std_arr = np.convolve(raw_m, np.ones(win)/win, mode='valid')
        ts_std  = ts_r[win-1:]
        rolling_std = np.array([raw_m[max(0,i-win):i].std() for i in range(win, len(raw_m))])
        ax2.plot(ts_r[win:], rolling_std, color='#9b59b6', lw=1.5,
                 label=f'T_max rolling std (win={win})')
        ax2.fill_between(ts_r[win:], 0, rolling_std, alpha=0.2, color='#9b59b6')
        style_ax(ax2, '', 'Time (s)', 'Std Dev (°C)')
        ax2.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white')
    else:
        style_ax(ax2, '', 'Time (s)', '')
        ax2.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                 color='gray', transform=ax2.transAxes)

    for sp in ax1.spines.values(): sp.set_color('#3a3a4a')
    for sp in ax2.spines.values(): sp.set_color('#3a3a4a')
    fig.savefig(out / '02_temperature_series.png', dpi=160, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.close(fig)
    print('  ✓ 02_temperature_series.png')


# ═══════════════════════════════════════════════════════════════════════════════
# 图 03 — 梯度分析
# ═══════════════════════════════════════════════════════════════════════════════

def plot_gradient_analysis(data: Dict, log_recs: List[Dict],
                           found_events: List[Dict], out: Path):
    fig = dark_fig((13, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for ax in axs: ax.set_facecolor(DARK_AX)

    grad = data['grad']

    # ── 上左：梯度幅值时序
    ax = axs[0]
    if grad:
        tg  = np.array([r['t']        for r in grad if r.get('t') is not None])
        mag = np.array([r.get('mean_mag', 0) or 0 for r in grad if r.get('t') is not None])
        mxm = np.array([r.get('max_mag',  0) or 0 for r in grad if r.get('t') is not None])
        pt  = np.array([r.get('peak_t',   0) or 0 for r in grad if r.get('t') is not None])
        ax.plot(tg, mag, color='#3498db', lw=2, label='Mean |∇T|')
        ax.plot(tg, mxm, color='#85c1e9', lw=1.2, ls='--', alpha=0.7, label='Max |∇T|')
        ax.fill_between(tg, 0, mag, alpha=0.15, color='#3498db')
        ax.axhline(0.3, color='#e74c3c', ls=':', lw=1.0, alpha=0.8,
                   label='min_gradient_mag=0.3')
        for ev in found_events:
            ax.axvline(ev['t'], color='lime', ls='--', lw=1.2, alpha=0.7)
        style_ax(ax, 'Gradient Magnitude vs Time', 'Time (s)', '°C/pixel')
        ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
    elif log_recs:
        tg  = np.array([r['t']  for r in log_recs])
        mag = np.array([r['gm'] for r in log_recs])
        ax.plot(tg, mag, color='#3498db', lw=2, label='Gradient mag (log)')
        style_ax(ax, 'Gradient Magnitude vs Time', 'Time (s)', '°C/pixel')
        ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
    else:
        style_ax(ax, 'Gradient Magnitude vs Time')
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                color='gray', transform=ax.transAxes)

    # ── 上右：梯度方向极坐标分布
    ax_polar = fig.add_subplot(gs[0, 1], polar=True)
    ax_polar.set_facecolor(DARK_AX)
    if grad:
        angles = np.array([r.get('dominant_angle', 0) or 0 for r in grad
                           if r.get('t') is not None])
        mags_w = np.array([r.get('mean_mag', 0) or 0 for r in grad
                           if r.get('t') is not None])
        # 加权方向直方图
        bins = np.linspace(-math.pi, math.pi, 37)
        hist, edges = np.histogram(angles, bins=bins, weights=mags_w)
        centers = (edges[:-1] + edges[1:]) / 2
        width   = edges[1] - edges[0]
        bars = ax_polar.bar(centers, hist, width=width, alpha=0.75,
                            color=plt.cm.cool(hist / (hist.max() + 1e-9)))
        ax_polar.set_title('Gradient Direction\n(magnitude-weighted)', color=TEXT_C,
                           fontsize=10, fontweight='bold', pad=12)
        ax_polar.tick_params(colors=TEXT_C)
        ax_polar.grid(color=GRID_C, alpha=0.5)
    else:
        ax_polar.set_title('Gradient Direction', color=TEXT_C, fontsize=10)
        ax_polar.text(0, 0, 'No data', ha='center', va='center', color='gray')
    # 移除 axs[1] 的原始子图（已被极坐标替换）

    # ── 下左：峰值温度 vs 梯度幅值散点
    ax = axs[2]
    if grad:
        pt  = np.array([r.get('peak_t',   22) or 22 for r in grad if r.get('t') is not None])
        mag = np.array([r.get('mean_mag', 0) or 0 for r in grad if r.get('t') is not None])
        sc  = ax.scatter(pt, mag, c=tg, cmap='plasma', s=15, alpha=0.6,
                         linewidths=0)
        cb  = plt.colorbar(sc, ax=ax)
        cb.set_label('Time (s)', color=TEXT_C, fontsize=8)
        cb.ax.yaxis.set_tick_params(color=TEXT_C)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_C)
        # 添加趋势线
        if len(pt) > 5:
            z = np.polyfit(pt, mag, 1)
            p = np.poly1d(z)
            xs = np.linspace(pt.min(), pt.max(), 100)
            ax.plot(xs, p(xs), '--', color='cyan', lw=1.5, alpha=0.8, label='Linear fit')
            ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        style_ax(ax, 'Peak Temperature vs Gradient Magnitude',
                 'Peak Temp in FOV (°C)', 'Mean |∇T| (°C/px)')
    else:
        style_ax(ax, 'Peak Temp vs Gradient')
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                color='gray', transform=ax.transAxes)

    # ── 下右：梯度幅值箱线图（按状态分组）
    ax = axs[3]
    if log_recs and grad:
        states_all = sorted(set(r['state'] for r in log_recs))
        tg_arr = np.array([r['t'] for r in grad if r.get('t') is not None])
        tl_arr = np.array([r['t'] for r in log_recs])
        st_arr = np.array([r['state'] for r in log_recs])
        mag_arr= np.array([r.get('mean_mag', 0) or 0 for r in grad
                           if r.get('t') is not None])
        # 为每个梯度时刻找最近的状态
        state_mags = {s: [] for s in states_all}
        for i, tg_i in enumerate(tg_arr):
            idx = np.argmin(np.abs(tl_arr - tg_i))
            s   = st_arr[idx]
            state_mags[s].append(mag_arr[i])
        valid_states = [s for s in states_all if len(state_mags[s]) > 1]
        if valid_states:
            bp = ax.boxplot([state_mags[s] for s in valid_states],
                            patch_artist=True, notch=False)
            for patch, s in zip(bp['boxes'], valid_states):
                patch.set_facecolor(STATE_COLORS.get(s, '#888888'))
                patch.set_alpha(0.7)
            for element in ['whiskers','caps','fliers']:
                for item in bp[element]:
                    item.set_color(TEXT_C)
            for median in bp['medians']:
                median.set_color('white')
            ax.set_xticks(range(1, len(valid_states)+1))
            ax.set_xticklabels(valid_states, rotation=25, ha='right',
                               fontsize=8, color=TEXT_C)
    style_ax(ax, '|∇T| by Navigation State', 'State', '°C/pixel')

    fig.savefig(out / '03_gradient_analysis.png', dpi=160, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.close(fig)
    print('  ✓ 03_gradient_analysis.png')


# ═══════════════════════════════════════════════════════════════════════════════
# 图 04 — 收敛距离
# ═══════════════════════════════════════════════════════════════════════════════

def plot_convergence(data: Dict, log_recs: List[Dict],
                     found_events: List[Dict], out: Path):
    fig = dark_fig((13, 6))
    ax = fig.add_subplot(111)
    ax.set_facecolor(DARK_AX)

    # 获取轨迹
    if data['traj']:
        ts, wxs, wys, _ = odom_to_world(data['traj'])
    elif log_recs:
        ts, wxs, wys = traj_from_logs(log_recs)
    else:
        style_ax(ax, 'Distance to Heat Sources')
        ax.text(0.5, 0.5, 'No trajectory data', ha='center', va='center',
                color='gray', transform=ax.transAxes)
        fig.savefig(out / '04_convergence.png', dpi=160, bbox_inches='tight',
                    facecolor=DARK_BG)
        plt.close(fig)
        print('  ✓ 04_convergence.png (no data)')
        return

    if len(ts) < 2:
        plt.close(fig)
        print('  [SKIP] 04_convergence.png (too few points)')
        return

    for s in SOURCES:
        sx, sy = s['xy']
        dist = np.sqrt((wxs - sx)**2 + (wys - sy)**2)
        ax.plot(ts, dist, color=s['color'], lw=2.5,
                label=f"{s['name']} (peak={s['peak']}°C)")
        # 到达时刻
        arrivals = np.where(dist < ARRIVAL_R)[0]
        if len(arrivals):
            ax.plot(ts[arrivals[0]], dist[arrivals[0]], 'v',
                    color=s['color'], ms=13, markeredgecolor='white',
                    markeredgewidth=1.5, zorder=6)
            ax.annotate(f"✓ {s['name']}", (ts[arrivals[0]], dist[arrivals[0]]),
                        xytext=(5, 5), textcoords='offset points',
                        color=s['color'], fontsize=9,
                        path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    ax.axhline(ARRIVAL_R, color='lime', ls='--', lw=2.0,
               label=f'arrival_radius = {ARRIVAL_R} m')
    ax.fill_between(ts, 0, ARRIVAL_R, alpha=0.08, color='lime')

    for ev in found_events:
        ax.axvline(ev['t'], color='white', ls=':', lw=1.0, alpha=0.5)

    style_ax(ax, 'Distance to Heat Sources — Convergence Analysis',
             'Time (s)', 'Euclidean Distance (m)')
    ax.legend(fontsize=10, facecolor='#1a1a2e', labelcolor='white')
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out / '04_convergence.png', dpi=160, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.close(fig)
    print('  ✓ 04_convergence.png')


# ═══════════════════════════════════════════════════════════════════════════════
# 图 05 — 状态机甘特图
# ═══════════════════════════════════════════════════════════════════════════════

def plot_state_timeline(log_recs: List[Dict], found_events: List[Dict], out: Path):
    fig = dark_fig((13, 5))
    ax = fig.add_subplot(111)
    ax.set_facecolor(DARK_AX)

    if not log_recs:
        style_ax(ax, 'State Machine Timeline (No log data available)')
        ax.text(0.5, 0.5, 'No controller log data.\nRun simulation to generate logs.',
                ha='center', va='center', color='gray', fontsize=12,
                transform=ax.transAxes)
        fig.savefig(out / '05_state_timeline.png', dpi=160, bbox_inches='tight',
                    facecolor=DARK_BG)
        plt.close(fig)
        print('  ✓ 05_state_timeline.png (no data)')
        return

    all_states = list(dict.fromkeys(r['state'] for r in log_recs))
    state_to_y = {s: i for i, s in enumerate(all_states)}
    n_states   = len(all_states)

    # 构建区间
    transitions = []
    for i, rec in enumerate(log_recs[:-1]):
        t0 = rec['t']
        t1 = log_recs[i+1]['t']
        transitions.append((rec['state'], t0, t1))
    if log_recs:
        transitions.append((log_recs[-1]['state'], log_recs[-1]['t'],
                            log_recs[-1]['t'] + 5.0))

    # 绘制甘特条
    for state, t0, t1 in transitions:
        y = state_to_y.get(state, 0)
        color = STATE_COLORS.get(state, '#888888')
        ax.barh(y, t1 - t0, left=t0, height=0.7, color=color, alpha=0.85,
                edgecolor='none')

    # 发现事件
    for ev in found_events:
        ax.axvline(ev['t'], color='lime', lw=2.5, ls='--', zorder=5)
        ax.text(ev['t'] + 1, n_states - 0.3, f"S{ev['idx']} Found",
                color='lime', fontsize=8, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    ax.set_yticks(range(n_states))
    ax.set_yticklabels(all_states, fontsize=10, color=TEXT_C)
    style_ax(ax, 'Navigation State Machine Timeline (Gantt)',
             'Time (s)', 'State')
    ax.set_xlim(left=0)

    # 图例
    handles = [mpatches.Patch(color=STATE_COLORS.get(s, '#888888'),
               label=s, alpha=0.85) for s in all_states]
    ax.legend(handles=handles, loc='lower right', fontsize=8,
              facecolor='#1a1a2e', labelcolor='white', framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out / '05_state_timeline.png', dpi=160, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.close(fig)
    print('  ✓ 05_state_timeline.png')


# ═══════════════════════════════════════════════════════════════════════════════
# 图 06 — 热图快照对比
# ═══════════════════════════════════════════════════════════════════════════════

def plot_thermal_snapshots(data: Dict, out: Path):
    snap_dir = data['data_dir'] / 'snapshots'
    snap_idx = data['snap_index']

    # 最多选6个等间距快照
    if not snap_idx:
        # 生成合成数据用于演示
        print('  [INFO] 无快照数据，生成仿真热场可视化...')
        _plot_synthetic_thermal(out)
        return

    n_snap = len(snap_idx)
    sel_idx = np.linspace(0, n_snap - 1, min(6, n_snap), dtype=int)
    selected = [snap_idx[i] for i in sel_idx]

    n_rows = 2
    n_cols = len(selected)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 7))
    fig.patch.set_facecolor(DARK_BG)

    if n_cols == 1:
        axes = [[axes[0]], [axes[1]]]

    for col, snap in enumerate(selected):
        idx  = snap['idx']
        t_s  = snap.get('t', idx * SNAPSHOT_INTERVAL_S)
        raw_path  = snap_dir / f'raw_{idx:04d}.npy'
        filt_path = snap_dir / f'filt_{idx:04d}.npy'

        for row, (arr_path, label) in enumerate(
                [(raw_path, 'Raw'), (filt_path, 'Filtered')]):
            ax = axes[row][col]
            ax.set_facecolor('#050510')
            if arr_path.exists():
                arr = np.load(arr_path)
                im  = ax.imshow(arr, cmap=THERMAL_CMAP, aspect='auto',
                                vmin=AMBIENT_T, vmax=arr.max())
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                             ).ax.yaxis.set_tick_params(color=TEXT_C)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        color='gray', transform=ax.transAxes)
            ax.set_title(f'{label}\nt={t_s:.0f}s', fontsize=9,
                         color=TEXT_C, fontweight='bold')
            ax.tick_params(colors=TEXT_C)
            for sp in ax.spines.values(): sp.set_color('#3a3a4a')

    fig.suptitle('Thermal Camera Snapshots: Raw vs Filtered\n'
                 '(Temporal sampling at equal intervals)',
                 color=TEXT_C, fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(out / '06_thermal_snapshots.png', dpi=160, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.close(fig)
    print('  ✓ 06_thermal_snapshots.png')


def _plot_synthetic_thermal(out: Path):
    """当无快照数据时，绘制仿真热场的合成可视化（展示系统能力）。"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.patch.set_facecolor(DARK_BG)
    titles = ['t=20s (near spawn)', 't=80s (searching)', 't=160s (approaching S0)',
              't=240s (S0 found)',  't=320s (searching S1)', 't=400s (near S1)']
    robot_positions = [(-5.5, 0.1), (-3.0, 1.5), (-1.5, 0.5),
                       (0.2,  0.3), (2.0,  1.0), (4.5, 2.2)]
    sensor_fov_x, sensor_fov_y = 4.0, 3.0

    for idx, (ax, title, rpos) in enumerate(zip(axes.flat, titles, robot_positions)):
        ax.set_facecolor('#050510')
        rx, ry = rpos
        # FOV 区域内的热场
        xs = np.linspace(rx - sensor_fov_x/2, rx + sensor_fov_x/2, 64)
        ys = np.linspace(ry - sensor_fov_y/2, ry + sensor_fov_y/2, 48)
        XX, YY = np.meshgrid(xs, ys)
        arr = np.full_like(XX, AMBIENT_T)
        for s in SOURCES:
            sx, sy = s['xy']
            arr += s['amp'] * np.exp(-((XX-sx)**2 + (YY-sy)**2) / (2*s['sigma']**2))
        # 添加噪声
        arr += np.random.default_rng(idx).normal(0, 0.5, arr.shape)
        im = ax.imshow(arr, cmap=THERMAL_CMAP, aspect='auto', origin='lower',
                       vmin=AMBIENT_T, extent=[xs[0], xs[-1], ys[0], ys[-1]])
        plt.colorbar(im, ax=ax, fraction=0.046
                     ).ax.yaxis.set_tick_params(color=TEXT_C)
        ax.set_title(title, fontsize=9, color=TEXT_C, fontweight='bold')
        ax.tick_params(colors=TEXT_C)
        for sp in ax.spines.values(): sp.set_color('#3a3a4a')

    fig.suptitle('Thermal FOV Snapshots (Simulated — Raw Field)',
                 color=TEXT_C, fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out / '06_thermal_snapshots.png', dpi=160, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.close(fig)
    print('  ✓ 06_thermal_snapshots.png (synthetic)')


# ═══════════════════════════════════════════════════════════════════════════════
# 图 07 — 梯度向量场
# ═══════════════════════════════════════════════════════════════════════════════

def plot_gradient_field(data: Dict, log_recs: List[Dict], out: Path):
    fig = dark_fig((13, 8))
    ax = fig.add_subplot(111)
    ax.set_facecolor(DARK_AX)

    # 背景热场
    XX, YY, bg = make_thermal_bg()
    cf = ax.contourf(XX, YY, bg, levels=25, cmap=THERMAL_CMAP, alpha=0.55)
    cb = plt.colorbar(cf, ax=ax, shrink=0.7)
    cb.set_label('Temperature (°C)', color=TEXT_C, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=TEXT_C)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_C)

    # 真实梯度场（解析计算）
    xv = np.linspace(-9, 9, 20)
    yv = np.linspace(-7, 7, 16)
    Xq, Yq = np.meshgrid(xv, yv)
    Gx = np.zeros_like(Xq)
    Gy = np.zeros_like(Yq)
    for s in SOURCES:
        sx, sy = s['xy']
        dx = Xq - sx; dy = Yq - sy
        coeff = s['amp'] / (s['sigma']**2) * np.exp(
            -(dx**2 + dy**2) / (2*s['sigma']**2))
        Gx -= coeff * dx   # ∂T/∂x （负号是因为梯度指向热源）
        Gy -= coeff * dy
    Gx = -Gx; Gy = -Gy  # 修正为指向热源方向（上升方向）
    mag = np.sqrt(Gx**2 + Gy**2) + 1e-9
    Gx /= mag; Gy /= mag  # 归一化
    q = ax.quiver(Xq, Yq, Gx, Gy,
                  np.sqrt(Gx**2 + Gy**2) * mag,
                  cmap='cool', alpha=0.7, scale=30, width=0.003,
                  headwidth=4, headlength=5)

    # 机器人轨迹
    if data['traj']:
        ts, wxs, wys, _ = odom_to_world(data['traj'])
        if len(ts) > 1:
            ax.plot(wxs, wys, '-', color='white', lw=1.5, alpha=0.6, zorder=5)
            ax.plot(wxs[0], wys[0], '^', color='#2ecc71', ms=12, zorder=8,
                    markeredgecolor='white', markeredgewidth=1.5)
    elif log_recs:
        ts, wxs, wys = traj_from_logs(log_recs)
        if len(ts) > 1:
            ax.plot(wxs, wys, '-', color='white', lw=1.5, alpha=0.6, zorder=5)

    # 热源
    for s in SOURCES:
        sx, sy = s['xy']
        ax.plot(sx, sy, '*', color=s['color'], ms=18, zorder=9,
                markeredgecolor='white', markeredgewidth=1.0)
        ax.add_patch(plt.Circle((sx, sy), s['sigma'], color=s['color'],
                                fill=False, lw=1.5, ls=':', alpha=0.6))
        ax.annotate(s['name'], (sx, sy), xytext=(8, 8),
                    textcoords='offset points', color=s['color'], fontsize=9,
                    fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    ax.plot(SPAWN_X, SPAWN_Y, '^', color='#2ecc71', ms=12, zorder=9,
            markeredgecolor='white', markeredgewidth=1.5, label='Spawn')
    ax.set_aspect('equal')
    style_ax(ax, 'Thermal Gradient Vector Field\n'
             '(Analytical gradient, arrows point toward heat sources)',
             'World X (m)', 'World Y (m)')
    ax.legend(handles=source_legend_handles() + [
        Line2D([0],[0], marker='^', color='w', mfc='#2ecc71', ms=10, lw=0, label='Spawn'),
    ], loc='lower right', fontsize=9, facecolor='#1a1a2e', labelcolor='white')

    fig.tight_layout()
    fig.savefig(out / '07_gradient_field.png', dpi=160, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.close(fig)
    print('  ✓ 07_gradient_field.png')


# ═══════════════════════════════════════════════════════════════════════════════
# 图 08 — 速度指令时序
# ═══════════════════════════════════════════════════════════════════════════════

def plot_velocity_profile(data: Dict, log_recs: List[Dict], out: Path):
    fig = dark_fig((13, 7))
    gs = GridSpec(3, 1, figure=fig, hspace=0.08, height_ratios=[2, 2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    for ax in [ax1, ax2, ax3]: ax.set_facecolor(DARK_AX)

    cmd = data['cmd']
    if cmd:
        tc   = np.array([r['t']     for r in cmd if r.get('t')     is not None])
        lin  = np.array([r.get('lin_x', 0) or 0 for r in cmd if r.get('t') is not None])
        ang  = np.array([r.get('ang_z', 0) or 0 for r in cmd if r.get('t') is not None])

        ax1.plot(tc, lin, color='#2ecc71', lw=1.8, label='linear.x (m/s)')
        ax1.fill_between(tc, 0, lin, where=lin > 0, alpha=0.2, color='#2ecc71')
        ax1.fill_between(tc, 0, lin, where=lin < 0, alpha=0.2, color='#e74c3c')
        ax1.axhline(0, color='gray', lw=0.8, alpha=0.5)
        ax1.axhline(0.22, color='cyan', lw=0.8, ls='--', alpha=0.6,
                    label='v_max=0.22 m/s')
        style_ax(ax1, '/cmd_vel — Linear Velocity', '', 'v (m/s)')
        ax1.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white', loc='upper right')

        ax2.plot(tc, ang, color='#e67e22', lw=1.8, label='angular.z (rad/s)')
        ax2.fill_between(tc, 0, ang, where=ang > 0, alpha=0.2, color='#e67e22')
        ax2.fill_between(tc, 0, ang, where=ang < 0, alpha=0.2, color='#9b59b6')
        ax2.axhline(0, color='gray', lw=0.8, alpha=0.5)
        style_ax(ax2, '', '', 'ω (rad/s)')
        ax2.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white', loc='upper right')

        # 曲率 = ang/lin（运动方向指示）
        curvature = np.where(np.abs(lin) > 0.02, ang / (lin + 1e-9), 0.0)
        curvature = np.clip(curvature, -5, 5)
        ax3.plot(tc, curvature, color='#9b59b6', lw=1.2, alpha=0.8,
                 label='Curvature κ = ω/v')
        ax3.axhline(0, color='gray', lw=0.8, alpha=0.5)
        ax3.fill_between(tc, 0, curvature, alpha=0.15, color='#9b59b6')
        style_ax(ax3, '', 'Time (s)', 'κ (1/m)')
        ax3.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white')

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
    else:
        style_ax(ax1, '/cmd_vel — Velocity Profile (No data)')
        ax1.text(0.5, 0.5, 'No /cmd_vel data recorded.\n'
                 'Check that collect_sim_data.py ran during simulation.',
                 ha='center', va='center', color='gray', fontsize=11,
                 transform=ax1.transAxes)

    for ax in [ax1, ax2, ax3]:
        for sp in ax.spines.values(): sp.set_color('#3a3a4a')

    fig.savefig(out / '08_velocity_profile.png', dpi=160, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.close(fig)
    print('  ✓ 08_velocity_profile.png')


# ═══════════════════════════════════════════════════════════════════════════════
# 图 09 — ThermalBeliefMap 重构
# ═══════════════════════════════════════════════════════════════════════════════

def plot_belief_map(data: Dict, log_recs: List[Dict], out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes: ax.set_facecolor('#050510')

    # 从轨迹和温度数据重构信念地图
    if data['traj']:
        ts, wxs, wys, _ = odom_to_world(data['traj'])
    elif log_recs:
        ts, wxs, wys = traj_from_logs(log_recs)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, 'No trajectory data', ha='center', va='center',
                    color='gray', transform=ax.transAxes)
        fig.savefig(out / '09_belief_map.png', dpi=160, bbox_inches='tight',
                    facecolor=DARK_BG)
        plt.close(fig)
        print('  ✓ 09_belief_map.png (no data)')
        return

    # 构建栅格
    res   = 0.25
    xmin, xmax = -10, 10
    ymin, ymax = -8,  8
    nx = int((xmax - xmin) / res) + 1
    ny = int((ymax - ymin) / res) + 1
    visit_map = np.zeros((ny, nx), dtype=np.float32)
    heat_map  = np.zeros((ny, nx), dtype=np.float32)

    # 温度数据（用于热度图）
    if data['thermal']:
        th_ts   = np.array([r['t'] for r in data['thermal'] if r.get('t') is not None])
        th_tmax = np.array([r.get('raw_max', AMBIENT_T) or AMBIENT_T
                            for r in data['thermal'] if r.get('t') is not None])
    else:
        th_ts   = ts.copy()
        th_tmax = np.full_like(ts, AMBIENT_T)

    for i, (wx, wy) in enumerate(zip(wxs, wys)):
        xi = int((wx - xmin) / res)
        yi = int((wy - ymin) / res)
        if 0 <= xi < nx and 0 <= yi < ny:
            # 高斯访问印记
            r_v = 3
            for di in range(-r_v, r_v+1):
                for dj in range(-r_v, r_v+1):
                    ii = xi + di; jj = yi + dj
                    if 0 <= ii < nx and 0 <= jj < ny:
                        w = np.exp(-(di**2+dj**2) / (2*1.5**2))
                        visit_map[jj, ii] += w
            # 温度映射
            t_i = ts[i]
            idx_th = np.argmin(np.abs(th_ts - t_i)) if len(th_ts) > 0 else 0
            trise  = th_tmax[idx_th] - AMBIENT_T if len(th_tmax) > 0 else 0
            if trise > 1.0:
                heat_map[yi, xi] = max(heat_map[yi, xi], trise)

    # 平滑
    if HAS_SCIPY:
        visit_map = gaussian_filter(visit_map, sigma=1.0)
        heat_map  = gaussian_filter(heat_map,  sigma=1.5)

    extent = [xmin, xmax, ymin, ymax]

    # ── 子图1：访问密度图
    ax = axes[0]
    im = ax.imshow(visit_map, origin='lower', extent=extent,
                   cmap='Blues', aspect='equal', interpolation='bilinear')
    plt.colorbar(im, ax=ax, shrink=0.85).ax.yaxis.set_tick_params(color=TEXT_C)
    for s in SOURCES:
        ax.plot(*s['xy'], '*', color=s['color'], ms=14, zorder=5,
                markeredgecolor='white', markeredgewidth=0.8)
    if len(wxs) > 1:
        ax.plot(wxs, wys, '-', color='white', lw=0.8, alpha=0.4)
    ax.plot(SPAWN_X, SPAWN_Y, '^', color='lime', ms=10, zorder=6)
    ax.set_title('Visit Density Map\n(ThermalBeliefMap.visit)',
                 color=TEXT_C, fontsize=10, fontweight='bold')
    ax.tick_params(colors=TEXT_C); ax.set_aspect('equal')
    for sp in ax.spines.values(): sp.set_color('#3a3a4a')

    # ── 子图2：热度信念图
    ax = axes[1]
    im = ax.imshow(heat_map, origin='lower', extent=extent,
                   cmap='hot', aspect='equal', interpolation='bilinear')
    plt.colorbar(im, ax=ax, shrink=0.85, label='ΔT rise (°C)'
                 ).ax.yaxis.set_tick_params(color=TEXT_C)
    for s in SOURCES:
        ax.plot(*s['xy'], '*', color='cyan', ms=14, zorder=5,
                markeredgecolor='white', markeredgewidth=0.8)
    if len(wxs) > 1:
        ax.plot(wxs, wys, '-', color='white', lw=0.8, alpha=0.4)
    ax.set_title('Heat Belief Map\n(ThermalBeliefMap.tmax)',
                 color=TEXT_C, fontsize=10, fontweight='bold')
    ax.tick_params(colors=TEXT_C); ax.set_aspect('equal')
    for sp in ax.spines.values(): sp.set_color('#3a3a4a')

    # ── 子图3：真实热场 vs 信念热场 叠加
    ax = axes[2]
    _, _, bg = make_thermal_bg(extent=(-10, 10, -8, 8))
    ax.imshow(bg, origin='lower', extent=extent, cmap=THERMAL_CMAP,
              aspect='equal', alpha=0.5, interpolation='bilinear')
    # 信念图轮廓
    if heat_map.max() > 0:
        xs_arr = np.linspace(xmin, xmax, nx)
        ys_arr = np.linspace(ymin, ymax, ny)
        ax.contour(xs_arr, ys_arr, heat_map,
                   levels=[heat_map.max() * 0.3, heat_map.max() * 0.6],
                   colors=['yellow', 'red'], linewidths=2, alpha=0.9)
    if len(wxs) > 1:
        ax.plot(wxs, wys, '-', color='white', lw=1.0, alpha=0.5)
    for s in SOURCES:
        ax.plot(*s['xy'], '*', color=s['color'], ms=14, zorder=5,
                markeredgecolor='white', markeredgewidth=0.8)
    ax.set_title('Ground Truth + Belief Contours\n(Yellow/Red = belief peaks)',
                 color=TEXT_C, fontsize=10, fontweight='bold')
    ax.tick_params(colors=TEXT_C); ax.set_aspect('equal')
    for sp in ax.spines.values(): sp.set_color('#3a3a4a')

    fig.suptitle('ThermalBeliefMap Reconstruction from Navigation Trajectory',
                 color=TEXT_C, fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out / '09_belief_map.png', dpi=160, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.close(fig)
    print('  ✓ 09_belief_map.png')


# ═══════════════════════════════════════════════════════════════════════════════
# 图 10 — 话题频率验证
# ═══════════════════════════════════════════════════════════════════════════════

def plot_topic_rates(data: Dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(DARK_BG)

    meta = data['meta']
    topic_rates = meta.get('topic_rates', {})

    # ── 左：目标 Hz vs 实际 Hz 对比条形图
    ax = axes[0]
    ax.set_facecolor(DARK_AX)
    target_hz = {
        '/odom':              50.0,
        '/sim/thermal_raw':   10.0,
        '/thermal/filtered':  10.0,
        '/thermal/field':      7.5,
        '/thermal/gradient':  10.0,
        '/cmd_vel':           10.0,
    }
    topics   = list(target_hz.keys())
    targets  = [target_hz[t] for t in topics]
    actuals  = [topic_rates.get(t, {}).get('mean_hz', 0.0) for t in topics]
    stds     = [topic_rates.get(t, {}).get('std_hz',  0.0) for t in topics]
    counts   = [topic_rates.get(t, {}).get('n_msgs',  0)   for t in topics]
    short_names = [t.split('/')[-1] for t in topics]

    x = np.arange(len(topics))
    w = 0.35
    b1 = ax.bar(x - w/2, targets, w, label='Target Hz',  color='#3498db', alpha=0.7)
    b2 = ax.bar(x + w/2, actuals, w, label='Actual Hz',  color='#2ecc71', alpha=0.7,
                yerr=stds, ecolor='white', capsize=4)
    # 标注消息计数
    for i, (bar, cnt) in enumerate(zip(b2, counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'n={cnt}', ha='center', va='bottom', fontsize=8, color=TEXT_C)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha='right', fontsize=9, color=TEXT_C)
    style_ax(ax, 'Topic Publishing Rates\n(Target vs Actual)', 'Topic', 'Frequency (Hz)')
    ax.legend(fontsize=10, facecolor='#1a1a2e', labelcolor='white')

    # ── 右：即时频率时序（每条话题）
    ax = axes[1]
    ax.set_facecolor(DARK_AX)
    # 计算各话题每秒消息数（从CSV数据）
    rate_colors = ['#e74c3c', '#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#1abc9c']
    plotted_any = False
    for color, (topic, label) in zip(rate_colors, [
            ('/sim/thermal_raw',  'thermal_raw'),
            ('/thermal/filtered', 'filtered'),
            ('/thermal/field',    'field'),
            ('/thermal/gradient', 'gradient'),
            ('/cmd_vel',          'cmd_vel'),
            ('/odom',             'odom'),
    ]):
        csv_map = {
            '/sim/thermal_raw':   data['thermal'],
            '/thermal/filtered':  data['thermal'],
            '/thermal/field':     data['field'],
            '/thermal/gradient':  data['grad'],
            '/cmd_vel':           data['cmd'],
            '/odom':              data['traj'],
        }
        rows = csv_map.get(topic, [])
        if not rows:
            continue
        t_arr = np.array([r['t'] for r in rows if r.get('t') is not None])
        if len(t_arr) < 5:
            continue
        # 计算窗口频率
        win = 2.0  # 2s 滑动窗口
        t_bins = np.arange(0, t_arr.max(), win)
        rates  = []
        for t0 in t_bins:
            cnt = np.sum((t_arr >= t0) & (t_arr < t0 + win))
            rates.append(cnt / win)
        ax.plot(t_bins, rates, color=color, lw=1.5, label=label, alpha=0.85)
        plotted_any = True

    if not plotted_any:
        ax.text(0.5, 0.5, 'Topic rate data not available\n'
                '(Run collect_sim_data.py during simulation)',
                ha='center', va='center', color='gray', fontsize=11,
                transform=ax.transAxes)
    style_ax(ax, 'Instantaneous Message Rate (2s window)', 'Time (s)', 'Msgs/s')
    ax.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white', loc='upper right')

    for ax in axes:
        for sp in ax.spines.values(): sp.set_color('#3a3a4a')

    fig.tight_layout()
    fig.savefig(out / '10_topic_rates.png', dpi=160, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.close(fig)
    print('  ✓ 10_topic_rates.png')


# ═══════════════════════════════════════════════════════════════════════════════
# 图 11 — 热源发现时间线
# ═══════════════════════════════════════════════════════════════════════════════

def plot_source_detection(data: Dict, log_recs: List[Dict],
                          found_events: List[Dict], out: Path):
    fig = dark_fig((13, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for ax in axes: ax.set_facecolor(DARK_AX)

    # ── 上左：发现时间线（水平条形图）
    ax = axes[0]
    total_t = 0.0
    if log_recs:
        total_t = log_recs[-1]['t']
    if found_events:
        prev_t = 0.0
        for i, ev in enumerate(found_events):
            ax.barh(i, ev['t'] - prev_t, left=prev_t, height=0.5,
                    color='#3a3a4a', alpha=0.5)
            ax.barh(i, 2.0, left=ev['t'] - 1, height=0.5,
                    color='lime', alpha=0.9)
            ax.text(ev['t'] + 3, i, f"t={ev['t']:.0f}s",
                    color='lime', va='center', fontsize=10, fontweight='bold')
            prev_t = ev['t']
        if total_t > 0:
            ax.barh(len(found_events), total_t, height=0.0, color='none')
        ax.set_yticks(range(len(found_events)))
        ax.set_yticklabels([f"Source #{ev['idx']}" for ev in found_events],
                           color=TEXT_C, fontsize=10)
        style_ax(ax, 'Source Discovery Timeline', 'Time (s)', '')
    else:
        style_ax(ax, 'Source Discovery Timeline')
        ax.text(0.5, 0.5, 'No source found events in logs\n(or simulation not completed)',
                ha='center', va='center', color='gray', transform=ax.transAxes)

    # ── 上右：发现温度 vs 真实峰值温度
    ax = axes[1]
    if found_events:
        found_T   = [ev.get('T', 0) for ev in found_events]
        source_T  = [SOURCES[min(ev['idx'], len(SOURCES)-1)]['peak'] for ev in found_events]
        x_pos     = np.arange(len(found_events))
        ax.bar(x_pos - 0.2, source_T, 0.35, label='Theoretical peak', color='#e74c3c', alpha=0.7)
        ax.bar(x_pos + 0.2, found_T,  0.35, label='T at detection',    color='#2ecc71', alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"S{ev['idx']}" for ev in found_events],
                           fontsize=10, color=TEXT_C)
        style_ax(ax, 'Detection Temperature vs Theoretical Peak',
                 'Source', 'Temperature (°C)')
        ax.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white')
    else:
        style_ax(ax, 'Detection Temperature')
        ax.text(0.5, 0.5, 'No detection events', ha='center', va='center',
                color='gray', transform=ax.transAxes)

    # ── 下左：状态时间占比饼图
    ax = axes[2]
    if log_recs:
        state_time: Dict[str, float] = {}
        for i, rec in enumerate(log_recs[:-1]):
            dt = log_recs[i+1]['t'] - rec['t']
            state_time[rec['state']] = state_time.get(rec['state'], 0.0) + max(0, dt)
        if state_time:
            labels  = list(state_time.keys())
            sizes   = [state_time[s] for s in labels]
            colors  = [STATE_COLORS.get(s, '#888888') for s in labels]
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'color': TEXT_C, 'fontsize': 9},
                wedgeprops={'edgecolor': DARK_BG, 'linewidth': 1.5})
            for at in autotexts: at.set_color(TEXT_C)
        style_ax(ax, 'Time Spent per State', '', '')
        ax.set_aspect('equal')
    else:
        style_ax(ax, 'Time per State')
        ax.text(0.5, 0.5, 'No log data', ha='center', va='center',
                color='gray', transform=ax.transAxes)

    # ── 下右：路径效率分析
    ax = axes[3]
    if data['traj']:
        ts, wxs, wys, _ = odom_to_world(data['traj'])
    elif log_recs:
        ts, wxs, wys = traj_from_logs(log_recs)
    else:
        ts = wxs = wys = np.array([])

    if len(ts) > 2:
        # 累积路径长度
        dx   = np.diff(wxs); dy = np.diff(wys)
        dists= np.sqrt(dx**2 + dy**2)
        cum_path = np.concatenate([[0], np.cumsum(dists)])

        # 对每个热源：最短直线距离（最优路径）
        for s in SOURCES:
            sx, sy = s['xy']
            d_spawn = math.hypot(SPAWN_X - sx, SPAWN_Y - sy)
            ax.axhline(d_spawn, color=s['color'], ls='--', lw=1.2, alpha=0.7,
                       label=f"Optimal to {s['name']} ({d_spawn:.1f}m)")

        ax.plot(ts, cum_path, color='white', lw=2, label='Actual path length')
        ax.fill_between(ts, 0, cum_path, alpha=0.12, color='white')
        style_ax(ax, 'Path Length vs Optimal', 'Time (s)', 'Distance (m)')
        ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
    else:
        style_ax(ax, 'Path Efficiency')
        ax.text(0.5, 0.5, 'No trajectory data', ha='center', va='center',
                color='gray', transform=ax.transAxes)

    for ax in axes:
        for sp in ax.spines.values(): sp.set_color('#3a3a4a')

    fig.savefig(out / '11_source_detection.png', dpi=160, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.close(fig)
    print('  ✓ 11_source_detection.png')


# ═══════════════════════════════════════════════════════════════════════════════
# 图 12 — 性能雷达图
# ═══════════════════════════════════════════════════════════════════════════════

def plot_performance_radar(data: Dict, log_recs: List[Dict],
                           found_events: List[Dict], out: Path):
    fig = dark_fig((13, 6))
    ax_radar = fig.add_subplot(121, polar=True)
    ax_bar   = fig.add_subplot(122)
    ax_radar.set_facecolor(DARK_AX)
    ax_bar.set_facecolor(DARK_AX)

    # ── 计算性能指标（0–1 归一化）──────────────────────────────────────────
    total_t     = log_recs[-1]['t'] if log_recs else 0.0
    n_found     = len(found_events)
    n_total_src = len(SOURCES)

    # 路径长度（从轨迹）
    path_len = 0.0
    if data['traj']:
        ts, wxs, wys, _ = odom_to_world(data['traj'])
        if len(ts) > 1:
            path_len = float(np.sum(np.sqrt(np.diff(wxs)**2 + np.diff(wys)**2)))
    elif log_recs:
        path_recs = [r for r in log_recs if 'path' in r]
        if path_recs: path_len = path_recs[-1]['path']

    # 最小理论路径（所有热源的TSP近似）
    optimal_path = sum(
        math.hypot(SOURCES[i]['xy'][0] - SOURCES[i-1]['xy'][0],
                   SOURCES[i]['xy'][1] - SOURCES[i-1]['xy'][1])
        for i in range(1, len(SOURCES))
    ) + math.hypot(SPAWN_X - SOURCES[0]['xy'][0], SPAWN_Y - SOURCES[0]['xy'][1])

    # 平均梯度信号质量
    avg_grad = 0.0
    if data['grad']:
        mags = [r.get('mean_mag', 0) or 0 for r in data['grad'] if r.get('t') is not None]
        avg_grad = float(np.mean(mags)) if mags else 0.0

    # 热场预处理效果（raw std vs filtered std）
    preproc_gain = 0.0
    th = data['thermal']
    if th:
        raw_stds  = [r.get('raw_std', 0) or 0 for r in th if r.get('raw_std') is not None]
        filt_stds = [r.get('filt_std', 0) or 0 for r in th if r.get('filt_std') is not None]
        if raw_stds and filt_stds and np.mean(raw_stds) > 0:
            preproc_gain = 1.0 - min(1.0, np.mean(filt_stds) / (np.mean(raw_stds) + 1e-9))

    # 归一化指标 [0-1]，越大越好
    def safe_div(a, b, max_val=None):
        if b <= 0: return 0.0
        v = a / b
        if max_val: v = min(v, max_val)
        return float(v)

    metrics = {
        'Source\nDetection':  n_found / max(1, n_total_src),
        'Path\nEfficiency':   min(1.0, optimal_path / max(path_len, 0.1)),
        'Gradient\nQuality':  min(1.0, avg_grad * 5),
        'Preproc\nEffect':    preproc_gain,
        'Time\nEfficiency':   max(0, 1.0 - total_t / 1200.0),  # 1200s = 最差
        'Topic\nReliability': min(1.0, sum(
            1 for t in data['meta'].get('topic_rates', {}).values()
            if t.get('mean_hz', 0) > 1.0) / 6.0),
    }

    labels   = list(metrics.keys())
    values   = list(metrics.values())
    N        = len(labels)
    angles   = [2 * math.pi / N * i for i in range(N)] + [0]
    values_c = values + [values[0]]

    ax_radar.plot(angles, values_c, 'o-', color=ACCENT, lw=2)
    ax_radar.fill(angles, values_c, alpha=0.25, color=ACCENT)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels, color=TEXT_C, fontsize=9)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_radar.set_yticklabels(['0.25', '0.5', '0.75', '1.0'],
                              color='gray', fontsize=7)
    ax_radar.grid(color=GRID_C, alpha=0.5)
    ax_radar.set_title('Navigation Performance\nRadar', color=TEXT_C,
                       fontsize=11, fontweight='bold', pad=15)

    # ── 右：指标详情条形图
    short_labels = [l.replace('\n', ' ') for l in labels]
    bar_colors   = [ACCENT if v >= 0.75 else '#e67e22' if v >= 0.5 else '#e74c3c'
                    for v in values]
    bars = ax_bar.barh(short_labels, values, color=bar_colors, alpha=0.8, height=0.6)
    for bar, val in zip(bars, values):
        ax_bar.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center', fontsize=10, color=TEXT_C,
                    fontweight='bold')
    ax_bar.axvline(0.5,  color='gray',   ls='--', lw=1.0, alpha=0.5)
    ax_bar.axvline(0.75, color='#2ecc71', ls='--', lw=1.0, alpha=0.5)
    ax_bar.set_xlim(0, 1.15)
    style_ax(ax_bar, 'Performance Metrics (0–1 normalized)',
             'Score', '')
    ax_bar.tick_params(axis='y', colors=TEXT_C)

    # 摘要文本框
    summary = (f'Sources Found: {n_found}/{n_total_src}\n'
               f'Path Length:   {path_len:.1f} m\n'
               f'Duration:      {total_t:.1f} s\n'
               f'Avg |∇T|:      {avg_grad:.3f} °C/px\n'
               f'Optimal path: ~{optimal_path:.1f} m')
    ax_bar.text(0.98, 0.02, summary, transform=ax_bar.transAxes,
                fontsize=8.5, va='bottom', ha='right', fontfamily='monospace',
                color=TEXT_C, bbox=dict(boxstyle='round,pad=0.6',
                fc='#1a1a2e', alpha=0.9, ec=ACCENT, lw=1.5))

    for ax in [ax_bar]:
        for sp in ax.spines.values(): sp.set_color('#3a3a4a')

    fig.tight_layout()
    fig.savefig(out / '12_performance_metrics.png', dpi=160, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.close(fig)
    print('  ✓ 12_performance_metrics.png')


# ═══════════════════════════════════════════════════════════════════════════════
# 图 13 — 综合面板（答辩主图）
# ═══════════════════════════════════════════════════════════════════════════════

def plot_dashboard(data: Dict, log_recs: List[Dict],
                   found_events: List[Dict], out: Path):
    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(
        'Thermal Navigation System — Comprehensive Performance Dashboard\n'
        'Route A: Gradient-Based Thermal Source Seeking  |  ROS2 Humble + Gazebo Classic',
        fontsize=14, fontweight='bold', color=TEXT_C, y=0.99)

    gs = GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.35,
                  left=0.06, right=0.97, top=0.94, bottom=0.06)

    # ── [0,0:2] 轨迹（占2列）
    ax_t = fig.add_subplot(gs[0, :2])
    ax_t.set_facecolor(DARK_AX)
    XX, YY, bg = make_thermal_bg()
    ax_t.contourf(XX, YY, bg, levels=20, cmap=THERMAL_CMAP, alpha=0.40)
    if data['traj']:
        ts, wxs, wys, _ = odom_to_world(data['traj'])
        if len(ts) > 1:
            sc = ax_t.scatter(wxs, wys, c=ts, cmap='cool', s=5, alpha=0.7,
                              zorder=4, linewidths=0)
            ax_t.plot(wxs, wys, '-', color='white', lw=0.8, alpha=0.3, zorder=3)
            ax_t.plot(wxs[0], wys[0], '^', color='#2ecc71', ms=11, zorder=8,
                      markeredgecolor='white', markeredgewidth=1.2)
    elif log_recs:
        ts, wxs, wys = traj_from_logs(log_recs)
        if len(ts) > 1:
            sc = ax_t.scatter(wxs, wys, c=ts, cmap='cool', s=10, alpha=0.7,
                              zorder=4, linewidths=0)
    for s in SOURCES:
        ax_t.plot(*s['xy'], '*', color=s['color'], ms=16, zorder=9,
                  markeredgecolor='white', markeredgewidth=0.8)
    ax_t.plot(SPAWN_X, SPAWN_Y, '^', color='#2ecc71', ms=11, zorder=9,
              markeredgecolor='white', markeredgewidth=1.2)
    ax_t.set_aspect('equal')
    style_ax(ax_t, 'Navigation Trajectory (color = time)', 'X (m)', 'Y (m)')

    # ── [0,2] 温度
    ax_T = fig.add_subplot(gs[0, 2])
    ax_T.set_facecolor(DARK_AX)
    if data['thermal']:
        ts_r  = np.array([r['t'] for r in data['thermal'] if r.get('t') is not None])
        raw_m = np.array([r.get('raw_max', AMBIENT_T) or AMBIENT_T
                          for r in data['thermal'] if r.get('t') is not None])
        ax_T.plot(ts_r, raw_m, color='#e74c3c', lw=2, label='T_max')
        ax_T.fill_between(ts_r, AMBIENT_T, raw_m, alpha=0.15, color='#e74c3c')
        for ev in found_events:
            ax_T.axvline(ev['t'], color='lime', ls='--', lw=1.2, alpha=0.7)
        ax_T.set_ylim(AMBIENT_T - 2, raw_m.max() + 5)
    elif log_recs:
        tl = np.array([r['t'] for r in log_recs])
        Tl = np.array([r['T'] for r in log_recs])
        ax_T.plot(tl, Tl, color='#e74c3c', lw=2)
    style_ax(ax_T, 'T_max in FOV (°C)', 't (s)', '°C')

    # ── [0,3] 梯度
    ax_G = fig.add_subplot(gs[0, 3])
    ax_G.set_facecolor(DARK_AX)
    if data['grad']:
        tg  = np.array([r['t'] for r in data['grad'] if r.get('t') is not None])
        mag = np.array([r.get('mean_mag', 0) or 0 for r in data['grad']
                        if r.get('t') is not None])
        ax_G.plot(tg, mag, color='#3498db', lw=2)
        ax_G.fill_between(tg, 0, mag, alpha=0.15, color='#3498db')
        ax_G.axhline(0.3, color='#e74c3c', ls=':', lw=1.0, alpha=0.6)
    elif log_recs:
        tl  = np.array([r['t']  for r in log_recs])
        gml = np.array([r['gm'] for r in log_recs])
        ax_G.plot(tl, gml, color='#3498db', lw=2)
    style_ax(ax_G, '|∇T| Mean (°C/px)', 't (s)', '°C/px')

    # ── [1,0:2] 收敛距离
    ax_D = fig.add_subplot(gs[1, :2])
    ax_D.set_facecolor(DARK_AX)
    if data['traj']:
        ts, wxs, wys, _ = odom_to_world(data['traj'])
    elif log_recs:
        ts, wxs, wys = traj_from_logs(log_recs)
    else:
        ts = wxs = wys = np.array([])
    if len(ts) > 1:
        for s in SOURCES:
            sx, sy = s['xy']
            dist = np.sqrt((wxs - sx)**2 + (wys - sy)**2)
            ax_D.plot(ts, dist, color=s['color'], lw=2, label=s['name'])
        ax_D.axhline(ARRIVAL_R, color='lime', ls='--', lw=1.5)
        ax_D.fill_between(ts, 0, ARRIVAL_R, alpha=0.07, color='lime')
    ax_D.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
    style_ax(ax_D, 'Distance to Heat Sources', 't (s)', 'm')

    # ── [1,2] 速度
    ax_V = fig.add_subplot(gs[1, 2])
    ax_V.set_facecolor(DARK_AX)
    if data['cmd']:
        tc   = np.array([r['t'] for r in data['cmd'] if r.get('t') is not None])
        lin  = np.array([r.get('lin_x', 0) or 0 for r in data['cmd']
                         if r.get('t') is not None])
        ang  = np.array([r.get('ang_z', 0) or 0 for r in data['cmd']
                         if r.get('t') is not None])
        ax_V.plot(tc, lin, color='#2ecc71', lw=1.5, label='v (m/s)')
        ax_V.plot(tc, ang * 0.2, color='#e67e22', lw=1.5, alpha=0.7,
                  label='ω×0.2 (rad/s)')
        ax_V.axhline(0, color='gray', lw=0.8, alpha=0.5)
        ax_V.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
    style_ax(ax_V, 'Velocity Commands', 't (s)', 'v (m/s) / ω×0.2')

    # ── [1,3] 状态甘特（紧凑版）
    ax_S = fig.add_subplot(gs[1, 3])
    ax_S.set_facecolor(DARK_AX)
    if log_recs:
        all_states  = list(dict.fromkeys(r['state'] for r in log_recs))
        state_to_y  = {s: i for i, s in enumerate(all_states)}
        for i, rec in enumerate(log_recs[:-1]):
            t0 = rec['t']; t1 = log_recs[i+1]['t']
            y  = state_to_y.get(rec['state'], 0)
            ax_S.barh(y, t1-t0, left=t0, height=0.65,
                      color=STATE_COLORS.get(rec['state'], '#888'), alpha=0.85)
        ax_S.set_yticks(range(len(all_states)))
        ax_S.set_yticklabels(all_states, fontsize=7, color=TEXT_C)
    style_ax(ax_S, 'State Timeline', 't (s)', '')

    # ── [2,0:3] 摘要文本 + 发现事件
    ax_sum = fig.add_subplot(gs[2, :3])
    ax_sum.set_facecolor(DARK_AX)
    ax_sum.axis('off')

    # 构建摘要
    total_t     = log_recs[-1]['t'] if log_recs else 0.0
    path_len    = 0.0
    if data['traj'] and len(data['traj']) > 1:
        ts, wxs, wys, _ = odom_to_world(data['traj'])
        path_len = float(np.sum(np.sqrt(np.diff(wxs)**2 + np.diff(wys)**2)))
    elif log_recs:
        pr = [r for r in log_recs if 'path' in r and r['path']]
        if pr: path_len = pr[-1]['path']
    avg_grad = 0.0
    if data['grad']:
        mags = [r.get('mean_mag', 0) or 0 for r in data['grad']
                if r.get('t') is not None]
        avg_grad = float(np.mean(mags)) if mags else 0.0
    n_found = len(found_events)

    rate_str = '  '.join(
        f"{t.split('/')[-1]}={v.get('mean_hz',0):.1f}Hz"
        for t, v in data['meta'].get('topic_rates', {}).items()
        if isinstance(v, dict))

    found_str = '\n'.join(
        f"  Source #{ev['idx']}: t={ev['t']:.0f}s  T={ev.get('T',0):.1f}°C"
        for ev in found_events) or '  (none detected)'

    summary_text = (
        f"{'─'*70}\n"
        f"THERMAL NAVIGATION PERFORMANCE SUMMARY  |  Algorithm: Hybrid Gradient Ascent v20\n"
        f"{'─'*70}\n"
        f"Duration:      {total_t:.1f} s           "
        f"Path Length:    {path_len:.1f} m\n"
        f"Sources Found: {n_found}/{len(SOURCES)}             "
        f"Avg |∇T|:       {avg_grad:.3f} °C/px\n"
        f"Sensor:        64×48 @ 10Hz          "
        f"Gradient Method: Sobel 3×3\n"
        f"Topic Rates:   {rate_str}\n"
        f"{'─'*70}\n"
        f"Source Discovery Events:\n{found_str}\n"
        f"{'─'*70}"
    )
    ax_sum.text(0.02, 0.97, summary_text, transform=ax_sum.transAxes,
                fontsize=8.5, va='top', fontfamily='monospace', color=TEXT_C,
                bbox=dict(boxstyle='round,pad=0.8', fc='#0d0d1a',
                          alpha=0.95, ec=ACCENT, lw=1.5))

    # ── [2,3] Topic Rates 条形
    ax_rate = fig.add_subplot(gs[2, 3])
    ax_rate.set_facecolor(DARK_AX)
    topic_rates = data['meta'].get('topic_rates', {})
    if topic_rates:
        t_names  = [k.split('/')[-1] for k in topic_rates]
        t_actual = [v.get('mean_hz', 0) if isinstance(v, dict) else 0
                    for v in topic_rates.values()]
        t_target = {'odom': 50, 'thermal_raw': 10, 'filtered': 10,
                    'field': 7.5, 'gradient': 5, 'cmd_vel': 10}
        colors_r = ['#2ecc71' if a >= t_target.get(n, 1) * 0.8 else '#e74c3c'
                    for n, a in zip(t_names, t_actual)]
        ax_rate.barh(t_names, t_actual, color=colors_r, alpha=0.8)
        style_ax(ax_rate, 'Topic Hz', 'Hz', '')
        ax_rate.tick_params(axis='y', colors=TEXT_C, labelsize=8)
    else:
        style_ax(ax_rate, 'Topic Rates')
        ax_rate.text(0.5, 0.5, 'No rate data', ha='center', va='center',
                     color='gray', transform=ax_rate.transAxes)

    fig.savefig(out / '13_dashboard.png', dpi=160, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.close(fig)
    print('  ✓ 13_dashboard.png')


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log_only = '--log-only' in sys.argv
    args     = [a for a in sys.argv[1:] if not a.startswith('--')]

    # ── 确定数据目录
    if args:
        data_dir = Path(args[0]).expanduser()
    else:
        data_dir = find_latest_collection()
        if data_dir is None and not log_only:
            print('[WARN] 未找到采集目录，切换至 log-only 模式。')
            log_only = True
            data_dir = Path('/tmp/no_collection')
        elif data_dir:
            print(f'自动找到最新采集目录：{data_dir}')

    # ── 加载采集数据
    if data_dir and data_dir.exists() and not log_only:
        data = load_collection(data_dir)
        out  = data_dir / 'figures'
    else:
        data = {'meta': {}, 'traj': [], 'thermal': [], 'field': [],
                'grad': [], 'cmd': [], 'snap_index': [],
                'data_dir': Path('/tmp')}
        out  = Path.home() / 'ros2_ws' / 'bags' / 'plots' / \
               datetime.now().strftime('%Y%m%d_%H%M%S')

    out.mkdir(parents=True, exist_ok=True)
    print(f'输出目录：{out}')

    # ── 加载 ROS 日志
    print('读取 ROS2 日志...')
    ros_session = find_latest_ros_session()
    log_recs, found_events = [], []
    if ros_session:
        print(f'  session: {ros_session}')
        text = read_session_text(ros_session)
        log_recs, found_events = parse_controller_logs(text)
        print(f'  日志记录: {len(log_recs)} 条  发现事件: {len(found_events)} 个')
    else:
        print('  [WARN] 未找到 ROS 日志，部分图将为空。')

    print(f'\n生成 13 张图表...')
    print(f'  采集数据: '
          f'traj={len(data["traj"])} '
          f'thermal={len(data["thermal"])} '
          f'grad={len(data["grad"])} '
          f'cmd={len(data["cmd"])} '
          f'snapshots={len(data["snap_index"])}')

    # ── 生成所有图
    plot_trajectory(data, log_recs, found_events, out)
    plot_temperature_series(data, log_recs, found_events, out)
    plot_gradient_analysis(data, log_recs, found_events, out)
    plot_convergence(data, log_recs, found_events, out)
    plot_state_timeline(log_recs, found_events, out)
    plot_thermal_snapshots(data, out)
    plot_gradient_field(data, log_recs, out)
    plot_velocity_profile(data, log_recs, out)
    plot_belief_map(data, log_recs, out)
    plot_topic_rates(data, out)
    plot_source_detection(data, log_recs, found_events, out)
    plot_performance_radar(data, log_recs, found_events, out)
    plot_dashboard(data, log_recs, found_events, out)

    print(f'\n✅ 全部 13 张图表已保存至：{out}')
    print('\n文件列表：')
    for f in sorted(out.glob('*.png')):
        size_kb = f.stat().st_size // 1024
        print(f'  {f.name:45s}  {size_kb:>5d} KB')


if __name__ == '__main__':
    main()
