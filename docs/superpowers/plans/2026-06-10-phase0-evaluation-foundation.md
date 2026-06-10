# 阶段 0：评测地基 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **For Codex CLI:** 按任务顺序执行，每个任务内严格按步骤执行（先测试后实现），每个任务结束时提交一次。所有命令在 `/home/hanchen/ros2_ws` 下运行。

**Goal:** 建立多 seed 统计、失败归因仪表、真值占据栅格、基线策略挂架和升级版场景矩阵 runner，使后续所有算法迭代都有可复现、可归因的评测基础。

**Architecture:** 三个新的纯 Python 模块（`matrix_stats.py` 统计/报告、`world_occupancy.py` SDF→占据栅格+视线检测、`attribution.py` 逐真值源归因）+ 对现有 scenario/sensor/controller/launch 的 seed 与 strategy 参数接线 + `run_multiscenario_matrix.py` 全量重写。纯模块零 ROS 依赖，由 pytest 覆盖；ROS 侧只做参数传递。

**Tech Stack:** Python 3.10, numpy, PyYAML, xml.etree（标准库），ROS 2 Humble launch（仅任务 6），pytest。

**依据 spec:** `docs/superpowers/specs/2026-06-10-thermal-multisource-program-design.md` §4 阶段 0、§5 评测体系。

**重要背景事实（执行前必读）：**

- 6 个世界文件已存在于 `src/thermal_robot/thermal_bringup/worlds/`（nav=开阔、obstacle_field=箱体、corridor_rooms=隔墙、mixed_rooms=混合、zigzag_corridors、sparse_islands）。本计划**不新建世界**，只为它们生成真值占据栅格。
- 采集器 `collect_sim_data.py` 已输出 `trajectory.csv`（列 `t,x,y,wx,wy,yaw,vx,wz`，wx/wy 为世界坐标）、`thermal_sources_truth.csv`（列 `t,id,status,x,y,strength,sigma,probability,confidence,observations`，status 为 `truth_active`/`truth_inactive`，10 Hz 全源记录）、`source_estimates.csv`（同列结构，status 为 tracker 状态）、`source_summary.json`（含 `localization_errors_m` 匹配列表，每项含 `truth_id` 和 `estimate_id`）、`metadata.json`（含 `counts`、`nav2_available`、`nav2_plan_count`）。归因模块**复用这些产物**，不改采集器。
- 测试导入约定：`src/thermal_robot/tests/test_thermal_system.py` 顶部把各包目录插入 `sys.path`；脚本模块用 `importlib.util.spec_from_file_location` 按文件路径加载（见该文件 718 行的现有模式）。
- 热足迹模型：以机器人为中心、随 yaw 旋转的 4.0m×3.0m 矩形（`scenario.py` 的 `SENSOR_FOV_X/Y`，`sensor_node.py` 115–122 行）。
- 随机性来源：sensor_node 噪声（`random_seed` 参数已存在，默认 42）；controller 的 Lévy/escape 用未播种的 Python `random` 模块（`controller_node.py` 1308/1608/1994 行）；动态源 random_walk 用 `DynamicHeatSource.seed` 的确定性公式（`scenario.py` 150 行）。

---

## 任务总览

| # | 任务 | 新建/修改 |
|---|---|---|
| 1 | matrix_stats.py：聚合 + Mann-Whitney U + Markdown 报告 | 新建（纯） |
| 2 | world_occupancy.py：SDF 解析 + 栅格化 + 视线检测 | 新建（纯） |
| 3 | 为 6 个世界生成真值占据栅格 | 生成数据 |
| 4 | scenario.apply_run_seed：seed 扰动 | 修改（纯） |
| 5 | static_five_sources.yaml 场景 | 新建配置 |
| 6 | seed/strategy 参数接线（sensor/controller/launch） | 修改 ROS 侧 |
| 7 | attribution.py：逐真值源归因 | 新建（纯） |
| 8 | run_multiscenario_matrix.py 重写（seeds/筛选/策略/归因/报告/health-only） | 重写 |
| 9 | v31 锚点 tag + 全量验证 + 冒烟 + devlog | 验证 |

所有新测试写入 `src/thermal_robot/tests/test_phase0_evaluation.py`（任务 1 创建该文件，后续任务追加）。

---

### 任务 1：matrix_stats.py — 聚合、Mann-Whitney U、Markdown 报告

**Files:**
- Create: `src/thermal_robot/scripts/matrix_stats.py`
- Create: `src/thermal_robot/tests/test_phase0_evaluation.py`

- [ ] **Step 1.1: 创建测试文件并写失败测试**

创建 `src/thermal_robot/tests/test_phase0_evaluation.py`，完整内容：

```python
#!/usr/bin/env python3
"""阶段 0 评测地基的纯算法单元测试（无 ROS 依赖）。

覆盖: matrix_stats / world_occupancy / scenario seed 扰动 / attribution / 矩阵 runner 配置。
运行: python3 -m pytest src/thermal_robot/tests/test_phase0_evaluation.py -v
"""

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[3]
SCRIPTS = WORKSPACE / 'src/thermal_robot/scripts'

for rel in [
    'src/thermal_robot/thermal_sensor_sim',
]:
    path = str(WORKSPACE / rel)
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestMatrixStats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ms = _load_script('matrix_stats')

    def test_rank_with_ties(self):
        self.assertEqual(self.ms._rank([1.0, 1.0, 2.0]), [1.5, 1.5, 3.0])

    def test_aggregate_case_runs_mean_std(self):
        runs = [
            {'passed': True, 'source_recall': 0.6, 'source_precision': 1.0,
             'duplicate_confirmations': 0, 'time_to_first_source': 30.0, 'path_length_m': 30.0},
            {'passed': True, 'source_recall': 0.4, 'source_precision': 1.0,
             'duplicate_confirmations': 0, 'time_to_first_source': 50.0, 'path_length_m': 50.0},
        ]
        agg = self.ms.aggregate_case_runs(runs)
        self.assertEqual(agg['n_runs'], 2)
        self.assertEqual(agg['n_passed'], 2)
        recall = agg['metrics']['source_recall']
        self.assertAlmostEqual(recall['mean'], 0.5, places=6)
        # 样本标准差: sqrt(((0.1)^2+(0.1)^2)/1) ≈ 0.1414
        self.assertAlmostEqual(recall['std'], 0.1414, places=3)
        self.assertEqual(recall['n'], 2)

    def test_aggregate_skips_none_metric(self):
        runs = [{'passed': False, 'source_recall': None, 'source_precision': None,
                 'duplicate_confirmations': None, 'time_to_first_source': None,
                 'path_length_m': None}]
        agg = self.ms.aggregate_case_runs(runs)
        self.assertIsNone(agg['metrics']['source_recall'])

    def test_mann_whitney_exact_separated(self):
        # a 全小于 b: U=0, 双侧精确 p = 2/C(6,3) = 0.1
        result = self.ms.mann_whitney_u([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        self.assertEqual(result['method'], 'exact')
        self.assertAlmostEqual(result['u'], 0.0, places=6)
        self.assertAlmostEqual(result['p_value'], 0.1, places=6)

    def test_mann_whitney_identical_groups_high_p(self):
        result = self.ms.mann_whitney_u([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertGreater(result['p_value'], 0.5)

    def test_mann_whitney_empty_input(self):
        result = self.ms.mann_whitney_u([], [1.0])
        self.assertIsNone(result['p_value'])

    def test_render_markdown_report_contains_cases(self):
        summary = {
            'config': {'preset': 'phase0', 'strategy': 'full', 'seeds': [101, 102],
                       'duration_s': 120.0, 'warmup_s': 36.0},
            'cases': {
                'open__static2': {
                    'n_runs': 2, 'n_passed': 2,
                    'metrics': {'source_recall': {'mean': 1.0, 'std': 0.0, 'min': 1.0,
                                                  'max': 1.0, 'n': 2, 'values': [1.0, 1.0]},
                                'source_precision': {'mean': 1.0, 'std': 0.0, 'min': 1.0,
                                                     'max': 1.0, 'n': 2, 'values': [1.0, 1.0]},
                                'duplicate_confirmations': {'mean': 0.0, 'std': 0.0, 'min': 0.0,
                                                            'max': 0.0, 'n': 2, 'values': [0.0, 0.0]},
                                'time_to_first_source': None,
                                'path_length_m': None},
                    'failure_counts': {'not_reached': 0, 'occluded': 0,
                                       'timing_missed': 0, 'not_confirmed': 0},
                },
            },
        }
        text = self.ms.render_markdown_report(summary)
        self.assertIn('open__static2', text)
        self.assertIn('1.000', text)
        self.assertIn('not_reached', text)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 1.2: 运行测试确认失败**

Run: `python3 -m pytest src/thermal_robot/tests/test_phase0_evaluation.py -v`
Expected: ERROR/FAIL（`matrix_stats.py` 不存在，`FileNotFoundError`）

- [ ] **Step 1.3: 实现 matrix_stats.py**

创建 `src/thermal_robot/scripts/matrix_stats.py`，完整内容：

```python
#!/usr/bin/env python3
"""Pure aggregation, rank statistics, and report rendering for the scenario matrix."""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional, Sequence

METRIC_KEYS = (
    'source_recall',
    'source_precision',
    'duplicate_confirmations',
    'time_to_first_source',
    'path_length_m',
)

FAILURE_CLASSES = ('not_reached', 'occluded', 'timing_missed', 'not_confirmed')

EXACT_TEST_MAX_COMBINATIONS = 20000


def _rank(values: Sequence[float]) -> List[float]:
    """Average ranks (1-based) with tie handling."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def aggregate_case_runs(runs: Sequence[Dict]) -> Dict:
    out = {
        'n_runs': len(runs),
        'n_passed': sum(1 for r in runs if r.get('passed')),
        'metrics': {},
    }
    for key in METRIC_KEYS:
        values = [float(r[key]) for r in runs
                  if r.get(key) is not None and math.isfinite(float(r[key]))]
        if not values:
            out['metrics'][key] = None
            continue
        mean = sum(values) / len(values)
        if len(values) > 1:
            var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        else:
            var = 0.0
        out['metrics'][key] = {
            'mean': round(mean, 4),
            'std': round(math.sqrt(var), 4),
            'min': round(min(values), 4),
            'max': round(max(values), 4),
            'n': len(values),
            'values': [round(v, 4) for v in values],
        }
    return out


def mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> Dict:
    """Two-sided Mann-Whitney U. Exact permutation when feasible, else
    normal approximation with tie correction and continuity correction."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return {'u': None, 'p_value': None, 'method': 'undefined', 'n1': n1, 'n2': n2}
    pooled = list(a) + list(b)
    ranks = _rank(pooled)
    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u = min(u1, n1 * n2 - u1)

    if math.comb(n1 + n2, n1) <= EXACT_TEST_MAX_COMBINATIONS:
        count = 0
        total = 0
        for combo in combinations(range(n1 + n2), n1):
            rr = sum(ranks[i] for i in combo)
            uu1 = rr - n1 * (n1 + 1) / 2.0
            uu = min(uu1, n1 * n2 - uu1)
            total += 1
            if uu <= u + 1e-9:
                count += 1
        p = min(1.0, count / total)
        method = 'exact'
    else:
        mu = n1 * n2 / 2.0
        n = n1 + n2
        tie_term = sum(t ** 3 - t for t in Counter(pooled).values())
        sigma_sq = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))
        if sigma_sq <= 0.0:
            p = 1.0
        else:
            z = (u - mu + 0.5) / math.sqrt(sigma_sq)
            p = min(1.0, math.erfc(abs(z) / math.sqrt(2.0)))
        method = 'normal_approx'
    return {'u': round(u, 4), 'p_value': round(p, 5), 'method': method, 'n1': n1, 'n2': n2}


def _fmt_metric(metric: Optional[Dict]) -> str:
    if not metric:
        return '—'
    return f"{metric['mean']:.3f}±{metric['std']:.3f} (n={metric['n']})"


def render_markdown_report(summary: Dict) -> str:
    cfg = summary.get('config', {})
    lines = [
        '# 场景矩阵评测报告',
        '',
        f"- preset: `{cfg.get('preset')}`  strategy: `{cfg.get('strategy')}`",
        f"- seeds: {cfg.get('seeds')}",
        f"- duration: {cfg.get('duration_s')}s  warmup: {cfg.get('warmup_s')}s",
        '',
        '## 每用例多 seed 统计',
        '',
        '| case | runs | passed | recall | precision | duplicates | t_first(s) |',
        '|---|---|---|---|---|---|---|',
    ]
    for name in sorted(summary.get('cases', {})):
        case = summary['cases'][name]
        m = case.get('metrics', {})
        lines.append(
            f"| {name} | {case.get('n_runs')} | {case.get('n_passed')} "
            f"| {_fmt_metric(m.get('source_recall'))} "
            f"| {_fmt_metric(m.get('source_precision'))} "
            f"| {_fmt_metric(m.get('duplicate_confirmations'))} "
            f"| {_fmt_metric(m.get('time_to_first_source'))} |"
        )
    lines += ['', '## 漏源失败归因汇总', '',
              '| case | ' + ' | '.join(FAILURE_CLASSES) + ' |',
              '|---|' + '---|' * len(FAILURE_CLASSES)]
    for name in sorted(summary.get('cases', {})):
        counts = summary['cases'][name].get('failure_counts') or {}
        row = ' | '.join(str(counts.get(k, 0)) for k in FAILURE_CLASSES)
        lines.append(f'| {name} | {row} |')
    lines.append('')
    return '\n'.join(lines)
```

- [ ] **Step 1.4: 运行测试确认通过**

Run: `python3 -m pytest src/thermal_robot/tests/test_phase0_evaluation.py -v`
Expected: `TestMatrixStats` 全部 PASS（7 个测试）

- [ ] **Step 1.5: 提交**

```bash
git add src/thermal_robot/scripts/matrix_stats.py src/thermal_robot/tests/test_phase0_evaluation.py
git commit -m "feat: add matrix aggregation and Mann-Whitney U statistics module"
```

---

### 任务 2：world_occupancy.py — SDF 解析、栅格化、视线检测

**Files:**
- Create: `src/thermal_robot/scripts/world_occupancy.py`
- Modify: `src/thermal_robot/tests/test_phase0_evaluation.py`（追加测试类）

- [ ] **Step 2.1: 追加失败测试**

在 `test_phase0_evaluation.py` 的 `if __name__ == '__main__':` 之前追加：

```python
SAMPLE_SDF = """<?xml version="1.0"?>
<sdf version="1.6">
  <world name="test_world">
    <include><uri>model://sun</uri></include>
    <model name="visual_only_marker">
      <static>true</static>
      <pose>-6.0 0 0 0 0 0</pose>
      <link name="link">
        <visual name="pad"><geometry><box><size>0.6 0.6 0.04</size></box></geometry></visual>
      </link>
    </model>
    <model name="wall_east">
      <static>true</static>
      <pose>2.0 0.0 0.45 0 0 0</pose>
      <link name="link">
        <collision name="collision"><geometry><box><size>2.0 1.0 0.9</size></box></geometry></collision>
      </link>
    </model>
    <model name="wall_rotated">
      <static>true</static>
      <pose>0.0 5.0 0.45 0 0 1.5707963</pose>
      <link name="link">
        <collision name="collision"><geometry><box><size>4.0 0.4 0.9</size></box></geometry></collision>
      </link>
    </model>
  </world>
</sdf>
"""


class TestWorldOccupancy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wo = _load_script('world_occupancy')

    def test_parse_boxes_skips_visual_only(self):
        boxes = self.wo.parse_world_boxes(SAMPLE_SDF)
        names = sorted(b.name for b in boxes)
        self.assertEqual(names, ['wall_east', 'wall_rotated'])

    def test_parse_box_pose_and_size(self):
        boxes = {b.name: b for b in self.wo.parse_world_boxes(SAMPLE_SDF)}
        wall = boxes['wall_east']
        self.assertAlmostEqual(wall.x, 2.0)
        self.assertAlmostEqual(wall.y, 0.0)
        self.assertAlmostEqual(wall.size_x, 2.0)
        self.assertAlmostEqual(wall.size_y, 1.0)

    def test_rasterize_marks_box_cells(self):
        boxes = self.wo.parse_world_boxes(SAMPLE_SDF)
        grid = self.wo.rasterize(boxes)
        self.assertTrue(self.wo.is_occupied(grid, 2.0, 0.0))
        self.assertFalse(self.wo.is_occupied(grid, -3.0, 0.0))

    def test_rasterize_respects_yaw(self):
        # wall_rotated: 4.0x0.4 绕 z 转 90°，占据范围 x∈[-0.2,0.2], y∈[3,7]
        boxes = self.wo.parse_world_boxes(SAMPLE_SDF)
        grid = self.wo.rasterize(boxes)
        self.assertTrue(self.wo.is_occupied(grid, 0.0, 6.5))
        self.assertFalse(self.wo.is_occupied(grid, 1.5, 6.5))

    def test_line_of_sight_blocked_and_clear(self):
        boxes = self.wo.parse_world_boxes(SAMPLE_SDF)
        grid = self.wo.rasterize(boxes)
        # 穿过 wall_east (x∈[1,3], y∈[-0.5,0.5])
        self.assertFalse(self.wo.line_of_sight(grid, -1.0, 0.0, 5.0, 0.0))
        # 从墙旁边绕过
        self.assertTrue(self.wo.line_of_sight(grid, -1.0, 2.0, 5.0, 2.0))

    def test_npz_roundtrip(self):
        import tempfile
        boxes = self.wo.parse_world_boxes(SAMPLE_SDF)
        grid = self.wo.rasterize(boxes)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'g.npz'
            self.wo.save_grid(grid, path)
            loaded = self.wo.load_grid(path)
        self.assertEqual(loaded.data.shape, grid.data.shape)
        self.assertTrue(self.wo.is_occupied(loaded, 2.0, 0.0))
```

- [ ] **Step 2.2: 运行测试确认失败**

Run: `python3 -m pytest src/thermal_robot/tests/test_phase0_evaluation.py::TestWorldOccupancy -v`
Expected: ERROR（`world_occupancy.py` 不存在）

- [ ] **Step 2.3: 实现 world_occupancy.py**

创建 `src/thermal_robot/scripts/world_occupancy.py`，完整内容：

```python
#!/usr/bin/env python3
"""Ground-truth occupancy grids from Gazebo SDF worlds, with line-of-sight checks.

用途:
  1. 归因仪表: 判断真值源与机器人之间的热视线是否被障碍物遮挡(上帝视角)。
  2. 后续阶段: 运行时可见性掩码的测试基准。

栅格约定与 WorldThermalGrid 一致: 中心 (-6,0), 50m x 50m, 分辨率 0.1m。
仅纳入"会遮挡热视线/激光"高度带的静态 box collision
(z 跨度与 [OBSTACLE_Z_MIN, OBSTACLE_Z_MAX] 相交)。
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

GRID_CENTER_X = -6.0
GRID_CENTER_Y = 0.0
GRID_SIZE_M = 50.0
GRID_RESOLUTION = 0.1
OBSTACLE_Z_MIN = 0.15
OBSTACLE_Z_MAX = 1.5
LOS_STEP_M = 0.05


@dataclass(frozen=True)
class WorldBox:
    name: str
    x: float
    y: float
    z: float
    yaw: float
    size_x: float
    size_y: float
    size_z: float


@dataclass
class TruthOccupancyGrid:
    resolution: float
    origin_x: float
    origin_y: float
    data: np.ndarray  # uint8 (H, W), 1 = occupied


def _parse_pose(text: str) -> Tuple[float, float, float, float]:
    parts = [float(v) for v in (text or '0 0 0 0 0 0').split()]
    while len(parts) < 6:
        parts.append(0.0)
    return parts[0], parts[1], parts[2], parts[5]


def parse_world_boxes(sdf_text: str) -> List[WorldBox]:
    root = ET.fromstring(sdf_text)
    boxes: List[WorldBox] = []
    for model in root.iter('model'):
        mx, my, mz, myaw = _parse_pose(model.findtext('pose', default='0 0 0 0 0 0'))
        for link in model.findall('link'):
            lx, ly, lz, lyaw = _parse_pose(link.findtext('pose', default='0 0 0 0 0 0'))
            for coll in link.findall('collision'):
                cx, cy, cz, cyaw = _parse_pose(coll.findtext('pose', default='0 0 0 0 0 0'))
                box = coll.find('geometry/box/size')
                if box is None or box.text is None:
                    continue
                sx, sy, sz = (float(v) for v in box.text.split()[:3])
                # 两级位姿合成(模型yaw作用于link/collision偏移; 仅取yaw)
                ox, oy = lx + cx, ly + cy
                wx = mx + math.cos(myaw) * ox - math.sin(myaw) * oy
                wy = my + math.sin(myaw) * ox + math.cos(myaw) * oy
                wz = mz + lz + cz
                yaw = myaw + lyaw + cyaw
                z_lo, z_hi = wz - sz / 2.0, wz + sz / 2.0
                if z_hi < OBSTACLE_Z_MIN or z_lo > OBSTACLE_Z_MAX:
                    continue
                boxes.append(WorldBox(model.get('name', 'unnamed'),
                                      wx, wy, wz, yaw, sx, sy, sz))
    return boxes


def rasterize(boxes: List[WorldBox],
              resolution: float = GRID_RESOLUTION,
              center_x: float = GRID_CENTER_X,
              center_y: float = GRID_CENTER_Y,
              size_m: float = GRID_SIZE_M) -> TruthOccupancyGrid:
    n = int(round(size_m / resolution))
    origin_x = center_x - size_m / 2.0
    origin_y = center_y - size_m / 2.0
    data = np.zeros((n, n), dtype=np.uint8)
    ys, xs = np.mgrid[0:n, 0:n]
    cell_wx = origin_x + (xs + 0.5) * resolution
    cell_wy = origin_y + (ys + 0.5) * resolution
    for box in boxes:
        dx = cell_wx - box.x
        dy = cell_wy - box.y
        cos_y, sin_y = math.cos(-box.yaw), math.sin(-box.yaw)
        u = cos_y * dx - sin_y * dy
        v = sin_y * dx + cos_y * dy
        inside = (np.abs(u) <= box.size_x / 2.0) & (np.abs(v) <= box.size_y / 2.0)
        data[inside] = 1
    return TruthOccupancyGrid(resolution=resolution, origin_x=origin_x,
                              origin_y=origin_y, data=data)


def is_occupied(grid: TruthOccupancyGrid, wx: float, wy: float) -> bool:
    ix = int(math.floor((wx - grid.origin_x) / grid.resolution))
    iy = int(math.floor((wy - grid.origin_y) / grid.resolution))
    h, w = grid.data.shape
    if ix < 0 or ix >= w or iy < 0 or iy >= h:
        return False
    return bool(grid.data[iy, ix])


def line_of_sight(grid: TruthOccupancyGrid,
                  x0: float, y0: float, x1: float, y1: float) -> bool:
    """True if the segment (x0,y0)->(x1,y1) crosses no occupied cell."""
    dist = math.hypot(x1 - x0, y1 - y0)
    steps = max(1, int(dist / LOS_STEP_M))
    for i in range(steps + 1):
        u = i / steps
        if is_occupied(grid, x0 + (x1 - x0) * u, y0 + (y1 - y0) * u):
            return False
    return True


def save_grid(grid: TruthOccupancyGrid, path: Path) -> None:
    np.savez_compressed(path, data=grid.data,
                        resolution=grid.resolution,
                        origin_x=grid.origin_x, origin_y=grid.origin_y)


def load_grid(path: Path) -> TruthOccupancyGrid:
    with np.load(path) as f:
        return TruthOccupancyGrid(
            resolution=float(f['resolution']),
            origin_x=float(f['origin_x']),
            origin_y=float(f['origin_y']),
            data=f['data'].astype(np.uint8),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worlds-dir', default='', help='directory of .world files')
    parser.add_argument('--world', default='', help='single .world file')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    worlds: List[Path] = []
    if args.world:
        worlds.append(Path(args.world).expanduser())
    if args.worlds_dir:
        worlds.extend(sorted(Path(args.worlds_dir).expanduser().glob('*.world')))
    if not worlds:
        parser.error('need --world or --worlds-dir')

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    for world in worlds:
        boxes = parse_world_boxes(world.read_text())
        grid = rasterize(boxes)
        out = out_dir / f'{world.stem}.npz'
        save_grid(grid, out)
        print(f'[occupancy] {world.name}: {len(boxes)} boxes, '
              f'{int(grid.data.sum())} occupied cells -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 2.4: 运行测试确认通过**

Run: `python3 -m pytest src/thermal_robot/tests/test_phase0_evaluation.py -v`
Expected: 全部 PASS（13 个测试）

- [ ] **Step 2.5: 提交**

```bash
git add src/thermal_robot/scripts/world_occupancy.py src/thermal_robot/tests/test_phase0_evaluation.py
git commit -m "feat: add SDF world occupancy rasterizer with line-of-sight checks"
```

---

### 任务 3：为 6 个世界生成真值占据栅格

**Files:**
- Create: `src/thermal_robot/thermal_bringup/worlds/occupancy/*.npz`（6 个文件，由 CLI 生成）

- [ ] **Step 3.1: 生成栅格**

```bash
python3 src/thermal_robot/scripts/world_occupancy.py \
  --worlds-dir src/thermal_robot/thermal_bringup/worlds \
  --out-dir src/thermal_robot/thermal_bringup/worlds/occupancy
```

Expected: 打印 6 行 `[occupancy] ...`。`thermal_scene_nav.world`（开阔世界）boxes 数应为 0 或仅含边界围墙；障碍世界 boxes 数 > 0。

- [ ] **Step 3.2: 人工合理性检查**

```bash
python3 - <<'EOF'
from pathlib import Path
import numpy as np
for p in sorted(Path('src/thermal_robot/thermal_bringup/worlds/occupancy').glob('*.npz')):
    with np.load(p) as f:
        occ = int(f['data'].sum())
    print(f'{p.stem:40s} occupied_cells={occ}')
EOF
```

Expected: nav 世界占据格数最少；obstacle_field / corridor_rooms / mixed_rooms / zigzag_corridors / sparse_islands 均显著大于 0。如果某障碍世界为 0，停下排查 `parse_world_boxes`（可能该世界用了非 box 几何，需把该几何类型按外接矩形近似加入解析——先看 world 文件再改）。

- [ ] **Step 3.3: 提交**

```bash
git add src/thermal_robot/thermal_bringup/worlds/occupancy/
git commit -m "feat: add ground-truth occupancy grids for all evaluation worlds"
```

---

### 任务 4：scenario.apply_run_seed — 运行级 seed 扰动

**Files:**
- Modify: `src/thermal_robot/thermal_sensor_sim/thermal_sensor_sim/scenario.py`
- Modify: `src/thermal_robot/tests/test_phase0_evaluation.py`

- [ ] **Step 4.1: 追加失败测试**

在 `test_phase0_evaluation.py` 追加：

```python
class TestScenarioRunSeed(unittest.TestCase):
    def _scenario(self):
        from thermal_sensor_sim.scenario import default_config_b_scenario
        return default_config_b_scenario(3)

    def test_seed_zero_is_noop(self):
        from thermal_sensor_sim.scenario import apply_run_seed
        sc = self._scenario()
        x0 = sc.sources[0].world_x
        seed0 = sc.sources[0].seed
        apply_run_seed(sc, 0, jitter_std_m=0.5)
        self.assertEqual(sc.sources[0].world_x, x0)
        self.assertEqual(sc.sources[0].seed, seed0)

    def test_seed_changes_source_seed_deterministically(self):
        from thermal_sensor_sim.scenario import apply_run_seed
        sc1, sc2 = self._scenario(), self._scenario()
        apply_run_seed(sc1, 101, jitter_std_m=0.3)
        apply_run_seed(sc2, 101, jitter_std_m=0.3)
        for a, b in zip(sc1.sources, sc2.sources):
            self.assertEqual(a.seed, b.seed)
            self.assertAlmostEqual(a.world_x, b.world_x, places=9)
            self.assertAlmostEqual(a.world_y, b.world_y, places=9)

    def test_different_seeds_differ(self):
        from thermal_sensor_sim.scenario import apply_run_seed
        sc1, sc2 = self._scenario(), self._scenario()
        apply_run_seed(sc1, 101, jitter_std_m=0.3)
        apply_run_seed(sc2, 102, jitter_std_m=0.3)
        self.assertNotAlmostEqual(sc1.sources[0].world_x, sc2.sources[0].world_x, places=9)
        self.assertNotEqual(sc1.sources[0].seed, sc2.sources[0].seed)

    def test_zero_jitter_keeps_positions(self):
        from thermal_sensor_sim.scenario import apply_run_seed
        sc = self._scenario()
        x0, y0 = sc.sources[0].world_x, sc.sources[0].world_y
        apply_run_seed(sc, 101, jitter_std_m=0.0)
        self.assertEqual((sc.sources[0].world_x, sc.sources[0].world_y), (x0, y0))
        self.assertNotEqual(sc.sources[0].seed, 1)  # seed 仍被偏移
```

- [ ] **Step 4.2: 运行测试确认失败**

Run: `python3 -m pytest src/thermal_robot/tests/test_phase0_evaluation.py::TestScenarioRunSeed -v`
Expected: FAIL（`ImportError: cannot import name 'apply_run_seed'`）

- [ ] **Step 4.3: 实现 apply_run_seed**

在 `scenario.py` 顶部 import 区（`import math` 之后）加：

```python
import random
import zlib
```

在 `default_config_b_sources()` 函数定义之前加：

```python
def apply_run_seed(scenario: "ThermalScenario", run_seed: int,
                   jitter_std_m: float = 0.0,
                   amplitude_jitter_frac: float = 0.0) -> "ThermalScenario":
    """Apply a run-level seed: offsets per-source motion seeds and optionally
    jitters source positions/amplitudes. run_seed <= 0 is a no-op so default
    launches keep historical behavior."""
    if run_seed is None or int(run_seed) <= 0:
        return scenario
    run_seed = int(run_seed)
    for src in scenario.sources:
        rng = random.Random(run_seed * 1000003
                            + zlib.crc32(src.source_id.encode('utf-8')))
        src.seed = src.seed + run_seed * 1000
        if jitter_std_m > 0.0:
            src.world_x += rng.gauss(0.0, jitter_std_m)
            src.world_y += rng.gauss(0.0, jitter_std_m)
        if amplitude_jitter_frac > 0.0:
            src.amplitude = max(1.0, src.amplitude
                                * (1.0 + rng.gauss(0.0, amplitude_jitter_frac)))
    return scenario
```

- [ ] **Step 4.4: 运行测试确认通过**

Run: `python3 -m pytest src/thermal_robot/tests/test_phase0_evaluation.py -v`
Expected: 全部 PASS（17 个测试）。再跑既有测试确认无回归：
Run: `python3 -m pytest src/thermal_robot/tests/test_thermal_system.py -q`
Expected: `34 passed`

- [ ] **Step 4.5: 提交**

```bash
git add src/thermal_robot/thermal_sensor_sim/thermal_sensor_sim/scenario.py src/thermal_robot/tests/test_phase0_evaluation.py
git commit -m "feat: add run-level seed perturbation to thermal scenarios"
```

---

### 任务 5：static_five_sources.yaml 场景

**Files:**
- Create: `src/thermal_robot/thermal_bringup/config/scenarios/static_five_sources.yaml`
- Modify: `src/thermal_robot/tests/test_phase0_evaluation.py`

- [ ] **Step 5.1: 追加失败测试**

```python
class TestStaticFiveScenario(unittest.TestCase):
    def test_loads_five_static_sources(self):
        from thermal_sensor_sim.scenario import load_scenario_file
        path = WORKSPACE / 'src/thermal_robot/thermal_bringup/config/scenarios/static_five_sources.yaml'
        sc = load_scenario_file(str(path))
        self.assertEqual(len(sc.sources), 5)
        for src in sc.sources:
            self.assertEqual(src.motion, 'static')
            self.assertGreaterEqual(src.amplitude, 14.0)
```

- [ ] **Step 5.2: 运行测试确认失败**

Run: `python3 -m pytest src/thermal_robot/tests/test_phase0_evaluation.py::TestStaticFiveScenario -v`
Expected: FAIL（FileNotFoundError）

- [ ] **Step 5.3: 创建场景文件**

创建 `src/thermal_robot/thermal_bringup/config/scenarios/static_five_sources.yaml`：

```yaml
name: static_five_sources
robot:
  spawn: {x: -6.0, y: 0.0}
sensor:
  fov_x_m: 4.0
  fov_y_m: 3.0
sources:
  - id: S1_north
    xy: [-2.0, 4.0]
    amplitude: 30.0
    sigma_m: 1.0
    motion: static
  - id: S2_east
    xy: [5.5, 1.5]
    amplitude: 24.0
    sigma_m: 0.9
    motion: static
  - id: S3_southeast
    xy: [3.0, -5.0]
    amplitude: 20.0
    sigma_m: 0.9
    motion: static
  - id: S4_southwest
    xy: [-7.0, -5.5]
    amplitude: 18.0
    sigma_m: 0.85
    motion: static
  - id: S5_west_weak
    xy: [-10.0, 2.5]
    amplitude: 16.0
    sigma_m: 0.8
    motion: static
```

（位置设计：覆盖四个象限 + 西侧弱源，源间距均 > 6m，避免 NMS 合并；坐标都在 50m 真值栅格内。）

- [ ] **Step 5.4: 运行测试确认通过**

Run: `python3 -m pytest src/thermal_robot/tests/test_phase0_evaluation.py -v`
Expected: 全部 PASS（18 个测试）

- [ ] **Step 5.5: 提交**

```bash
git add src/thermal_robot/thermal_bringup/config/scenarios/static_five_sources.yaml src/thermal_robot/tests/test_phase0_evaluation.py
git commit -m "feat: add static five-source evaluation scenario"
```

---

### 任务 6：seed/strategy 参数接线（sensor_node、controller_node、launch）

**Files:**
- Modify: `src/thermal_robot/thermal_sensor_sim/thermal_sensor_sim/sensor_node.py`
- Modify: `src/thermal_robot/thermal_motion_controller/thermal_motion_controller/controller_node.py`
- Modify: `src/thermal_robot/thermal_bringup/launch/sim_nav_slam_launch.py`

本任务无纯单测（ROS 节点参数接线），验证方式为 `py_compile` + `colcon build` + 任务 9 的冒烟运行。

- [ ] **Step 6.1: sensor_node 接入场景 seed**

在 `sensor_node.py` 的 import 区，将：

```python
from thermal_sensor_sim.scenario import (
    SENSOR_FOV_X,
    SENSOR_FOV_Y,
    SPAWN_X,
    SPAWN_Y,
    SourceState,
    default_config_b_scenario,
    load_scenario_file,
)
```

改为（增加 `apply_run_seed`）：

```python
from thermal_sensor_sim.scenario import (
    SENSOR_FOV_X,
    SENSOR_FOV_Y,
    SPAWN_X,
    SPAWN_Y,
    SourceState,
    apply_run_seed,
    default_config_b_scenario,
    load_scenario_file,
)
```

在 `self.declare_parameter('scenario_file', '')` 之后追加两行：

```python
        self.declare_parameter('scenario_seed', 0)
        self.declare_parameter('scenario_jitter_std_m', 0.0)
```

在 `scenario_file = str(self.get_parameter('scenario_file').value or '')` 之后追加：

```python
        scenario_seed = int(self.get_parameter('scenario_seed').value)
        jitter_std = float(self.get_parameter('scenario_jitter_std_m').value)
```

将 `self._rng     = np.random.default_rng(seed)` 改为（scenario_seed > 0 时噪声也跟随运行 seed）：

```python
        noise_seed = scenario_seed if scenario_seed > 0 else seed
        self._rng     = np.random.default_rng(noise_seed)
```

在 `self._sources = self._scenario.sources` 这一行**之前**插入：

```python
        apply_run_seed(self._scenario, scenario_seed, jitter_std_m=jitter_std)
```

并把启动日志（`sensor_node v13 | ...` 那条 `get_logger().info`）的字符串末尾追加 `f' | run_seed={scenario_seed}'`。

- [ ] **Step 6.2: controller_node 接入 random_seed 与 strategy 参数**

在 `controller_node.py` 中找到这一行（约 301 行）：

```python
        self.declare_parameter('levy_post_confirm_step',     10.0)
```

在其后追加：

```python
        self.declare_parameter('random_seed',                0)
        self.declare_parameter('strategy',                   'full')
```

找到参数读取区中这一行（约 405 行）：

```python
        self._levy_mu      = float(g('levy_mu').value)
```

在该参数读取代码块的末尾（同一连续赋值块的最后一行之后，保持缩进一致）追加：

```python
        self._random_seed  = int(g('random_seed').value)
        self._strategy_mode = str(g('strategy').value or 'full')
        if self._strategy_mode not in ('full', 'frontier', 'levy'):
            self.get_logger().warn(f'unknown strategy={self._strategy_mode}, using full')
            self._strategy_mode = 'full'
        if self._random_seed > 0:
            random.seed(self._random_seed)
            np.random.seed(self._random_seed % (2**31))
        self.get_logger().info(
            f'strategy={self._strategy_mode} random_seed={self._random_seed}')
```

- [ ] **Step 6.3: controller_node 策略门控（基线挂架）**

三处修改，全部基于已核实的现有代码：

(a) `_refresh_frontier`（约 1312 行）。在函数体第一行（`if not force and ...` 之前）插入：

```python
        if self._strategy_mode == 'levy':
            if force or (now - self._frontier_last_upd) >= self._frontier_upd:
                self._do_levy_jump(now)
            return
```

(b) `_do_levy_jump`（约 1337 行）。在函数体开头（`step=self._levy_step()` 之前）插入：

```python
        if self._strategy_mode == 'frontier':
            target = self._source_seek_selector.select_frontier(
                self._strategy_context(now), min_d=2.0, max_d=16.0, dist_sigma=8.0)
            if target is not None:
                fx, fy = target.xy
                self._frontier_target = (fx, fy)
                self._frontier_last_upd = now
                self._search_rounds = 0
                self.get_logger().info(
                    f'[FRONTIER/baseline-fallback] →({fx:.1f},{fy:.1f})')
                return
            # 地图尚无可用前沿时仍退回 Lévy，避免机器人冻结
```

(c) `_coarse_waypoint`（约 1486 行）。将函数开头的：

```python
    def _coarse_waypoint(self):
        target = self._source_seek_selector.select_coarse_waypoint(
            self._strategy_context(),
            min_d=self._survey_wp_min_d,
            max_d=self._survey_wp_max_d,
        )
```

改为：

```python
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
        else:
            target = self._source_seek_selector.select_coarse_waypoint(
                self._strategy_context(),
                min_d=self._survey_wp_min_d,
                max_d=self._survey_wp_max_d,
            )
```

注意：`select_frontier` 可能返回 `None`，`_coarse_waypoint` 现有代码已处理 `target is None` 分支（返回 None → 调用方走 nav2_preferred 默认路径），无需额外改动。`select_levy_jump` 总是返回目标。FINE 状态（梯度上升/确认）三种策略下完全一致——基线只替换**探索目标选择**，保证对比公平。

- [ ] **Step 6.4: launch 文件加 run_seed / strategy / jitter 参数**

在 `sim_nav_slam_launch.py` 的 import 区追加：

```python
from launch_ros.parameter_descriptions import ParameterValue
```

在 `world_file = LaunchConfiguration('world_file', default=default_world_file)` 之后追加：

```python
    run_seed = LaunchConfiguration('run_seed', default='0')
    strategy = LaunchConfiguration('strategy', default='full')
    scenario_jitter = LaunchConfiguration('scenario_jitter_std_m', default='0.0')
```

在 `DeclareLaunchArgument('world_file', default_value=default_world_file),` 之后追加：

```python
        DeclareLaunchArgument('run_seed', default_value='0'),
        DeclareLaunchArgument('strategy', default_value='full'),
        DeclareLaunchArgument('scenario_jitter_std_m', default_value='0.0'),
```

将 sensor_node 的 Node 参数：

```python
                parameters=[params_file, {
                    'use_sim_time': use_sim_t,
                    'scenario_file': scenario_file,
                }],
```

改为：

```python
                parameters=[params_file, {
                    'use_sim_time': use_sim_t,
                    'scenario_file': scenario_file,
                    'scenario_seed': ParameterValue(run_seed, value_type=int),
                    'scenario_jitter_std_m': ParameterValue(scenario_jitter, value_type=float),
                }],
```

将 controller_node 的 Node 参数：

```python
                parameters=[params_file, {'use_sim_time': use_sim_t}],
```

（注意是 `executable='controller_node'` 那个 Node，不要改错到其他节点）改为：

```python
                parameters=[params_file, {
                    'use_sim_time': use_sim_t,
                    'random_seed': ParameterValue(run_seed, value_type=int),
                    'strategy': strategy,
                }],
```

- [ ] **Step 6.5: 验证编译与构建**

```bash
python3 -m py_compile \
  src/thermal_robot/thermal_sensor_sim/thermal_sensor_sim/sensor_node.py \
  src/thermal_robot/thermal_motion_controller/thermal_motion_controller/controller_node.py \
  src/thermal_robot/thermal_bringup/launch/sim_nav_slam_launch.py
source /opt/ros/humble/setup.bash && source install/setup.bash
colcon build --packages-select thermal_sensor_sim thermal_motion_controller thermal_bringup
source install/setup.bash
ros2 launch thermal_bringup sim_nav_slam_launch.py --show-args
```

Expected: py_compile 无输出；构建 3 包通过；`--show-args` 列出 `run_seed`、`strategy`、`scenario_jitter_std_m`。

- [ ] **Step 6.6: 既有测试回归**

Run: `python3 -m pytest src/thermal_robot/tests/ -q`
Expected: 全部通过（34 + 18 个）

- [ ] **Step 6.7: 提交**

```bash
git add src/thermal_robot/thermal_sensor_sim/thermal_sensor_sim/sensor_node.py \
        src/thermal_robot/thermal_motion_controller/thermal_motion_controller/controller_node.py \
        src/thermal_robot/thermal_bringup/launch/sim_nav_slam_launch.py
git commit -m "feat: wire run_seed and strategy baseline parameters through launch"
```

---

### 任务 7：attribution.py — 逐真值源失败归因

**Files:**
- Create: `src/thermal_robot/scripts/attribution.py`
- Modify: `src/thermal_robot/tests/test_phase0_evaluation.py`

归因四分类的判定优先级（spec §5.3）：

1. `not_confirmed`（到了没确认）：无遮挡可见累计时长 ≥ `min_visible_s`（默认 1.0s）但未匹配确认；
2. `occluded`（被挡住）：可见时长不足，且存在"源活跃 + 在 FOV 内 + 视线被挡"事件；
3. `timing_missed`（时机错过）：可见时长不足、无遮挡事件，但源**不活跃时**进过 FOV；
4. `not_reached`（没走到）：其余情况（从未有效进入视场）。

- [ ] **Step 7.1: 追加失败测试**

```python
class TestAttribution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.attr = _load_script('attribution')
        cls.wo = _load_script('world_occupancy')

    @staticmethod
    def _traj(n=50, wx=0.0, wy=0.0, yaw=0.0, dt=0.1):
        return [{'t': i * dt, 'wx': wx, 'wy': wy, 'yaw': yaw} for i in range(n)]

    @staticmethod
    def _truth(source_id, x, y, n=50, dt=0.1, active=True, amplitude=20.0, sigma=1.0):
        status = 'truth_active' if active else 'truth_inactive'
        return [{'t': i * dt, 'id': source_id, 'status': status, 'x': x, 'y': y,
                 'strength': amplitude, 'sigma': sigma} for i in range(n)]

    def test_visible_unmatched_is_not_confirmed(self):
        # 源在机器人正前方 1.5m, FOV 4x3 内, 无遮挡, 持续 5s, 未匹配
        result = self.attr.compute_attribution(
            traj_rows=self._traj(),
            truth_rows=self._truth('A', 1.5, 0.0),
            matched_truth_ids=set(),
            estimate_rows=[],
            fov_x=4.0, fov_y=3.0, occupancy=None)
        rec = result['per_source']['A']
        self.assertTrue(rec['ever_in_fov'])
        self.assertGreaterEqual(rec['visible_total_s'], 1.0)
        self.assertEqual(rec['failure_class'], 'not_confirmed')

    def test_far_source_is_not_reached(self):
        result = self.attr.compute_attribution(
            traj_rows=self._traj(),
            truth_rows=self._truth('B', 15.0, 15.0),
            matched_truth_ids=set(),
            estimate_rows=[],
            fov_x=4.0, fov_y=3.0, occupancy=None)
        rec = result['per_source']['B']
        self.assertFalse(rec['ever_in_fov'])
        self.assertEqual(rec['failure_class'], 'not_reached')
        self.assertAlmostEqual(rec['min_center_dist_m'], math.hypot(15, 15), places=2)

    def test_wall_between_is_occluded(self):
        boxes = self.wo.parse_world_boxes(SAMPLE_SDF)  # wall_east: x∈[1,3], y∈[-0.5,0.5]
        grid = self.wo.rasterize(boxes)
        # 机器人在原点, 源在 (3.6, 0): 在 FOV 内(x<=2? 否) — 改近: 源 (1.8,0) 在墙体内侧不行
        # 用旋转后的机器人: 站 (-0.5,0) 朝东, 源在 (3.4,0), FOV 半长 2m → fx=3.9 超出
        # 简化: FOV 放大到 10x6 验证遮挡逻辑本身
        result = self.attr.compute_attribution(
            traj_rows=self._traj(wx=-0.5, wy=0.0, yaw=0.0),
            truth_rows=self._truth('C', 4.0, 0.0),
            matched_truth_ids=set(),
            estimate_rows=[],
            fov_x=10.0, fov_y=6.0, occupancy=grid)
        rec = result['per_source']['C']
        self.assertEqual(rec['failure_class'], 'occluded')
        self.assertGreater(rec['occluded_events'], 0)

    def test_inactive_in_fov_is_timing_missed(self):
        result = self.attr.compute_attribution(
            traj_rows=self._traj(),
            truth_rows=self._truth('D', 1.5, 0.0, active=False),
            matched_truth_ids=set(),
            estimate_rows=[],
            fov_x=4.0, fov_y=3.0, occupancy=None)
        rec = result['per_source']['D']
        self.assertEqual(rec['failure_class'], 'timing_missed')

    def test_matched_source_has_no_failure_class(self):
        result = self.attr.compute_attribution(
            traj_rows=self._traj(),
            truth_rows=self._truth('A', 1.5, 0.0),
            matched_truth_ids={'A'},
            estimate_rows=[{'t': 2.0, 'id': 'src_1', 'status': 'confirmed',
                            'x': 1.4, 'y': 0.1}],
            fov_x=4.0, fov_y=3.0, occupancy=None)
        rec = result['per_source']['A']
        self.assertIsNone(rec['failure_class'])
        self.assertEqual(result['failure_counts']['not_confirmed'], 0)

    def test_visibility_windows_recorded(self):
        result = self.attr.compute_attribution(
            traj_rows=self._traj(n=100),
            truth_rows=self._truth('A', 1.5, 0.0, n=100),
            matched_truth_ids=set(),
            estimate_rows=[],
            fov_x=4.0, fov_y=3.0, occupancy=None)
        windows = result['per_source']['A']['visible_windows']
        self.assertEqual(len(windows), 1)
        self.assertAlmostEqual(windows[0]['duration_s'], 9.9, delta=0.3)
```

- [ ] **Step 7.2: 运行测试确认失败**

Run: `python3 -m pytest src/thermal_robot/tests/test_phase0_evaluation.py::TestAttribution -v`
Expected: ERROR（attribution.py 不存在）

- [ ] **Step 7.3: 实现 attribution.py**

创建 `src/thermal_robot/scripts/attribution.py`，完整内容：

```python
#!/usr/bin/env python3
"""Per-truth-source failure attribution for a collector run directory.

输入(均为采集器既有产物): trajectory.csv, thermal_sources_truth.csv,
source_estimates.csv, source_summary.json(取已匹配的 truth_id)。
可选: 世界真值占据栅格 npz(由 world_occupancy.py 生成), 提供遮挡判定。

输出 attribution.json:
  per_source: 每个真值源的最小视距/可见窗口/遮挡事件/tracker 轨迹/失败分类
  failure_counts: 四类失败计数 (not_reached / occluded / timing_missed / not_confirmed)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

SCRIPTS_DIR = Path(__file__).resolve().parent

_wo_spec = importlib.util.spec_from_file_location(
    'world_occupancy', SCRIPTS_DIR / 'world_occupancy.py')
world_occupancy = importlib.util.module_from_spec(_wo_spec)
sys.modules['world_occupancy'] = world_occupancy
_wo_spec.loader.exec_module(world_occupancy)

FAILURE_CLASSES = ('not_reached', 'occluded', 'timing_missed', 'not_confirmed')
DEFAULT_MIN_VISIBLE_S = 1.0
TIME_MATCH_TOLERANCE_S = 0.6
WINDOW_GAP_S = 0.5


def _read_csv(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _f(row: Dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _nearest_record(records: Sequence[Dict], t: float) -> Optional[Dict]:
    """records 按 t 升序; 返回时间最近且 |dt|<=容差 的记录。"""
    if not records:
        return None
    lo, hi = 0, len(records) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if _f(records[mid], 't') < t:
            lo = mid + 1
        else:
            hi = mid
    best = records[lo]
    if lo > 0 and abs(_f(records[lo - 1], 't') - t) < abs(_f(best, 't') - t):
        best = records[lo - 1]
    if abs(_f(best, 't') - t) > TIME_MATCH_TOLERANCE_S:
        return None
    return best


def _group_windows(samples: List[Dict]) -> List[Dict]:
    """samples: [{'t','d','strength'}] 连续(间隔<=WINDOW_GAP_S)分组为窗口。"""
    windows: List[Dict] = []
    cur: List[Dict] = []
    for s in samples:
        if cur and s['t'] - cur[-1]['t'] > WINDOW_GAP_S:
            windows.append(cur)
            cur = []
        cur.append(s)
    if cur:
        windows.append(cur)
    out = []
    for group in windows:
        out.append({
            't_start': round(group[0]['t'], 2),
            't_end': round(group[-1]['t'], 2),
            'duration_s': round(group[-1]['t'] - group[0]['t'], 2),
            'min_dist_m': round(min(g['d'] for g in group), 2),
            'max_strength': round(max(g['strength'] for g in group), 2),
        })
    return out


def _count_events(flags: List[Dict]) -> int:
    """与 _group_windows 相同的分组逻辑, 只返回组数。"""
    return len(_group_windows(flags))


def compute_attribution(
    traj_rows: Sequence[Dict],
    truth_rows: Sequence[Dict],
    matched_truth_ids: Set[str],
    estimate_rows: Sequence[Dict],
    fov_x: float = 4.0,
    fov_y: float = 3.0,
    occupancy=None,
    min_visible_s: float = DEFAULT_MIN_VISIBLE_S,
) -> Dict:
    truth_by_id: Dict[str, List[Dict]] = {}
    for row in truth_rows:
        truth_by_id.setdefault(str(row['id']), []).append(row)
    for records in truth_by_id.values():
        records.sort(key=lambda r: _f(r, 't'))

    traj = sorted(traj_rows, key=lambda r: _f(r, 't'))

    per_source: Dict[str, Dict] = {}
    failure_counts = {key: 0 for key in FAILURE_CLASSES}

    for source_id, records in truth_by_id.items():
        visible: List[Dict] = []
        occluded: List[Dict] = []
        fov_inactive: List[Dict] = []
        min_dist = float('inf')
        ever_in_fov = False
        active_samples = 0
        total_samples = 0

        for pose in traj:
            t = _f(pose, 't')
            rec = _nearest_record(records, t)
            if rec is None:
                continue
            total_samples += 1
            active = rec.get('status') == 'truth_active'
            sx, sy = _f(rec, 'x'), _f(rec, 'y')
            wx, wy, yaw = _f(pose, 'wx'), _f(pose, 'wy'), _f(pose, 'yaw')
            dx, dy = sx - wx, sy - wy
            d = math.hypot(dx, dy)
            cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
            fx = cos_y * dx - sin_y * dy
            fy = sin_y * dx + cos_y * dy
            in_fov = abs(fx) <= fov_x / 2.0 and abs(fy) <= fov_y / 2.0
            if active:
                active_samples += 1
                min_dist = min(min_dist, d)
            if not in_fov:
                continue
            ever_in_fov = True
            if not active:
                fov_inactive.append({'t': t, 'd': d, 'strength': 0.0})
                continue
            los = (occupancy is None
                   or world_occupancy.line_of_sight(occupancy, wx, wy, sx, sy))
            sample = {'t': t, 'd': d, 'strength': _f(rec, 'strength')}
            if los:
                visible.append(sample)
            else:
                occluded.append(sample)

        windows = _group_windows(visible)
        visible_total = sum(w['duration_s'] for w in windows)
        # 真值 10Hz 记录的活跃时长估计
        dt_truth = 0.1
        if len(records) >= 2:
            span = _f(records[-1], 't') - _f(records[0], 't')
            dt_truth = max(1e-3, span / max(1, len(records) - 1))
        active_total = sum(
            1 for r in records if r.get('status') == 'truth_active') * dt_truth

        # tracker 视角: 距该源任意时刻位置 2.5m 内的估计记录
        first_detect_t = None
        statuses_seen: Set[str] = set()
        for est in estimate_rows:
            rec = _nearest_record(records, _f(est, 't'))
            if rec is None:
                continue
            d = math.hypot(_f(est, 'x') - _f(rec, 'x'), _f(est, 'y') - _f(rec, 'y'))
            if d <= 2.5:
                statuses_seen.add(str(est.get('status')))
                if first_detect_t is None or _f(est, 't') < first_detect_t:
                    first_detect_t = _f(est, 't')

        matched = source_id in matched_truth_ids
        if matched:
            failure = None
        elif visible_total >= min_visible_s:
            failure = 'not_confirmed'
        elif occluded:
            failure = 'occluded'
        elif fov_inactive:
            failure = 'timing_missed'
        else:
            failure = 'not_reached'
        if failure is not None:
            failure_counts[failure] += 1

        per_source[source_id] = {
            'matched': matched,
            'failure_class': failure,
            'ever_in_fov': ever_in_fov,
            'min_center_dist_m': round(min_dist, 3) if math.isfinite(min_dist) else None,
            'visible_total_s': round(visible_total, 2),
            'visible_windows': windows,
            'occluded_events': _count_events(occluded),
            'fov_inactive_events': _count_events(fov_inactive),
            'active_total_s': round(active_total, 2),
            'visible_over_active_ratio': round(
                visible_total / active_total, 3) if active_total > 0 else None,
            'tracker_first_detect_t': first_detect_t,
            'tracker_statuses_seen': sorted(statuses_seen),
        }

    return {
        'per_source': per_source,
        'failure_counts': failure_counts,
        'n_truth_sources': len(per_source),
        'n_matched': sum(1 for r in per_source.values() if r['matched']),
    }


def compute_for_run_dir(run_dir: Path, occupancy_path: Optional[Path],
                        fov_x: float, fov_y: float) -> Dict:
    summary = {}
    summary_path = run_dir / 'source_summary.json'
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    matched_ids = {
        str(m['truth_id']) for m in summary.get('localization_errors_m', [])
    }
    occupancy = None
    if occupancy_path and Path(occupancy_path).exists():
        occupancy = world_occupancy.load_grid(Path(occupancy_path))
    return compute_attribution(
        traj_rows=_read_csv(run_dir / 'trajectory.csv'),
        truth_rows=_read_csv(run_dir / 'thermal_sources_truth.csv'),
        matched_truth_ids=matched_ids,
        estimate_rows=_read_csv(run_dir / 'source_estimates.csv'),
        fov_x=fov_x, fov_y=fov_y, occupancy=occupancy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('--occupancy', default='', help='truth occupancy npz')
    parser.add_argument('--fov-x', type=float, default=4.0)
    parser.add_argument('--fov-y', type=float, default=3.0)
    parser.add_argument('--out', default='attribution.json')
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    occ = Path(args.occupancy).expanduser() if args.occupancy else None
    result = compute_for_run_dir(run_dir, occ, args.fov_x, args.fov_y)
    out = run_dir / args.out
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result['failure_counts'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 7.4: 运行测试确认通过**

Run: `python3 -m pytest src/thermal_robot/tests/test_phase0_evaluation.py -v`
Expected: 全部 PASS（24 个测试）

- [ ] **Step 7.5: 提交**

```bash
git add src/thermal_robot/scripts/attribution.py src/thermal_robot/tests/test_phase0_evaluation.py
git commit -m "feat: add per-truth-source failure attribution tool"
```

---

### 任务 8：run_multiscenario_matrix.py 重写

**Files:**
- Modify(重写): `src/thermal_robot/scripts/run_multiscenario_matrix.py`
- Modify: `src/thermal_robot/tests/test_phase0_evaluation.py`

新能力：`phase0` preset（4 世界 × 6 场景）、`--seeds` 多 seed、`--worlds/--cases` 筛选、`--strategy` 基线切换、`--jitter` 场景扰动、每 run 自动归因、`matrix_stats` 聚合 + `matrix_report.md`、`--health-only` 模式（用 `nav2_health_check.py` 替代采集器，验证障碍世界 Nav2 健康）。保留旧 preset（representative/extended/variable/full）与 `--case` 以兼容历史命令。

- [ ] **Step 8.1: 追加失败测试**

```python
class TestMatrixRunnerConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = _load_script('run_multiscenario_matrix')

    def test_phase0_grid_is_4x6(self):
        cases = self.runner.phase0_cases()
        self.assertEqual(len(cases), 24)
        worlds = {c.world.name for c in cases}
        scenarios = {c.scenario.name for c in cases}
        self.assertEqual(len(worlds), 4)
        self.assertEqual(len(scenarios), 6)

    def test_phase0_case_names_use_class_keys(self):
        names = {c.name for c in self.runner.phase0_cases()}
        self.assertIn('open__static2', names)
        self.assertIn('boxes__dyn5', names)
        self.assertIn('walls__birthdeath', names)
        self.assertIn('mixed__static5', names)

    def test_filter_cases_by_world_and_case(self):
        cases = self.runner.phase0_cases()
        only_open = self.runner.filter_cases(cases, worlds='open', case_names='')
        self.assertEqual(len(only_open), 6)
        one = self.runner.filter_cases(cases, worlds='', case_names='open__static2')
        self.assertEqual(len(one), 1)
        two_worlds = self.runner.filter_cases(cases, worlds='open,boxes', case_names='')
        self.assertEqual(len(two_worlds), 12)

    def test_parse_seeds(self):
        self.assertEqual(self.runner.parse_seeds('101,102,103'), [101, 102, 103])
        self.assertEqual(self.runner.parse_seeds(' 7 '), [7])

    def test_occupancy_path_for_world(self):
        cases = self.runner.phase0_cases()
        boxes_case = next(c for c in cases if c.name.startswith('boxes__'))
        occ = self.runner.occupancy_path_for_world(boxes_case.world)
        self.assertTrue(str(occ).endswith('thermal_scene_obstacle_field.npz'))

    def test_legacy_presets_still_present(self):
        self.assertGreaterEqual(len(self.runner.REPRESENTATIVE_CASES), 4)
        self.assertGreaterEqual(len(self.runner.VARIABLE_SOURCE_CASES), 3)
```

- [ ] **Step 8.2: 运行测试确认失败**

Run: `python3 -m pytest src/thermal_robot/tests/test_phase0_evaluation.py::TestMatrixRunnerConfig -v`
Expected: FAIL（`phase0_cases`/`filter_cases`/`parse_seeds` 不存在）

- [ ] **Step 8.3: 重写 run_multiscenario_matrix.py**

用以下完整内容**替换**整个文件（保留的旧符号：`MatrixCase`、`REPRESENTATIVE_CASES`、`EXTENDED_CASES`、`VARIABLE_SOURCE_CASES`、`FULL_WORLDS`、`FULL_SCENARIOS`、`_full_cases`、`_parse_case`，既有测试 T-PY23 依赖它们）：

```python
#!/usr/bin/env python3
"""Run multi-world, multi-thermal-scenario closed-loop simulation matrices.

阶段0评测地基:
  - phase0 preset: 4 世界类 x 6 源配置 x N seeds
  - --seeds: 多 seed 重复, 输出均值±标准差 (matrix_stats)
  - --worlds/--cases: 子集筛选
  - --strategy: full | frontier | levy 基线切换
  - 每 run 自动生成 attribution.json (含遮挡归因, 需真值占据栅格)
  - matrix_summary.json + matrix_report.md
  - --health-only: 用 nav2_health_check 替代采集器, 验证 Nav2 健康
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence


WORKSPACE = Path(__file__).resolve().parents[3]
SCRIPTS = WORKSPACE / "src/thermal_robot/scripts"
BRINGUP = WORKSPACE / "src/thermal_robot/thermal_bringup"
WORLDS = BRINGUP / "worlds"
OCCUPANCY_DIR = WORLDS / "occupancy"
SCENARIOS = BRINGUP / "config/scenarios"
CONFIG_B = BRINGUP / "config/config_b_sources.yaml"
COLLECTOR = SCRIPTS / "collect_sim_data.py"
ATTRIBUTION = SCRIPTS / "attribution.py"
HEALTH_CHECK = SCRIPTS / "nav2_health_check.py"


def _load_matrix_stats():
    spec = importlib.util.spec_from_file_location("matrix_stats", SCRIPTS / "matrix_stats.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class MatrixCase:
    name: str
    world: Path
    scenario: Path


# ── phase0 网格: 世界类 x 源配置 ─────────────────────────────────────────────
PHASE0_WORLD_CLASSES = {
    "open": WORLDS / "thermal_scene_nav.world",
    "boxes": WORLDS / "thermal_scene_obstacle_field.world",
    "walls": WORLDS / "thermal_scene_corridor_rooms.world",
    "mixed": WORLDS / "thermal_scene_mixed_rooms.world",
}
PHASE0_SCENARIO_CLASSES = {
    "static2": SCENARIOS / "static_two_sources.yaml",
    "static3": CONFIG_B,
    "static5": SCENARIOS / "static_five_sources.yaml",
    "dyn4": SCENARIOS / "dynamic_four_sources.yaml",
    "dyn5": SCENARIOS / "dynamic_five_sources.yaml",
    "birthdeath": SCENARIOS / "dynamic_appear_disappear_sources.yaml",
}


def phase0_cases() -> List[MatrixCase]:
    return [
        MatrixCase(f"{wkey}__{skey}", world, scenario)
        for wkey, world in PHASE0_WORLD_CLASSES.items()
        for skey, scenario in PHASE0_SCENARIO_CLASSES.items()
    ]


# ── 历史 preset(兼容保留) ────────────────────────────────────────────────────
REPRESENTATIVE_CASES = [
    MatrixCase("open_config_b", WORLDS / "thermal_scene_nav.world", CONFIG_B),
    MatrixCase("obstacle_linear", WORLDS / "thermal_scene_obstacle_field.world", SCENARIOS / "dynamic_linear_sources.yaml"),
    MatrixCase("corridor_appear", WORLDS / "thermal_scene_corridor_rooms.world", SCENARIOS / "dynamic_appear_disappear_sources.yaml"),
    MatrixCase("mixed_waypoint", WORLDS / "thermal_scene_mixed_rooms.world", SCENARIOS / "dynamic_waypoint_random_sources.yaml"),
]

EXTENDED_CASES = [
    *REPRESENTATIVE_CASES,
    MatrixCase("zigzag_circular", WORLDS / "thermal_scene_zigzag_corridors.world", SCENARIOS / "dynamic_circular_sources.yaml"),
    MatrixCase("islands_static_offset", WORLDS / "thermal_scene_sparse_islands.world", SCENARIOS / "static_offset_sources.yaml"),
]

VARIABLE_SOURCE_CASES = [
    MatrixCase("open_static_2src", WORLDS / "thermal_scene_nav.world", SCENARIOS / "static_two_sources.yaml"),
    MatrixCase("mixed_dynamic_4src", WORLDS / "thermal_scene_mixed_rooms.world", SCENARIOS / "dynamic_four_sources.yaml"),
    MatrixCase("zigzag_dynamic_5src", WORLDS / "thermal_scene_zigzag_corridors.world", SCENARIOS / "dynamic_five_sources.yaml"),
]

FULL_WORLDS = [
    WORLDS / "thermal_scene_nav.world",
    WORLDS / "thermal_scene_obstacle_field.world",
    WORLDS / "thermal_scene_corridor_rooms.world",
    WORLDS / "thermal_scene_mixed_rooms.world",
    WORLDS / "thermal_scene_zigzag_corridors.world",
    WORLDS / "thermal_scene_sparse_islands.world",
]
FULL_SCENARIOS = [
    CONFIG_B,
    SCENARIOS / "static_offset_sources.yaml",
    SCENARIOS / "dynamic_linear_sources.yaml",
    SCENARIOS / "dynamic_circular_sources.yaml",
    SCENARIOS / "dynamic_appear_disappear_sources.yaml",
    SCENARIOS / "dynamic_waypoint_random_sources.yaml",
    SCENARIOS / "static_two_sources.yaml",
    SCENARIOS / "dynamic_four_sources.yaml",
    SCENARIOS / "dynamic_five_sources.yaml",
]


def _case_name_from_paths(world: Path, scenario: Path) -> str:
    world_name = world.stem.replace("thermal_scene_", "")
    scenario_name = scenario.stem.replace("_sources", "")
    return f"{world_name}__{scenario_name}"


def _full_cases() -> List[MatrixCase]:
    return [
        MatrixCase(_case_name_from_paths(world, scenario), world, scenario)
        for world in FULL_WORLDS
        for scenario in FULL_SCENARIOS
    ]


def _parse_case(raw: str) -> MatrixCase:
    parts = raw.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--case must be name:world_file:scenario_file")
    name, world, scenario = parts
    return MatrixCase(name=name, world=Path(world).expanduser(), scenario=Path(scenario).expanduser())


# ── 筛选与 seed 解析 ─────────────────────────────────────────────────────────
def parse_seeds(raw: str) -> List[int]:
    return [int(v.strip()) for v in raw.split(",") if v.strip()]


def filter_cases(cases: Sequence[MatrixCase], worlds: str, case_names: str) -> List[MatrixCase]:
    out = list(cases)
    if worlds:
        keys = {w.strip() for w in worlds.split(",") if w.strip()}
        out = [c for c in out if c.name.split("__")[0] in keys]
    if case_names:
        keys = {n.strip() for n in case_names.split(",") if n.strip()}
        out = [c for c in out if c.name in keys]
    return out


def occupancy_path_for_world(world: Path) -> Path:
    return OCCUPANCY_DIR / f"{world.stem}.npz"


# ── 进程工具 ─────────────────────────────────────────────────────────────────
def _sourced_command(command: str) -> List[str]:
    return [
        "bash",
        "-lc",
        "source /opt/ros/humble/setup.bash && "
        f"source {shlex.quote(str(WORKSPACE / 'install/setup.bash'))} && "
        f"{command}",
    ]


def _ensure_files(cases: Sequence[MatrixCase]) -> None:
    required = [COLLECTOR, ATTRIBUTION, HEALTH_CHECK, WORKSPACE / "install/setup.bash"]
    for case in cases:
        required.extend([case.world, case.scenario])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n  " + "\n  ".join(missing))


def _stop_process_group(proc: subprocess.Popen, grace_s: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    pgid = os.getpgid(proc.pid)
    os.killpg(pgid, signal.SIGINT)
    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    os.killpg(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    os.killpg(pgid, signal.SIGKILL)
    proc.wait(timeout=5.0)


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


# ── 单次运行 ─────────────────────────────────────────────────────────────────
def run_case(
    case: MatrixCase,
    run_dir: Path,
    duration_s: float,
    warmup_s: float,
    domain_id: int,
    min_recall: float,
    seed: int,
    strategy: str,
    jitter_std_m: float,
    health_only: bool,
) -> Dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "ros_logs"
    log_dir.mkdir(exist_ok=True)
    case_home = run_dir / "home"
    gazebo_log_dir = run_dir / "gazebo_logs"
    case_home.mkdir(exist_ok=True)
    gazebo_log_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["ROS_LOG_DIR"] = str(log_dir)
    env["ROS_DOMAIN_ID"] = str(domain_id)
    gazebo_port = 11345 + int(domain_id)
    env["GAZEBO_MASTER_URI"] = f"http://127.0.0.1:{gazebo_port}"
    env["GAZEBO_LOG_PATH"] = str(gazebo_log_dir)
    model_paths = [
        "/home/hanchen/.gazebo/models",
        "/usr/share/gazebo-11/models",
        env.get("GAZEBO_MODEL_PATH", ""),
    ]
    env["GAZEBO_MODEL_PATH"] = ":".join(path for path in model_paths if path)
    env["GAZEBO_MODEL_DATABASE_URI"] = ""
    env["HOME"] = str(case_home)
    env["PYTHONUNBUFFERED"] = "1"

    launch_cmd = " ".join([
        "ros2", "launch", "thermal_bringup", "sim_nav_slam_launch.py",
        "use_rviz:=false",
        "use_gzclient:=false",
        f"world_file:={shlex.quote(str(case.world))}",
        f"scenario_file:={shlex.quote(str(case.scenario))}",
        f"run_seed:={int(seed)}",
        f"strategy:={strategy}",
        f"scenario_jitter_std_m:={jitter_std_m}",
    ])
    if health_only:
        payload_cmd = " ".join([
            "python3", shlex.quote(str(HEALTH_CHECK)),
            "--timeout", str(max(20.0, duration_s)), "--json",
        ])
    else:
        payload_cmd = " ".join([
            "python3", shlex.quote(str(COLLECTOR)),
            "--out-dir", shlex.quote(str(run_dir)),
            "--duration", str(duration_s),
        ])

    launch_log_path = run_dir / "launch.log"
    payload_log_path = run_dir / ("health.log" if health_only else "collector.log")
    started_at = time.time()
    with open(launch_log_path, "w") as launch_log:
        launch_proc = subprocess.Popen(
            _sourced_command(launch_cmd),
            cwd=str(WORKSPACE),
            env=env,
            stdout=launch_log,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    payload_returncode = None
    payload_stdout = ""
    try:
        time.sleep(warmup_s)
        with open(payload_log_path, "w") as payload_log:
            payload_run = subprocess.run(
                _sourced_command(payload_cmd),
                cwd=str(WORKSPACE),
                env=env,
                stdout=subprocess.PIPE if health_only else payload_log,
                stderr=subprocess.STDOUT,
                timeout=duration_s + 60.0,
                check=False,
            )
            payload_returncode = payload_run.returncode
            if health_only:
                payload_stdout = (payload_run.stdout or b"").decode("utf-8", "replace")
                payload_log.write(payload_stdout)
    finally:
        _stop_process_group(launch_proc)
        time.sleep(2.0)

    result: Dict = {
        "name": case.name,
        "seed": int(seed),
        "strategy": strategy,
        "world": str(case.world),
        "scenario": str(case.scenario),
        "run_dir": str(run_dir),
        "ros_domain_id": domain_id,
        "elapsed_wall_s": round(time.time() - started_at, 3),
    }

    if health_only:
        health: Dict = {}
        try:
            start = payload_stdout.find("{")
            if start >= 0:
                health = json.loads(payload_stdout[start:])
        except json.JSONDecodeError:
            health = {}
        (run_dir / "nav2_health.json").write_text(json.dumps(health, indent=2))
        result["passed"] = bool(health.get("ok")) and payload_returncode == 0
        result["nav2_health"] = health
        return result

    # 归因(纯后处理, 不需要 ROS 环境)
    occupancy = occupancy_path_for_world(case.world)
    attr_cmd = [sys.executable, str(ATTRIBUTION), str(run_dir)]
    if occupancy.exists():
        attr_cmd += ["--occupancy", str(occupancy)]
    subprocess.run(attr_cmd, cwd=str(WORKSPACE), check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    metadata = _load_json(run_dir / "metadata.json")
    summary = _load_json(run_dir / "source_summary.json")
    attribution = _load_json(run_dir / "attribution.json")
    counts = metadata.get("counts", {})
    required_counts = ["thermal_stats", "field_stats", "map_stats", "grad_stats", "truth_sources", "cmd_vel"]
    missing_counts = [key for key in required_counts if int(counts.get(key, 0) or 0) <= 0]
    recall = float(summary.get("source_recall", 0.0) or 0.0)
    duplicates = int(summary.get("duplicate_confirmations", 0) or 0)
    passed = (
        payload_returncode == 0
        and not missing_counts
        and bool(summary)
        and recall >= min_recall
        and duplicates == 0
    )
    result.update({
        "collector_returncode": payload_returncode,
        "passed": passed,
        "missing_counts": missing_counts,
        "source_recall": summary.get("source_recall"),
        "source_precision": summary.get("source_precision"),
        "truth_count": summary.get("truth_count"),
        "matched_count": summary.get("matched_count"),
        "confirmed_count": summary.get("confirmed_count"),
        "duplicate_confirmations": summary.get("duplicate_confirmations"),
        "time_to_first_source": summary.get("time_to_first_source"),
        "path_length_m": summary.get("path_length_m"),
        "nav2_available": metadata.get("nav2_available"),
        "nav2_plan_count": metadata.get("nav2_plan_count"),
        "failure_counts": attribution.get("failure_counts"),
        "counts": counts,
    })
    return result


def select_cases(args: argparse.Namespace) -> List[MatrixCase]:
    if args.case:
        return args.case
    if args.preset == "phase0":
        return filter_cases(phase0_cases(), args.worlds, args.cases)
    if args.preset == "full":
        return _full_cases()
    if args.preset == "variable":
        return VARIABLE_SOURCE_CASES
    if args.preset == "extended":
        return EXTENDED_CASES
    return REPRESENTATIVE_CASES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["phase0", "representative", "extended", "variable", "full"], default="phase0")
    parser.add_argument("--case", type=_parse_case, action="append", help="name:world_file:scenario_file")
    parser.add_argument("--worlds", default="", help="phase0 世界类筛选, 例: open,boxes")
    parser.add_argument("--cases", default="", help="phase0 用例名筛选, 例: open__static2")
    parser.add_argument("--seeds", default="101,102,103,104,105", help="逗号分隔的运行 seed 列表")
    parser.add_argument("--strategy", choices=["full", "frontier", "levy"], default="full")
    parser.add_argument("--jitter", type=float, default=0.0, help="scenario_jitter_std_m")
    parser.add_argument("--out-root", default="", help="default: /tmp/thermal_matrix_<timestamp>")
    parser.add_argument("--duration", type=float, default=120.0, help="collector duration per run")
    parser.add_argument("--warmup", type=float, default=36.0, help="launch warmup before payload starts")
    parser.add_argument("--domain-start", type=int, default=71)
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--health-only", action="store_true", help="只验证 Nav2 健康, 不采集")
    args = parser.parse_args()

    matrix_stats = _load_matrix_stats()
    cases = select_cases(args)
    if not cases:
        raise ValueError("case selection is empty; check --worlds/--cases filters")
    _ensure_files(cases)
    seeds = parse_seeds(args.seeds) if not args.health_only else parse_seeds(args.seeds)[:1]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root).expanduser() if args.out_root else Path(f"/tmp/thermal_matrix_{ts}")
    out_root.mkdir(parents=True, exist_ok=True)
    n_runs = len(cases) * len(seeds)
    if args.domain_start + n_runs - 1 > 232:
        # 域 id 循环复用(串行执行, 仅需相邻 run 不同即可)
        pass

    runs: List[Dict] = []
    run_idx = 0
    for case in cases:
        for seed in seeds:
            run_idx += 1
            domain_id = args.domain_start + ((run_idx - 1) % 150)
            print(f"[matrix] run {run_idx}/{n_runs}: {case.name} seed={seed} strategy={args.strategy}")
            result = run_case(
                case=case,
                run_dir=out_root / case.name / f"seed{seed}",
                duration_s=args.duration,
                warmup_s=args.warmup,
                domain_id=domain_id,
                min_recall=args.min_recall,
                seed=seed,
                strategy=args.strategy,
                jitter_std_m=args.jitter,
                health_only=args.health_only,
            )
            runs.append(result)
            status = "PASS" if result.get("passed") else "FAIL"
            if args.health_only:
                print(f"[matrix] {status} {case.name} seed={seed} (nav2 health)")
            else:
                print(
                    f"[matrix] {status} {case.name} seed={seed}: "
                    f"recall={result.get('source_recall')} precision={result.get('source_precision')} "
                    f"dup={result.get('duplicate_confirmations')}"
                )

    cases_agg: Dict[str, Dict] = {}
    for case in cases:
        case_runs = [r for r in runs if r["name"] == case.name]
        agg = matrix_stats.aggregate_case_runs(case_runs)
        fc_total = {k: 0 for k in matrix_stats.FAILURE_CLASSES}
        for r in case_runs:
            for k, v in (r.get("failure_counts") or {}).items():
                fc_total[k] = fc_total.get(k, 0) + int(v)
        agg["failure_counts"] = fc_total
        cases_agg[case.name] = agg

    summary = {
        "config": {
            "preset": args.preset,
            "strategy": args.strategy,
            "seeds": seeds,
            "duration_s": args.duration,
            "warmup_s": args.warmup,
            "jitter_std_m": args.jitter,
            "health_only": args.health_only,
        },
        "n_runs": len(runs),
        "n_passed": sum(1 for item in runs if item.get("passed")),
        "all_passed": all(item.get("passed") for item in runs),
        "cases": cases_agg,
        "runs": runs,
    }
    out_path = out_root / "matrix_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    report_path = out_root / "matrix_report.md"
    report_path.write_text(matrix_stats.render_markdown_report(summary))
    print(f"[matrix] wrote {out_path}")
    print(f"[matrix] wrote {report_path}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 8.4: 运行全部测试**

Run: `python3 -m pytest src/thermal_robot/tests/ -q`
Expected: 全部通过（34 + 30 个）。特别注意既有 `test_T_PY23_matrix_runner_covers_world_scenario_combinations` 必须仍通过（依赖保留的旧符号）。

- [ ] **Step 8.5: 提交**

```bash
git add src/thermal_robot/scripts/run_multiscenario_matrix.py src/thermal_robot/tests/test_phase0_evaluation.py
git commit -m "feat: multi-seed matrix runner with baselines, attribution and reports"
```

---

### 任务 9：v31 锚点、全量验证、冒烟运行、devlog

**Files:**
- Create: `docs/devlog/<日期时间>-phase0-evaluation-foundation.md`

- [ ] **Step 9.1: 打 v31 锚点 tag**

基线对比需要一个冻结的参照实现。`strategy=full` 即当前 v31 行为，tag 标记其代码状态：

```bash
git tag -f v31-anchor
git tag -l v31-anchor
```

Expected: 输出 `v31-anchor`

- [ ] **Step 9.2: 全量静态验证**

```bash
python3 -m py_compile \
  src/thermal_robot/scripts/matrix_stats.py \
  src/thermal_robot/scripts/world_occupancy.py \
  src/thermal_robot/scripts/attribution.py \
  src/thermal_robot/scripts/run_multiscenario_matrix.py
python3 -m pytest src/thermal_robot/tests/ -q
source /opt/ros/humble/setup.bash && source install/setup.bash
colcon build --packages-select thermal_sensor_sim thermal_motion_controller thermal_bringup
```

Expected: 全部通过、构建成功。

- [ ] **Step 9.3: 单 run 冒烟（开阔世界，验证 seed/采集/归因/报告全链路）**

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
python3 src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset phase0 --cases open__static2 --seeds 101 \
  --duration 45 --warmup 36 --domain-start 201 \
  --out-root /tmp/phase0_smoke
```

Expected:
- 退出码 0，打印 `PASS open__static2 seed=101`；
- `/tmp/phase0_smoke/open__static2/seed101/` 内存在 `attribution.json`、`source_summary.json`、`metadata.json`；
- `/tmp/phase0_smoke/matrix_summary.json` 和 `matrix_report.md` 存在且报告含 `open__static2` 行。

检查 seed 确实生效（launch.log 中 sensor_node 启动行应含 `run_seed=101`）：

```bash
grep "run_seed=101" /tmp/phase0_smoke/open__static2/seed101/launch.log
```

Expected: 命中一行。若未命中，排查任务 6 的参数接线。

- [ ] **Step 9.4: 基线策略冒烟（确认 strategy 参数生效）**

```bash
python3 src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset phase0 --cases open__static2 --seeds 101 \
  --strategy levy --duration 45 --warmup 36 --domain-start 203 \
  --out-root /tmp/phase0_smoke_levy
grep "strategy=levy" /tmp/phase0_smoke_levy/open__static2/seed101/launch.log
```

Expected: 命中（controller 启动日志行 `strategy=levy random_seed=101`）。

- [ ] **Step 9.5: 障碍世界 Nav2 健康检查**

```bash
python3 src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset phase0 --worlds boxes,walls,mixed --cases '' --seeds 101 \
  --health-only --duration 40 --warmup 40 --domain-start 205 \
  --out-root /tmp/phase0_nav2_health
cat /tmp/phase0_nav2_health/matrix_summary.json | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print(d['n_passed'], '/', d['n_runs'])"
```

注意 `--worlds boxes,walls,mixed` 不带 `--cases` 时会跑 3 世界 × 6 场景 = 18 个 health run，太多；改用每世界一个代表场景：

```bash
python3 src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset phase0 --cases boxes__static3,walls__static3,mixed__static3 --seeds 101 \
  --health-only --duration 40 --warmup 40 --domain-start 205 \
  --out-root /tmp/phase0_nav2_health
```

Expected: 3/3 passed。**如果有 FAIL**：查看对应 `nav2_health.json` 与 `launch.log`，这就是 spec §6 风险 1（障碍世界 Nav2 不稳）的实测数据——不要静默跳过，把失败现象（哪个世界、lifecycle 还是 plan 失败）记录到 devlog，作为阶段 1 可达性检查降级路径的输入。健康失败**不阻塞**本任务完成（评测仪表本身已交付），但必须如实记录。

- [ ] **Step 9.6: 清理仿真残留**

```bash
bash src/thermal_robot/kill_gz.sh
pgrep -a gzserver || echo "gazebo clean"
```

Expected: `gazebo clean`

- [ ] **Step 9.7: 写 devlog**

创建 `docs/devlog/<YYYY-MM-DD-HHMM>-phase0-evaluation-foundation.md`（时间用实际值），内容结构沿用项目惯例：

```markdown
# <日期时间> 阶段0评测地基: 多seed统计、失败归因、基线挂架

## 背景
依据 docs/superpowers/specs/2026-06-10-thermal-multisource-program-design.md 阶段0。
（简述本轮交付了什么）

## 改动
（按任务1-8列出新模块与接线，一行一条）

## 验证
- python3 -m pytest src/thermal_robot/tests/ -q: <实际数字> passed
- colcon build: 通过
- 冒烟 run: /tmp/phase0_smoke 结果摘要（recall/归因计数）
- 基线冒烟: strategy=levy 生效证据
- 障碍世界 Nav2 健康: <实际 3/3 或失败详情>

## 中间失败与处理
（如实记录执行中遇到的问题）

## 结论与下一步
阶段0门槛核对:
- [ ] 扩展矩阵多seed指标可复现（冒烟验证链路, 全量矩阵留待门槛评审运行）
- [ ] 基线全部跑通（full/levy 冒烟; frontier 同链路）
- [ ] 每局自动生成归因报告
下一步: 运行完整 phase0 矩阵 (--preset phase0, 5 seeds, 约一夜) 提交门槛评审。
```

- [ ] **Step 9.8: 最终提交**

```bash
git add docs/devlog/
git commit -m "docs: log phase0 evaluation foundation delivery"
```

---

## 计划自审记录（已执行）

1. **Spec 覆盖**：阶段 0 四项交付物——seeds/统计（任务 1、8）、归因仪表（任务 2、3、7）、障碍世界+真值栅格+Nav2 健康（任务 3、9.5；世界文件已存在故无需新建）、基线挂架（任务 6、8）。§5.4 的 CPU 耗时指标属于阶段 1 门槛（spec 写明"从阶段 1 起"），本阶段不实现。Mann-Whitney U 在本阶段交付为库函数（任务 1），门槛对比时调用。
2. **占位符**：无 TBD/TODO；所有代码块完整。
3. **类型一致性**：`MatrixCase(name, world, scenario)` 全文一致；`TruthOccupancyGrid` 字段与 `load_grid/save_grid/line_of_sight/is_occupied` 签名一致；`compute_attribution` 参数与测试调用一致；`aggregate_case_runs`/`render_markdown_report` 的 summary 字典结构在任务 1 测试与任务 8 runner 中一致。
4. **已知风险提示**：任务 3 若某障碍世界解析出 0 个 box（用了非 box 几何），按步骤内指引扩展解析器；任务 9.5 的 Nav2 健康失败按指引记录而非跳过。
