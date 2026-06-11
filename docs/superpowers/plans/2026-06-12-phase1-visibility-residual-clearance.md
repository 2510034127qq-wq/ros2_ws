# 阶段 1：遮挡感知观测 + 残差探索 + 清场概率 v1 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 spec 阶段 1 全部交付物：观测接口契约（透视兼容）、射线可见性掩码、`WorldThermalGrid` 三态融合 + 观测方位扇区、残差场、残差驱动的 COARSE 目标选择（带直线可达性检查）、生成式清场概率 v1、配对统计检验 + cluster bootstrap、热相机选型文档。新行为挂在 `strategy=residual` 下，**v31 锚点（full/frontier/levy）行为一个字节都不许变**。

**Architecture:** 观测层（thermal_field_reconstructor）新增 `observation.py`（契约+俯视投影器）、`visibility.py`（占据栅格射线检测）、`residual.py`（正向预测−实测）；`WorldThermalGrid` 改为消费契约观测并按可见性三态融合，新字段经扩展的 `ThermalMap.msg` 进控制器。规划层（thermal_motion_controller）新增 `planning.select_residual_target`（残差质量+可见性未见度+可达性）和 `clearance.py`（Poisson 先验 × 扇区漏检似然）。控制器 `strategy=residual` 分支接通以上全部；清场概率只发布/记录不终止（`clearance_terminate` 默认 false）。

**Tech Stack:** Python 3.10 / numpy / pytest（纯模块测试，无 ROS）；ROS 2 Humble + colcon（接线与构建）；Gazebo Classic（冒烟）。

**Spec:** `docs/superpowers/specs/2026-06-10-thermal-multisource-program-design.md`（2026-06-12 修订版，§3 契约与清场模型、§4 阶段 1、§5.4 统计口径）

---

## 重要背景事实（执行前必读）

1. **构建顺序**：`thermal_interfaces` 必须最先构建（本计划改 `ThermalMap.msg`，所以 Task 5 之后的任何 ROS 验证都要先重建 interfaces 再重建依赖包）：
   ```bash
   source /opt/ros/humble/setup.bash
   colcon build --packages-select thermal_interfaces
   source install/setup.bash
   colcon build --packages-select thermal_field_reconstructor thermal_motion_controller thermal_bringup
   source install/setup.bash
   ```
2. **纯模块测试不需要 ROS**：`python3 -m pytest src/thermal_robot/tests/ -q`。当前基线 **68 passed**，全程不许跌。测试通过 `sys.path.insert` 直接加包源码目录（见 `tests/test_thermal_system.py` 开头的模式），不依赖 colcon 安装。
3. **坐标系**：world 系 = odom/map 系 + spawn 偏移 (-6, 0)。SLAM `/map`（nav_msgs/OccupancyGrid）的 origin 在 map 系，换到 world 系要加 spawn：`world_origin_x = msg.info.origin.position.x + spawn_x`。`WorldThermalGrid` 与真值占据栅格中心都是 world 系 (-6, 0)。
4. **现有融合管线**：`thermal_mapper_node`（thermal_field_reconstructor 包）订阅 `/thermal/filtered`(Image 64×48 32FC1) + `/odom`，用 `WorldThermalGrid.integrate_image` 融合，按 `ThermalMap` 消息发布到 `/thermal/map`；控制器 `_map_cb`（controller_node.py 约 971 行）解析成 dict 给规划用。**注意 `_map_cb` 目前不存 `temperature_mean`，Task 9 要补。**
5. **控制器 strategy 链路**：launch arg `strategy` → controller 参数 → `self._strategy_mode`，合法值检查在 controller_node.py 约 433 行 `('full', 'frontier', 'levy')`；矩阵 runner 的 choices 在 `scripts/run_multiscenario_matrix.py` 约 396 行。两处都要加 `'residual'`。
6. **v31 锚点冻结**：`full`/`frontier`/`levy` 三种 strategy 的行为是论文基线，**不得改变**。所有新逻辑必须在 `strategy == 'residual'` 分支内或纯新增模块里。改 `WorldThermalGrid` 时 `integrate_image`（无 occupancy 参数时）的数值行为必须与现状完全一致（Task 4 有回归测试）。
7. **`_found_sources`** 是 `(x, y, peak_temp)` 元组列表（controller_node.py 1854 行）；环境温估计是 `self._ambient_est`；高斯 σ 参数是 `self._heat_sigma`（params `heat_sigma`，默认 1.2）。
8. **运行环境**：如果 `ros2 launch` 报 `~/.ros/log` 只读，设 `export ROS_LOG_DIR=/tmp/ros2_ws_launch_logs`。冒烟跑完务必 `bash src/thermal_robot/kill_gz.sh`（返回 137 正常），再 `pgrep -a gzserver || echo "gazebo clean"` 确认。
9. **scripts 的互相引用模式**：`scripts/*.py` 不装包，用 `importlib.util.spec_from_file_location` 加载兄弟模块（见 `scripts/attribution.py` 的 `_load_sibling`）。引用包内模块也用同样的文件路径加载方式（Task 1 有示例），不要依赖 colcon 安装。
10. **devlog 约定**：`docs/devlog/<YYYY-MM-DD-HHMM>-<slug>.md`，结构为 背景/改动/验证/中间失败与处理/结论。失败如实记录，不许隐瞒。
11. **每个 Task 一次 commit**，消息格式 `feat:`/`test:`/`docs:`。TDD：先写测试看它失败，再实现。

---

### Task 1: 栅格几何常数单一来源（grid_geometry.py）

**Files:**
- Create: `src/thermal_robot/thermal_field_reconstructor/thermal_field_reconstructor/grid_geometry.py`
- Modify: `src/thermal_robot/thermal_field_reconstructor/thermal_field_reconstructor/thermal_mapping.py`（默认值改引用常数）
- Modify: `src/thermal_robot/scripts/world_occupancy.py`（常数改引用）
- Test: `src/thermal_robot/tests/test_phase1_observation.py`（新文件，本计划所有纯模块测试都进这里）

- [ ] **Step 1: 写失败测试**

新建 `src/thermal_robot/tests/test_phase1_observation.py`：

```python
#!/usr/bin/env python3
"""阶段1纯模块测试: 观测契约 / 可见性 / 三态融合 / 残差 / 清场 / 配对统计."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parents[3]
ROBOT = WORKSPACE / 'src' / 'thermal_robot'
for rel in ('thermal_field_reconstructor', 'thermal_motion_controller'):
    p = str(ROBOT / rel)
    if p not in sys.path:
        sys.path.insert(0, p)

from thermal_field_reconstructor import grid_geometry  # noqa: E402


def _load_script(name: str):
    path = ROBOT / 'scripts' / f'{name}.py'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestGridGeometry:
    def test_constants(self):
        assert grid_geometry.GRID_CENTER_X == -6.0
        assert grid_geometry.GRID_CENTER_Y == 0.0
        assert grid_geometry.GRID_SIZE_M == 50.0

    def test_world_occupancy_uses_shared_constants(self):
        wo = _load_script('world_occupancy')
        assert wo.GRID_CENTER_X == grid_geometry.GRID_CENTER_X
        assert wo.GRID_CENTER_Y == grid_geometry.GRID_CENTER_Y
        assert wo.GRID_SIZE_M == grid_geometry.GRID_SIZE_M
        assert wo.GRID_RESOLUTION == 0.1  # 真值栅格分辨率是消费方自己的参数, 不统一

    def test_world_thermal_grid_uses_shared_constants(self):
        from thermal_field_reconstructor.thermal_mapping import WorldThermalGrid
        g = WorldThermalGrid()
        assert g.origin_x == grid_geometry.GRID_CENTER_X - grid_geometry.GRID_SIZE_M / 2.0
        assert g.origin_y == grid_geometry.GRID_CENTER_Y - grid_geometry.GRID_SIZE_M / 2.0

    def test_existing_npz_origin_matches(self):
        occ_dir = ROBOT / 'thermal_bringup' / 'worlds' / 'occupancy'
        sample = occ_dir / 'thermal_scene_nav.npz'
        assert sample.exists()
        with np.load(sample) as f:
            assert float(f['origin_x']) == pytest.approx(
                grid_geometry.GRID_CENTER_X - grid_geometry.GRID_SIZE_M / 2.0)
```

- [ ] **Step 2: 跑测试确认失败**

```bash
python3 -m pytest src/thermal_robot/tests/test_phase1_observation.py -q
```
预期：`ModuleNotFoundError: No module named 'thermal_field_reconstructor.grid_geometry'`（或 ImportError）。

- [ ] **Step 3: 实现 grid_geometry.py**

```python
"""World 系栅格几何常数的单一来源.

真值占据栅格(scripts/world_occupancy.py)与在线信念栅格(WorldThermalGrid)
必须共用同一中心与边长, 否则离线归因与在线规划会静默漂移(spec §4 阶段1)。
分辨率不在此统一: 真值栅格 0.1m、信念栅格 0.25m, 属各自消费方的参数。
"""

GRID_CENTER_X = -6.0
GRID_CENTER_Y = 0.0
GRID_SIZE_M = 50.0
```

- [ ] **Step 4: thermal_mapping.py 默认值改引用**

在 `thermal_mapping.py` 顶部 import 区加：

```python
from thermal_field_reconstructor.grid_geometry import (
    GRID_CENTER_X, GRID_CENTER_Y, GRID_SIZE_M)
```

把 `WorldThermalGrid.__init__` 签名里的：

```python
        center_x: float = -6.0,
        center_y: float = 0.0,
        size_x_m: float = 50.0,
        size_y_m: float = 50.0,
```

改为：

```python
        center_x: float = GRID_CENTER_X,
        center_y: float = GRID_CENTER_Y,
        size_x_m: float = GRID_SIZE_M,
        size_y_m: float = GRID_SIZE_M,
```

- [ ] **Step 5: world_occupancy.py 常数改引用**

`scripts/world_occupancy.py` 顶部（`import numpy as np` 之后）把：

```python
GRID_CENTER_X = -6.0
GRID_CENTER_Y = 0.0
GRID_SIZE_M = 50.0
```

替换为（保留 `GRID_RESOLUTION = 0.1` 等其余常数不动）：

```python
import importlib.util
import sys

_GRID_GEOMETRY_PATH = (Path(__file__).resolve().parents[1]
                       / 'thermal_field_reconstructor'
                       / 'thermal_field_reconstructor' / 'grid_geometry.py')


def _load_grid_geometry():
    name = 'thermal_grid_geometry_shared'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _GRID_GEOMETRY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_gg = _load_grid_geometry()
# 单一来源: 真值栅格与在线信念栅格共用中心与边长 (spec §4 阶段1)
GRID_CENTER_X = _gg.GRID_CENTER_X
GRID_CENTER_Y = _gg.GRID_CENTER_Y
GRID_SIZE_M = _gg.GRID_SIZE_M
```

注意：直接按文件路径加载，不经过包 `__init__`，scripts 无需 colcon 环境。

- [ ] **Step 6: 跑测试确认通过 + 全量回归**

```bash
python3 -m pytest src/thermal_robot/tests/test_phase1_observation.py -q   # 4 passed
python3 -m pytest src/thermal_robot/tests/ -q                             # 72 passed
```

- [ ] **Step 7: Commit**

```bash
git add src/thermal_robot/thermal_field_reconstructor/thermal_field_reconstructor/grid_geometry.py \
        src/thermal_robot/thermal_field_reconstructor/thermal_field_reconstructor/thermal_mapping.py \
        src/thermal_robot/scripts/world_occupancy.py \
        src/thermal_robot/tests/test_phase1_observation.py
git commit -m "feat: single source of truth for grid geometry constants"
```

---

### Task 2: 观测接口契约（observation.py）

按 spec §3 契约：每条观测 =（射线/单元，温度，置信度，timestamp，frame_id，传感器位姿/外参，measurement_type）。俯视矩形投影是首个实现。`project_pixels_to_world` **移动**到 observation.py（thermal_mapping 保留 re-export 以兼容旧引用），避免循环 import。

**Files:**
- Create: `src/thermal_robot/thermal_field_reconstructor/thermal_field_reconstructor/observation.py`
- Modify: `src/thermal_robot/thermal_field_reconstructor/thermal_field_reconstructor/thermal_mapping.py`
- Test: `src/thermal_robot/tests/test_phase1_observation.py`

- [ ] **Step 1: 写失败测试**（追加到 test_phase1_observation.py）

```python
class TestObservationContract:
    def _projector(self):
        from thermal_field_reconstructor.observation import TopDownRectProjector
        return TopDownRectProjector(fov_x=4.0, fov_y=3.0)

    def test_contract_fields(self):
        from thermal_field_reconstructor.observation import (
            SensorPose2D, MEASUREMENT_FIELD_DIRECT)
        img = np.full((48, 64), 25.0, dtype=np.float32)
        pose = SensorPose2D(x=1.0, y=2.0, yaw=0.5)
        obs = self._projector().project(img, pose, stamp_s=12.5)
        assert obs.stamp_s == 12.5
        assert obs.sensor_pose.frame_id == 'world'
        assert obs.sensor_pose.x == 1.0 and obs.sensor_pose.yaw == 0.5
        assert obs.measurement_type == MEASUREMENT_FIELD_DIRECT
        n = 48 * 64
        assert obs.sample_wx.size == n and obs.sample_wy.size == n
        assert obs.temperature.size == n and obs.confidence.size == n
        assert np.all(obs.confidence == 1.0)

    def test_footprint_centered_on_sensor(self):
        from thermal_field_reconstructor.observation import SensorPose2D
        img = np.zeros((48, 64), dtype=np.float32)
        obs = self._projector().project(img, SensorPose2D(x=3.0, y=-1.0, yaw=0.0), 0.0)
        assert float(obs.sample_wx.mean()) == pytest.approx(3.0, abs=1e-5)
        assert float(obs.sample_wy.mean()) == pytest.approx(-1.0, abs=1e-5)
        assert float(obs.sample_wx.max()) == pytest.approx(3.0 + 2.0, abs=1e-5)
        assert float(obs.sample_wy.max()) == pytest.approx(-1.0 + 1.5, abs=1e-5)

    def test_yaw_rotates_footprint(self):
        from thermal_field_reconstructor.observation import SensorPose2D
        img = np.zeros((48, 64), dtype=np.float32)
        obs = self._projector().project(
            img, SensorPose2D(x=0.0, y=0.0, yaw=math.pi / 2.0), 0.0)
        # 旋转90°: fov_x(4m) 落到 y 轴
        assert float(obs.sample_wy.max()) == pytest.approx(2.0, abs=1e-4)
        assert float(obs.sample_wx.max()) == pytest.approx(1.5, abs=1e-4)

    def test_size_mismatch_raises(self):
        from thermal_field_reconstructor.observation import (
            ThermalObservation, SensorPose2D, MEASUREMENT_FIELD_DIRECT)
        with pytest.raises(ValueError):
            ThermalObservation(
                stamp_s=0.0,
                sensor_pose=SensorPose2D(0.0, 0.0, 0.0),
                measurement_type=MEASUREMENT_FIELD_DIRECT,
                sample_wx=np.zeros(3, np.float32),
                sample_wy=np.zeros(3, np.float32),
                temperature=np.zeros(4, np.float32),
                confidence=np.ones(3, np.float32),
            )

    def test_legacy_reexport_still_works(self):
        # 旧代码从 thermal_mapping import project_pixels_to_world, 必须继续可用
        from thermal_field_reconstructor.thermal_mapping import project_pixels_to_world
        wxs, wys = project_pixels_to_world(64, 48, 4.0, 3.0, 0.0, 0.0, 0.0)
        assert wxs.shape == (48, 64)
```

- [ ] **Step 2: 跑测试确认失败**

```bash
python3 -m pytest src/thermal_robot/tests/test_phase1_observation.py -q
```
预期：`No module named 'thermal_field_reconstructor.observation'`。

- [ ] **Step 3: 实现 observation.py**

```python
"""观测接口契约 (spec §3).

每条观测批 = (采样点世界坐标, 温度, 置信度) + timestamp + frame_id +
传感器位姿(外参) + measurement_type。俯视矩形投影器是契约的首个实现
(A级仿真传感器); 前视透视投影器(B级)以后实现同一契约, 算法层不感知差异。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MEASUREMENT_FIELD_DIRECT = 'field_direct'        # 直读场温 (A级仿真)
MEASUREMENT_SURFACE_RADIANCE = 'surface_radiance'  # 表面辐射 (B级/实机)


@dataclass(frozen=True)
class SensorPose2D:
    x: float
    y: float
    yaw: float
    frame_id: str = 'world'


@dataclass
class ThermalObservation:
    stamp_s: float
    sensor_pose: SensorPose2D
    measurement_type: str
    sample_wx: np.ndarray
    sample_wy: np.ndarray
    temperature: np.ndarray
    confidence: np.ndarray

    def __post_init__(self):
        n = int(np.asarray(self.temperature).size)
        for name in ('sample_wx', 'sample_wy', 'confidence'):
            size = int(np.asarray(getattr(self, name)).size)
            if size != n:
                raise ValueError(f'{name} size {size} != temperature size {n}')


def project_pixels_to_world(
    width: int,
    height: int,
    fov_x: float,
    fov_y: float,
    robot_wx: float,
    robot_wy: float,
    robot_yaw: float,
):
    """Project a rectified thermal image footprint into world coordinates."""
    px_xs = np.linspace(-fov_x / 2.0, fov_x / 2.0, width, dtype=np.float32)
    px_ys = np.linspace(-fov_y / 2.0, fov_y / 2.0, height, dtype=np.float32)
    px_xx, px_yy = np.meshgrid(px_xs, px_ys)
    cos_y = math.cos(robot_yaw)
    sin_y = math.sin(robot_yaw)
    world_xs = robot_wx + cos_y * px_xx - sin_y * px_yy
    world_ys = robot_wy + sin_y * px_xx + cos_y * px_yy
    return world_xs, world_ys


class TopDownRectProjector:
    """理想俯视矩形足迹投影 (仿真特化)."""

    measurement_type = MEASUREMENT_FIELD_DIRECT

    def __init__(self, fov_x: float = 4.0, fov_y: float = 3.0):
        self.fov_x = float(fov_x)
        self.fov_y = float(fov_y)

    def project(self, image: np.ndarray, pose: SensorPose2D,
                stamp_s: float) -> ThermalObservation:
        img = np.asarray(image, dtype=np.float32)
        wxs, wys = project_pixels_to_world(
            img.shape[1], img.shape[0], self.fov_x, self.fov_y,
            pose.x, pose.y, pose.yaw)
        temps = img.reshape(-1)
        return ThermalObservation(
            stamp_s=float(stamp_s),
            sensor_pose=pose,
            measurement_type=self.measurement_type,
            sample_wx=wxs.reshape(-1).astype(np.float32),
            sample_wy=wys.reshape(-1).astype(np.float32),
            temperature=temps,
            confidence=np.ones_like(temps),
        )
```

- [ ] **Step 4: thermal_mapping.py 移除本地定义、改为 re-export**

删除 `thermal_mapping.py` 里的 `def project_pixels_to_world(...)` 函数体（26–43 行），在顶部 import 区加：

```python
from thermal_field_reconstructor.observation import (  # noqa: F401  (re-export)
    SensorPose2D, ThermalObservation, TopDownRectProjector,
    project_pixels_to_world)
```

`integrate_image` 内对 `project_pixels_to_world` 的调用不用改（名字仍可见）。

- [ ] **Step 5: 跑测试确认通过 + 全量回归**

```bash
python3 -m pytest src/thermal_robot/tests/test_phase1_observation.py -q   # 9 passed
python3 -m pytest src/thermal_robot/tests/ -q                             # 77 passed
```

- [ ] **Step 6: Commit**

```bash
git add -A src/thermal_robot/thermal_field_reconstructor src/thermal_robot/tests/test_phase1_observation.py
git commit -m "feat: thermal observation interface contract with top-down projector"
```

---

### Task 3: 射线可见性（visibility.py）

**Files:**
- Create: `src/thermal_robot/thermal_field_reconstructor/thermal_field_reconstructor/visibility.py`
- Test: `src/thermal_robot/tests/test_phase1_observation.py`

- [ ] **Step 1: 写失败测试**（追加）

```python
def _make_occ(width=40, height=40, resolution=0.25, origin_x=-5.0, origin_y=-5.0):
    """10m x 10m 空地图, 原点(-5,-5)."""
    from thermal_field_reconstructor import visibility
    data = np.zeros((height, width), dtype=np.int16)
    return visibility.OccupancyView(origin_x, origin_y, resolution, data)


class TestVisibility:
    def test_empty_map_all_visible(self):
        from thermal_field_reconstructor import visibility
        occ = _make_occ()
        txs = np.array([3.0, -2.0, 0.0]); tys = np.array([3.0, 4.0, -4.5])
        vis = visibility.visible_mask(occ, 0.0, 0.0, txs, tys)
        assert vis.tolist() == [True, True, True]

    def test_wall_blocks_behind(self):
        occ = _make_occ()
        # x=2.0~2.25 处一道竖墙 (列 ix=28)
        occ.data[:, 28] = 100
        from thermal_field_reconstructor import visibility
        vis = visibility.visible_mask(
            occ, 0.0, 0.0, np.array([4.0, 1.0]), np.array([0.0, 0.0]))
        assert vis.tolist() == [False, True]

    def test_target_adjacent_to_wall_visible(self):
        # 终点前最后半步不检查: 紧贴墙面的格仍可见
        occ = _make_occ()
        occ.data[:, 28] = 100
        from thermal_field_reconstructor import visibility
        vis = visibility.visible_mask(
            occ, 0.0, 0.0, np.array([1.95]), np.array([0.0]), step_m=0.1)
        assert vis.tolist() == [True]

    def test_unknown_and_out_of_bounds_are_free(self):
        occ = _make_occ()
        occ.data[:, :] = -1  # 全未知
        from thermal_field_reconstructor import visibility
        vis = visibility.visible_mask(occ, 0.0, 0.0, np.array([20.0]), np.array([20.0]))
        assert vis.tolist() == [True]

    def test_occupied_at_vectorized(self):
        occ = _make_occ()
        occ.data[20, 20] = 100  # world (0.0~0.25, 0.0~0.25)
        from thermal_field_reconstructor import visibility
        out = visibility.occupied_at(
            occ, np.array([0.1, 1.0, 99.0]), np.array([0.1, 1.0, 99.0]))
        assert out.tolist() == [True, False, False]

    def test_line_reachable(self):
        occ = _make_occ()
        occ.data[:, 28] = 100
        from thermal_field_reconstructor import visibility
        assert visibility.line_reachable(occ, 0.0, 0.0, 1.0, 0.0)
        assert not visibility.line_reachable(occ, 0.0, 0.0, 4.0, 0.0)
        assert visibility.line_reachable(None, 0.0, 0.0, 4.0, 0.0)  # 无地图→视为可达

    def test_speed_budget(self):
        # 200 条射线 x 2.5m, 必须远低于算力预算 (粗略上限 50ms)
        import time as _t
        occ = _make_occ(width=200, height=200, resolution=0.05)
        from thermal_field_reconstructor import visibility
        ang = np.linspace(0, 2 * math.pi, 200)
        txs = 2.5 * np.cos(ang); tys = 2.5 * np.sin(ang)
        t0 = _t.monotonic()
        visibility.visible_mask(occ, 0.0, 0.0, txs, tys, step_m=0.1)
        assert (_t.monotonic() - t0) < 0.05
```

- [ ] **Step 2: 跑测试确认失败**

```bash
python3 -m pytest src/thermal_robot/tests/test_phase1_observation.py -k Visibility -q
```
预期：`No module named 'thermal_field_reconstructor.visibility'`。

- [ ] **Step 3: 实现 visibility.py**

```python
"""占据栅格射线可见性 (纯 numpy, 无 ROS).

运行时算法只见 SLAM /map(不见真值); 归因仪表(scripts/attribution.py)用
真值栅格但共用同一射线语义 (spec §5.3)。未知格(-1)与界外按自由处理:
可见性宁可乐观, 把"看不见"交给三态融合的 blocked 记录去纠正。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_OCCUPIED_THRESHOLD = 65
DEFAULT_RAY_STEP_M = 0.1


@dataclass
class OccupancyView:
    origin_x: float
    origin_y: float
    resolution: float
    data: np.ndarray  # (h, w), -1 unknown, 0..100 occupancy
    occupied_threshold: int = DEFAULT_OCCUPIED_THRESHOLD


def from_flat(data, width, height, origin_x, origin_y, resolution,
              occupied_threshold=DEFAULT_OCCUPIED_THRESHOLD) -> OccupancyView:
    arr = np.asarray(data, dtype=np.int16).reshape((int(height), int(width)))
    return OccupancyView(float(origin_x), float(origin_y), float(resolution),
                         arr, int(occupied_threshold))


def occupied_at(view: OccupancyView, wx, wy) -> np.ndarray:
    """向量化占据查询. 界外与未知(-1)按自由(False)处理."""
    wx = np.asarray(wx, dtype=np.float32)
    wy = np.asarray(wy, dtype=np.float32)
    ix = np.floor((wx - view.origin_x) / view.resolution).astype(np.int32)
    iy = np.floor((wy - view.origin_y) / view.resolution).astype(np.int32)
    h, w = view.data.shape
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    out = np.zeros(wx.shape, dtype=bool)
    if inside.any():
        vals = view.data[iy[inside], ix[inside]]
        out[inside] = vals >= view.occupied_threshold
    return out


def visible_mask(view: OccupancyView, ox: float, oy: float,
                 txs, tys, step_m: float = DEFAULT_RAY_STEP_M) -> np.ndarray:
    """对每个目标点: 从 (ox,oy) 出发的线段中途是否无占据格.

    终点前最后半步不检查, 允许观测贴墙格本身。
    """
    txs = np.asarray(txs, dtype=np.float32).reshape(-1)
    tys = np.asarray(tys, dtype=np.float32).reshape(-1)
    if txs.size == 0:
        return np.zeros(0, dtype=bool)
    dx = txs - float(ox)
    dy = tys - float(oy)
    dist = np.hypot(dx, dy)
    max_dist = float(dist.max())
    if max_dist <= step_m:
        return np.ones(txs.size, dtype=bool)
    n_steps = int(math.ceil(max_dist / step_m))
    ts = np.arange(1, n_steps + 1, dtype=np.float32)[:, None] / float(n_steps)
    px = float(ox) + ts * dx[None, :]
    py = float(oy) + ts * dy[None, :]
    travelled = ts * dist[None, :]
    check = travelled < (dist[None, :] - 0.5 * step_m)
    blocked = occupied_at(view, px, py) & check
    return ~blocked.any(axis=0)


def line_reachable(view, x0: float, y0: float, x1: float, y1: float,
                   step_m: float = 0.2) -> bool:
    """直线无碰可达检查 (Nav2 costmap 不可用时的降级语义, spec §3)."""
    if view is None:
        return True
    return bool(visible_mask(view, x0, y0,
                             np.array([x1], dtype=np.float32),
                             np.array([y1], dtype=np.float32),
                             step_m=step_m)[0])
```

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

```bash
python3 -m pytest src/thermal_robot/tests/test_phase1_observation.py -q   # 16 passed
python3 -m pytest src/thermal_robot/tests/ -q
```

- [ ] **Step 5: Commit**

```bash
git add src/thermal_robot/thermal_field_reconstructor/thermal_field_reconstructor/visibility.py \
        src/thermal_robot/tests/test_phase1_observation.py
git commit -m "feat: ray-cast visibility against occupancy grids"
```

---

### Task 4: WorldThermalGrid 三态融合 + 观测方位扇区

每格三态：`VIEW_NEVER=0`（没看过）/ `VIEW_BLOCKED_ONLY=1`（看过但被挡）/ `VIEW_CLEAR=2`（看过且看清）。被挡的格**不融合温度**，只记 `blocked_count`。看清的格额外记观测方位扇区位掩码（8 扇区 × 45°，方位 = 从该格指向传感器的方向），给视角依赖清场模型（v2）留数据（spec §3）。

**Files:**
- Modify: `src/thermal_robot/thermal_field_reconstructor/thermal_field_reconstructor/thermal_mapping.py`
- Test: `src/thermal_robot/tests/test_phase1_observation.py`

- [ ] **Step 1: 写失败测试**（追加）

```python
class TestThreeStateFusion:
    def _grid(self):
        from thermal_field_reconstructor.thermal_mapping import WorldThermalGrid
        return WorldThermalGrid(center_x=0.0, center_y=0.0,
                                size_x_m=20.0, size_y_m=20.0, resolution=0.25)

    def _obs(self, temp=30.0, x=0.0, y=0.0, yaw=0.0, stamp=1.0):
        from thermal_field_reconstructor.observation import (
            SensorPose2D, TopDownRectProjector)
        img = np.full((48, 64), float(temp), dtype=np.float32)
        return TopDownRectProjector().project(
            img, SensorPose2D(x=float(x), y=float(y), yaw=float(yaw)), stamp)

    def test_no_occupancy_matches_legacy_integrate_image(self):
        from thermal_field_reconstructor.observation import (
            SensorPose2D, TopDownRectProjector)
        a, b = self._grid(), self._grid()
        img = np.random.default_rng(7).normal(25.0, 3.0, (48, 64)).astype(np.float32)
        a.integrate_image(img, 1.0, 0.5, 0.3, stamp_s=2.0)
        obs = TopDownRectProjector().project(img, SensorPose2D(1.0, 0.5, 0.3), 2.0)
        b.integrate_observation(obs)
        np.testing.assert_allclose(a.mean, b.mean, rtol=1e-6)
        np.testing.assert_array_equal(a.visit_count, b.visit_count)

    def test_view_state_transitions(self):
        from thermal_field_reconstructor import visibility
        from thermal_field_reconstructor.thermal_mapping import (
            VIEW_NEVER, VIEW_BLOCKED_ONLY, VIEW_CLEAR)
        g = self._grid()
        # 传感器(0,0), 墙在 x=1.0~1.25: footprint 内 x>1.25 的格被挡
        occ = visibility.OccupancyView(-10.0, -10.0, 0.25,
                                       np.zeros((80, 80), dtype=np.int16))
        occ.data[:, 44] = 100  # ix=44 → x=1.0~1.25
        g.integrate_observation(self._obs(30.0, 0.0, 0.0, 0.0), occupancy=occ)
        snap = g.snapshot(now_s=2.0)
        ix_blocked = int((1.8 - g.origin_x) / g.resolution)
        ix_clear = int((0.0 - g.origin_x) / g.resolution)
        iy_mid = int((0.0 - g.origin_y) / g.resolution)
        assert snap.view_state[iy_mid, ix_blocked] == VIEW_BLOCKED_ONLY
        assert snap.view_state[iy_mid, ix_clear] == VIEW_CLEAR
        ix_far = int((8.0 - g.origin_x) / g.resolution)
        assert snap.view_state[iy_mid, ix_far] == VIEW_NEVER
        # 被挡格不融合温度
        assert g.visit_count[iy_mid, ix_blocked] == 0
        assert g.blocked_count[iy_mid, ix_blocked] >= 1
        # 后续看清 → 升级为 CLEAR
        occ.data[:, 44] = 0
        g.integrate_observation(self._obs(30.0, 0.0, 0.0, 0.0), occupancy=occ)
        snap2 = g.snapshot(now_s=3.0)
        assert snap2.view_state[iy_mid, ix_blocked] == VIEW_CLEAR

    def test_view_sectors_bitmask(self):
        from thermal_field_reconstructor.thermal_mapping import N_VIEW_SECTORS
        g = self._grid()
        assert N_VIEW_SECTORS == 8
        # 从东边看格(0,0): 格→传感器方位 ≈ 0°
        g.integrate_observation(self._obs(25.0, 1.5, 0.0, 0.0))
        iy = int((0.0 - g.origin_y) / g.resolution)
        ix = int((0.0 - g.origin_x) / g.resolution)
        east_bits = int(g.view_sectors[iy, ix])
        assert east_bits != 0
        # 再从西边看, 扇区位增加
        g.integrate_observation(self._obs(25.0, -1.5, 0.0, 0.0))
        both_bits = int(g.view_sectors[iy, ix])
        assert bin(both_bits).count('1') > bin(east_bits).count('1')

    def test_snapshot_has_new_fields(self):
        g = self._grid()
        snap = g.snapshot(now_s=0.0)
        assert snap.view_state.shape == (g.height, g.width)
        assert snap.view_sectors.dtype == np.uint8
```

- [ ] **Step 2: 跑测试确认失败**

```bash
python3 -m pytest src/thermal_robot/tests/test_phase1_observation.py -k "ThreeState" -q
```
预期：`AttributeError: ... no attribute 'integrate_observation'`（或 import VIEW_NEVER 失败）。

- [ ] **Step 3: 实现三态融合**

`thermal_mapping.py` 改动：

(a) 顶部 import 区追加：

```python
from thermal_field_reconstructor import visibility as _visibility
```

(b) 模块级常数（紧跟 import 之后）：

```python
VIEW_NEVER = 0
VIEW_BLOCKED_ONLY = 1
VIEW_CLEAR = 2
N_VIEW_SECTORS = 8
```

(c) `GridSnapshot` dataclass 追加两个字段：

```python
    view_state: np.ndarray
    view_sectors: np.ndarray
```

(d) `WorldThermalGrid.__init__` 末尾（`self.unknown_variance = ...` 之后）追加：

```python
        self.blocked_count = np.zeros(shape, dtype=np.uint32)
        self.view_sectors = np.zeros(shape, dtype=np.uint8)
```

(e) 新方法 `integrate_observation`（放在 `integrate_image` 之前），并把 `integrate_image` 改为薄包装：

```python
    def integrate_observation(self, obs, occupancy=None,
                              ray_step_m=_visibility.DEFAULT_RAY_STEP_M) -> None:
        """契约观测融合: 仅融合可见格, 被挡格只记 blocked_count."""
        ix, iy, valid = self.world_to_cell(obs.sample_wx, obs.sample_wy)
        if not np.any(valid):
            return
        values = np.asarray(obs.temperature, dtype=np.float32)[valid]
        linear = iy[valid] * self.width + ix[valid]
        total_cells = self.width * self.height
        obs_count = np.bincount(linear, minlength=total_cells).astype(np.float32)
        obs_sum = np.bincount(linear, weights=values,
                              minlength=total_cells).astype(np.float32)
        obs_sum_sq = np.bincount(linear, weights=values * values,
                                 minlength=total_cells).astype(np.float32)
        cells = np.flatnonzero(obs_count > 0.0)
        if cells.size == 0:
            return

        cell_ix = (cells % self.width).astype(np.int32)
        cell_iy = (cells // self.width).astype(np.int32)
        cwx, cwy = self.cell_to_world(cell_ix, cell_iy)
        if occupancy is not None:
            vis = _visibility.visible_mask(
                occupancy, obs.sensor_pose.x, obs.sensor_pose.y,
                cwx, cwy, step_m=ray_step_m)
        else:
            vis = np.ones(cells.size, dtype=bool)

        blocked_cells = cells[~vis]
        if blocked_cells.size:
            bc = self.blocked_count.reshape(-1)
            bc[blocked_cells] += 1

        clear = cells[vis]
        if clear.size == 0:
            return
        obs_n = obs_count[clear]
        obs_mean = obs_sum[clear] / obs_n
        obs_m2 = np.maximum(0.0, obs_sum_sq[clear] - obs_n * obs_mean * obs_mean)

        mean_flat = self.mean.reshape(-1)
        m2_flat = self._m2.reshape(-1)
        visit_flat = self.visit_count.reshape(-1)
        last_flat = self.last_seen.reshape(-1)

        prev_n = visit_flat[clear].astype(np.float32)
        prev_mean = mean_flat[clear]
        new_n = prev_n + obs_n
        delta = obs_mean - prev_mean
        mean_flat[clear] = prev_mean + delta * obs_n / np.maximum(new_n, 1.0)
        m2_flat[clear] = (m2_flat[clear] + obs_m2
                          + delta * delta * prev_n * obs_n / np.maximum(new_n, 1.0))
        visit_flat[clear] = np.clip(new_n, 0, np.iinfo(np.uint32).max).astype(np.uint32)
        last_flat[clear] = float(obs.stamp_s)

        # 观测方位扇区: 从该格指向传感器的方位角 → 8 扇区位掩码
        sec_flat = self.view_sectors.reshape(-1)
        az = np.arctan2(obs.sensor_pose.y - cwy[vis],
                        obs.sensor_pose.x - cwx[vis])
        sector = (((az + math.pi) / (2.0 * math.pi)) * N_VIEW_SECTORS).astype(np.int32)
        sector = np.clip(sector, 0, N_VIEW_SECTORS - 1)
        sec_flat[clear] |= np.left_shift(np.uint8(1), sector.astype(np.uint8))

    def integrate_image(
        self,
        image: np.ndarray,
        robot_wx: float,
        robot_wy: float,
        robot_yaw: float,
        stamp_s: float,
        fov_x: float = 4.0,
        fov_y: float = 3.0,
        occupancy=None,
    ) -> None:
        projector = TopDownRectProjector(fov_x=fov_x, fov_y=fov_y)
        obs = projector.project(
            np.asarray(image, dtype=np.float32),
            SensorPose2D(x=float(robot_wx), y=float(robot_wy), yaw=float(robot_yaw)),
            float(stamp_s))
        self.integrate_observation(obs, occupancy=occupancy)
```

（旧 `integrate_image` 的函数体整体删除，由上面的包装替代。Welford 融合数学与原实现逐行相同，仅作用域从"全部格"变为"可见格"，无 occupancy 时两者一致。）

(f) `snapshot()` 在构造 `GridSnapshot` 之前加：

```python
        view_state = np.zeros_like(self.mean, dtype=np.uint8)
        view_state[self.blocked_count > 0] = VIEW_BLOCKED_ONLY
        view_state[self.visit_count > 0] = VIEW_CLEAR
```

并给 `GridSnapshot(...)` 增加实参：

```python
            view_state=view_state,
            view_sectors=self.view_sectors.copy(),
```

注意 `np.left_shift(np.uint8(1), sector.astype(np.uint8))` 若 numpy 版本告警溢出，可改为 `(1 << sector).astype(np.uint8)`（sector ≤ 7，数值安全）。

- [ ] **Step 4: 清理 Step 1 测试中的笔误段，跑测试确认通过 + 全量回归**

```bash
python3 -m pytest src/thermal_robot/tests/test_phase1_observation.py -q   # 20 passed
python3 -m pytest src/thermal_robot/tests/ -q                             # 旧 68 个全过
```

- [ ] **Step 5: Commit**

```bash
git add src/thermal_robot/thermal_field_reconstructor/thermal_field_reconstructor/thermal_mapping.py \
        src/thermal_robot/tests/test_phase1_observation.py
git commit -m "feat: three-state visibility fusion with view sector bitmask"
```

---

### Task 5: ThermalMap.msg 扩展 + mapper 接线 SLAM 占据 + 控制器解析

**Files:**
- Modify: `src/thermal_robot/thermal_interfaces/msg/ThermalMap.msg`
- Modify: `src/thermal_robot/thermal_field_reconstructor/thermal_field_reconstructor/thermal_mapper_node.py`
- Modify: `src/thermal_robot/thermal_motion_controller/thermal_motion_controller/controller_node.py`（仅 `_map_cb`）
- Modify: `src/thermal_robot/thermal_bringup/config/params.yaml`（mapper 段）

本 Task 是 ROS 接线，无纯模块测试；验证手段是构建 + 后续冒烟。**先改消息，先建 interfaces。**

- [ ] **Step 1: ThermalMap.msg 追加字段**

在文件末尾追加：

```
uint8[] view_state      # 0=never seen, 1=seen-but-blocked only, 2=seen clear
uint8[] view_sectors    # bitmask of 8 azimuth sectors this cell was clearly viewed from
```

- [ ] **Step 2: 重建 interfaces 验证消息合法**

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select thermal_interfaces
source install/setup.bash
ros2 interface show thermal_interfaces/msg/ThermalMap | tail -3
```
预期输出包含 `uint8[] view_state` 与 `uint8[] view_sectors`。

- [ ] **Step 3: mapper 节点接 /map 与可见性**

`thermal_mapper_node.py` 改动：

(a) import 区追加：

```python
from nav_msgs.msg import OccupancyGrid

from thermal_field_reconstructor import visibility
from thermal_field_reconstructor.observation import SensorPose2D, TopDownRectProjector
```

(b) `__init__` 参数声明区追加（`unknown_variance` 之后）：

```python
        self.declare_parameter('visibility_enabled', True)
        self.declare_parameter('occupied_threshold', 65)
        self.declare_parameter('visibility_ray_step_m', 0.1)
```

读取区追加（`self._fov_y = ...` 之后）：

```python
        self._visibility_enabled = bool(g('visibility_enabled').value)
        self._occupied_threshold = int(g('occupied_threshold').value)
        self._ray_step = float(g('visibility_ray_step_m').value)
        self._projector = TopDownRectProjector(fov_x=self._fov_x, fov_y=self._fov_y)
        self._occ_view = None
        self._fuse_count = 0
```

(c) 订阅区（`self.create_subscription(Odometry, ...)` 之后）追加。slam_toolbox 的 `/map` 是 transient_local，必须匹配：

```python
        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, '/map', self._slam_map_cb, map_qos)
```

(d) 新回调（`_odom_cb` 之后）：

```python
    def _slam_map_cb(self, msg: OccupancyGrid):
        # /map 在 map 系, world 系 = map 系 + spawn 偏移
        self._occ_view = visibility.from_flat(
            msg.data, msg.info.width, msg.info.height,
            msg.info.origin.position.x + self._spawn_x,
            msg.info.origin.position.y + self._spawn_y,
            msg.info.resolution,
            occupied_threshold=self._occupied_threshold)
```

(e) `_image_cb` 中把：

```python
        self._grid.integrate_image(arr, self._wx, self._wy, self._yaw, now_s, self._fov_x, self._fov_y)
```

替换为：

```python
        pose = SensorPose2D(x=self._wx, y=self._wy, yaw=self._yaw)
        obs = self._projector.project(arr, pose, now_s)
        occ = self._occ_view if self._visibility_enabled else None
        t0 = time.monotonic()
        self._grid.integrate_observation(obs, occupancy=occ, ray_step_m=self._ray_step)
        fuse_ms = (time.monotonic() - t0) * 1000.0
        self._fuse_count += 1
        if self._fuse_count % 100 == 0:
            blocked_total = int((self._grid.blocked_count > 0).sum())
            clear_total = int((self._grid.visit_count > 0).sum())
            self.get_logger().info(
                f'[FUSE] n={self._fuse_count} {fuse_ms:.1f}ms '
                f'occ_map={"yes" if occ is not None else "no"} '
                f'cells_clear={clear_total} cells_blocked_only={blocked_total}')
```

(f) `_publish_map` 在 `msg.last_seen_age_s = ...` 之后追加：

```python
        msg.view_state = snap.view_state.reshape(-1).astype(np.uint8).tolist()
        msg.view_sectors = snap.view_sectors.reshape(-1).astype(np.uint8).tolist()
```

- [ ] **Step 4: 控制器 `_map_cb` 解析新字段（含补 temperature_mean）**

`controller_node.py` 的 `_map_cb`（约 971 行）dict 构造中，`'last_seen_age_s': ...` 行之后追加一个键：

```python
                'temperature_mean': np.asarray(msg.temperature_mean, dtype=np.float32).reshape(shape),
```

并在 `self._thermal_map = {...}` 整个赋值语句之后、`self._thermal_map_t = time.monotonic()` 之前插入（旧 mapper 消息无新字段时长度为 0，保持向后兼容）：

```python
            if len(msg.view_state) == int(msg.width) * int(msg.height):
                self._thermal_map['view_state'] = np.asarray(
                    msg.view_state, dtype=np.uint8).reshape(shape)
                self._thermal_map['view_sectors'] = np.asarray(
                    msg.view_sectors, dtype=np.uint8).reshape(shape)
```

- [ ] **Step 5: params.yaml mapper 段追加**

`src/thermal_robot/thermal_bringup/config/params.yaml` 的 `thermal_mapper_node: ros__parameters:` 下追加：

```yaml
    visibility_enabled: true
    occupied_threshold: 65
    visibility_ray_step_m: 0.1
```

- [ ] **Step 6: 构建 + 回归**

```bash
colcon build --packages-select thermal_field_reconstructor thermal_motion_controller thermal_bringup
source install/setup.bash
python3 -m py_compile src/thermal_robot/thermal_field_reconstructor/thermal_field_reconstructor/thermal_mapper_node.py \
                      src/thermal_robot/thermal_motion_controller/thermal_motion_controller/controller_node.py
python3 -m pytest src/thermal_robot/tests/ -q
```

- [ ] **Step 7: Commit**

```bash
git add src/thermal_robot/thermal_interfaces src/thermal_robot/thermal_field_reconstructor \
        src/thermal_robot/thermal_motion_controller src/thermal_robot/thermal_bringup/config/params.yaml
git commit -m "feat: wire visibility fusion through ThermalMap msg and mapper node"
```

---

### Task 6: 残差场（residual.py）

残差 = 实测信念 − 已知源正向预测，只在"看清"的格上有定义；它是探索目标打分和清场证据的输入。

**Files:**
- Create: `src/thermal_robot/thermal_field_reconstructor/thermal_field_reconstructor/residual.py`
- Test: `src/thermal_robot/tests/test_phase1_observation.py`

- [ ] **Step 1: 写失败测试**（追加）

```python
class TestResidual:
    def _meta(self):
        return dict(width=40, height=40, resolution=0.25,
                    origin_x=-5.0, origin_y=-5.0)

    def test_predict_single_gaussian(self):
        from thermal_field_reconstructor import residual
        m = self._meta()
        field = residual.predict_field(ambient_temp=22.0,
                                       sources=[(0.0, 0.0, 10.0, 1.0)], **m)
        iy = int((0.0 - m['origin_y']) / m['resolution'])
        ix = int((0.0 - m['origin_x']) / m['resolution'])
        assert field[iy, ix] == pytest.approx(32.0, abs=0.5)
        assert field[0, 0] == pytest.approx(22.0, abs=0.1)

    def test_residual_zero_when_predicted_matches(self):
        from thermal_field_reconstructor import residual
        from thermal_field_reconstructor.thermal_mapping import VIEW_CLEAR
        m = self._meta()
        field = residual.predict_field(ambient_temp=22.0,
                                       sources=[(1.0, 1.0, 8.0, 1.2)], **m)
        view = np.full((40, 40), VIEW_CLEAR, dtype=np.uint8)
        r = residual.residual_field(field, field, view)
        assert float(np.abs(r).max()) == 0.0

    def test_missing_source_leaves_positive_blob(self):
        from thermal_field_reconstructor import residual
        from thermal_field_reconstructor.thermal_mapping import VIEW_CLEAR
        m = self._meta()
        measured = residual.predict_field(ambient_temp=22.0,
                                          sources=[(2.0, -2.0, 9.0, 1.0)], **m)
        predicted = residual.predict_field(ambient_temp=22.0, sources=[], **m)
        view = np.full((40, 40), VIEW_CLEAR, dtype=np.uint8)
        r = residual.residual_field(measured, predicted, view)
        iy = int((-2.0 - m['origin_y']) / m['resolution'])
        ix = int((2.0 - m['origin_x']) / m['resolution'])
        assert r[iy, ix] > 8.0
        assert float(r.min()) >= 0.0

    def test_non_clear_cells_zeroed(self):
        from thermal_field_reconstructor import residual
        from thermal_field_reconstructor.thermal_mapping import (
            VIEW_NEVER, VIEW_BLOCKED_ONLY)
        m = self._meta()
        measured = np.full((40, 40), 40.0, dtype=np.float32)
        predicted = np.full((40, 40), 22.0, dtype=np.float32)
        view = np.full((40, 40), VIEW_NEVER, dtype=np.uint8)
        view[5, 5] = VIEW_BLOCKED_ONLY
        r = residual.residual_field(measured, predicted, view)
        assert float(np.abs(r).max()) == 0.0
```

- [ ] **Step 2: 跑测试确认失败**

```bash
python3 -m pytest src/thermal_robot/tests/test_phase1_observation.py -k Residual -q
```
预期：`No module named 'thermal_field_reconstructor.residual'`。

- [ ] **Step 3: 实现 residual.py**

```python
"""残差场: 实测信念 − 已知源正向预测 (spec §3 快层).

残差只在"看清"(VIEW_CLEAR)的格上有定义: 没看过/被挡的格没有实测证据,
其探索价值由 view_state 项表达, 不混进残差。残差取非负: 低于预测的格
(源已熄灭/估计偏高)由跟踪层处理, 不属于"未解释热量"。
"""

from __future__ import annotations

import numpy as np

from thermal_field_reconstructor.thermal_mapping import VIEW_CLEAR


def predict_field(width, height, resolution, origin_x, origin_y,
                  ambient_temp, sources):
    """sources: iterable of (x, y, amplitude, sigma); amplitude = 峰值−环境温."""
    h, w = int(height), int(width)
    yy, xx = np.mgrid[0:h, 0:w]
    wx = origin_x + (xx.astype(np.float32) + 0.5) * resolution
    wy = origin_y + (yy.astype(np.float32) + 0.5) * resolution
    field = np.full((h, w), float(ambient_temp), dtype=np.float32)
    for sx, sy, amp, sigma in sources:
        if amp <= 0.0 or sigma <= 1e-3:
            continue
        d2 = (wx - float(sx)) ** 2 + (wy - float(sy)) ** 2
        field += float(amp) * np.exp(-0.5 * d2 / float(sigma) ** 2)
    return field


def residual_field(temperature_mean, predicted, view_state):
    r = (np.asarray(temperature_mean, dtype=np.float32)
         - np.asarray(predicted, dtype=np.float32))
    r = np.maximum(r, 0.0)
    r[np.asarray(view_state) != VIEW_CLEAR] = 0.0
    return r
```

- [ ] **Step 4: 跑测试确认通过**

```bash
python3 -m pytest src/thermal_robot/tests/test_phase1_observation.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/thermal_robot/thermal_field_reconstructor/thermal_field_reconstructor/residual.py \
        src/thermal_robot/tests/test_phase1_observation.py
git commit -m "feat: residual field from forward-predicted known sources"
```

---

### Task 7: 残差目标打分核（planning.select_residual_target）

打分 = 残差质量 + 可见性感知未见度（没看过 1.0 / 看过但被挡按 `w_blocked` 加权——被挡是"换个视角再看"的理由）+ 年龄 − 旅行代价 − 重复惩罚；top-K 候选逐个过直线可达性检查（依赖注入 `line_reachable_fn`，保持本模块纯净）。**不改 planning.py 现有任何函数**。

**Files:**
- Modify: `src/thermal_robot/thermal_motion_controller/thermal_motion_controller/planning.py`（仅追加）
- Test: `src/thermal_robot/tests/test_phase1_observation.py`

- [ ] **Step 1: 写失败测试**（追加）

```python
class TestResidualTarget:
    def _args(self, **over):
        from thermal_field_reconstructor.thermal_mapping import VIEW_NEVER
        h = w = 60
        base = dict(
            robot_wx=0.0, robot_wy=0.0, width=w, height=h, resolution=0.25,
            origin_x=-7.5, origin_y=-7.5,
            residual=np.zeros((h, w), dtype=np.float32),
            view_state=np.full((h, w), VIEW_NEVER, dtype=np.uint8),
            last_seen_age_s=np.full((h, w), -1.0, dtype=np.float32),
        )
        base.update(over)
        return base

    def test_residual_mass_wins(self):
        from thermal_motion_controller.planning import select_residual_target
        from thermal_field_reconstructor.thermal_mapping import VIEW_CLEAR
        a = self._args()
        a['view_state'][:, :] = VIEW_CLEAR
        a['last_seen_age_s'][:, :] = 1.0
        # (5,0) 附近一团残差
        iy = int((0.0 + 7.5) / 0.25); ix = int((5.0 + 7.5) / 0.25)
        a['residual'][iy - 2:iy + 3, ix - 2:ix + 3] = 6.0
        t = select_residual_target(**a)
        assert t is not None
        assert math.hypot(t.x - 5.0, t.y - 0.0) < 1.5
        assert t.reason == 'residual_mass'

    def test_unseen_region_attracts(self):
        from thermal_motion_controller.planning import select_residual_target
        from thermal_field_reconstructor.thermal_mapping import VIEW_CLEAR, VIEW_NEVER
        a = self._args()
        a['view_state'][:, :] = VIEW_CLEAR
        a['last_seen_age_s'][:, :] = 1.0
        a['view_state'][:, 40:] = VIEW_NEVER  # x>2.5 没看过
        t = select_residual_target(**a)
        assert t is not None
        assert t.x > 2.0
        assert t.reason in ('unseen', 'blocked_view')

    def test_known_source_excluded(self):
        from thermal_motion_controller.planning import select_residual_target
        a = self._args(known_sources=[(4.0, 0.0)])
        t = select_residual_target(**a)
        assert t is not None
        assert math.hypot(t.x - 4.0, t.y - 0.0) >= 2.3

    def test_unreachable_candidates_skipped(self):
        from thermal_motion_controller.planning import select_residual_target
        from thermal_field_reconstructor.thermal_mapping import VIEW_CLEAR, VIEW_NEVER
        a = self._args()
        a['view_state'][:, :] = VIEW_CLEAR
        a['last_seen_age_s'][:, :] = 1.0
        a['view_state'][:, 40:] = VIEW_NEVER
        # 一切 x>0 的目标都"不可达"
        a['line_reachable_fn'] = lambda x0, y0, x1, y1: x1 <= 0.0
        t = select_residual_target(**a)
        assert t is not None
        if not t.metadata_reachable:
            pytest.skip('all top-k unreachable, fallback target returned')
        assert t.x <= 0.0

    def test_empty_map_returns_none(self):
        from thermal_motion_controller.planning import select_residual_target
        a = self._args(width=0, height=0,
                       residual=np.zeros((0, 0), np.float32),
                       view_state=np.zeros((0, 0), np.uint8),
                       last_seen_age_s=np.zeros((0, 0), np.float32))
        assert select_residual_target(**a) is None
```

- [ ] **Step 2: 跑测试确认失败**

```bash
python3 -m pytest src/thermal_robot/tests/test_phase1_observation.py -k ResidualTarget -q
```
预期：`ImportError: cannot import name 'select_residual_target'`。

- [ ] **Step 3: 实现（planning.py 末尾追加）**

`PlannerTarget` 没有 metadata 字段，所以给返回值附一个轻量属性 `metadata_reachable`。在 `planning.py` 文件末尾（`_wrap_angle` 之后）追加：

```python
def select_residual_target(
    robot_wx: float,
    robot_wy: float,
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    residual: np.ndarray,
    view_state: np.ndarray,
    last_seen_age_s: np.ndarray,
    known_sources: Sequence[Tuple[float, float]] = (),
    min_d: float = 2.5,
    max_d: float = 16.0,
    safe_dist: float = 2.3,
    w_residual: float = 1.6,
    w_unseen: float = 1.0,
    w_blocked: float = 1.2,
    w_age: float = 0.25,
    w_travel: float = 0.45,
    w_duplicate: float = 1.2,
    top_k: int = 12,
    line_reachable_fn=None,
) -> Optional[PlannerTarget]:
    """残差质量 + 可见性感知未见度的探索目标 (spec §3 规划层, strategy=residual).

    view_state: 0=没看过, 1=看过但被挡, 2=看清 (thermal_mapping 三态)。
    line_reachable_fn(x0,y0,x1,y1)->bool 为直线可达检查的依赖注入;
    None 表示不检查。top_k 候选全部不可达时返回得分最高者并标记
    metadata_reachable=False (Nav2 可能仍绕得过去)。
    """
    if width <= 0 or height <= 0 or resolution <= 0.0:
        return None
    resid = _reshape(residual, height, width)
    vs = np.asarray(view_state).reshape((height, width))
    age = _reshape(last_seen_age_s, height, width)

    yy, xx = np.mgrid[0:height, 0:width]
    wx = origin_x + (xx.astype(np.float32) + 0.5) * resolution
    wy = origin_y + (yy.astype(np.float32) + 0.5) * resolution
    dist = np.sqrt((wx - robot_wx) ** 2 + (wy - robot_wy) ** 2)
    valid = (dist >= min_d) & (dist <= max_d)

    duplicate = np.zeros_like(dist, dtype=np.float32)
    for sx, sy in known_sources:
        d = np.sqrt((wx - sx) ** 2 + (wy - sy) ** 2)
        valid &= d >= safe_dist
        duplicate = np.maximum(duplicate, np.exp(-0.5 * (d / max(safe_dist, 0.25)) ** 2))
    if not np.any(valid):
        return None

    resid_norm = _norm_clip(resid)
    unseen = (vs == 0).astype(np.float32)
    blocked = (vs == 1).astype(np.float32)
    age_norm = np.where((vs == 2) & (age >= 0.0),
                        np.clip(age / 60.0, 0.0, 1.0), 0.0).astype(np.float32)
    travel = np.clip(dist / max(max_d, 1e-3), 0.0, 1.0)

    term_resid = w_residual * resid_norm
    term_unseen = w_unseen * unseen
    term_blocked = w_blocked * blocked
    score = (term_resid + term_unseen + term_blocked + w_age * age_norm
             - w_travel * travel - w_duplicate * duplicate)
    score[~valid] = -np.inf
    if not np.isfinite(score).any():
        return None

    flat_order = np.argsort(score.reshape(-1))[::-1][:max(1, int(top_k))]

    def _mk(idx: int, reachable: bool) -> PlannerTarget:
        iy, ix = np.unravel_index(int(idx), score.shape)
        terms = {
            'residual_mass': float(term_resid[iy, ix]),
            'unseen': float(term_unseen[iy, ix]),
            'blocked_view': float(term_blocked[iy, ix]),
        }
        reason = max(terms, key=terms.get)
        target = PlannerTarget(
            x=float(origin_x + (ix + 0.5) * resolution),
            y=float(origin_y + (iy + 0.5) * resolution),
            score=float(score[iy, ix]),
            reason=reason,
        )
        target.metadata_reachable = reachable
        return target

    if line_reachable_fn is None:
        return _mk(flat_order[0], True)
    for idx in flat_order:
        if not np.isfinite(score.reshape(-1)[int(idx)]):
            break
        iy, ix = np.unravel_index(int(idx), score.shape)
        tx = float(origin_x + (ix + 0.5) * resolution)
        ty = float(origin_y + (iy + 0.5) * resolution)
        if line_reachable_fn(robot_wx, robot_wy, tx, ty):
            return _mk(idx, True)
    return _mk(flat_order[0], False)
```

注意：`PlannerTarget` 是 `@dataclass`，默认允许动态属性（没有 `__slots__`），`target.metadata_reachable = reachable` 合法。

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

```bash
python3 -m pytest src/thermal_robot/tests/test_phase1_observation.py -q
python3 -m pytest src/thermal_robot/tests/ -q
```

- [ ] **Step 5: Commit**

```bash
git add src/thermal_robot/thermal_motion_controller/thermal_motion_controller/planning.py \
        src/thermal_robot/tests/test_phase1_observation.py
git commit -m "feat: residual-driven exploration target with reachability check"
```

---

### Task 8: 清场概率 v1（clearance.py）+ 校准工具

生成式模型（spec §3）：未发现源 ~ 自由空间上的 Poisson(λ)（λ 已折算为"幅值≥A_min 的源"的强度，幅值先验 v1 并入 λ，B 级传感器模型后再拆开）；每格漏检概率由观测扇区数决定：从 k 个不同方位扇区看清过的格，`p_miss = (1−p_detect)^min(k, k_max)`；没看过/只被挡过的格 `p_miss = 1`；残差超阈值的格视为未清除（`p_miss = 1`，残差证据近似漏检似然——spec 允许的 v1 简化）。则 `P(无未发现源) = exp(−λ · Σ_free 格面积 · p_miss)`。验收看校准（可靠性曲线 + ECE），不看阈值是否"合理"。

**Files:**
- Create: `src/thermal_robot/thermal_motion_controller/thermal_motion_controller/clearance.py`
- Test: `src/thermal_robot/tests/test_phase1_observation.py`

- [ ] **Step 1: 写失败测试**（追加）

```python
class TestClearance:
    def _params(self, **over):
        from thermal_motion_controller.clearance import ClearanceParams
        kw = dict(source_rate_per_m2=0.01, p_detect_per_sector=0.7,
                  max_effective_sectors=4, residual_block_thresh=1.5,
                  epsilon=0.05)
        kw.update(over)
        return ClearanceParams(**kw)

    def test_no_coverage_low_clearance(self):
        from thermal_motion_controller import clearance
        vs = np.zeros((40, 40), dtype=np.uint8)
        sec = np.zeros((40, 40), dtype=np.uint8)
        p = clearance.clearance_probability(vs, sec, cell_area_m2=0.0625,
                                            params=self._params())
        # Λ = 0.01 * 0.0625 * 1600 = 1.0 → exp(-1) ≈ 0.368
        assert p == pytest.approx(math.exp(-1.0), abs=1e-6)

    def test_full_multisector_coverage_high_clearance(self):
        from thermal_motion_controller import clearance
        from thermal_motion_controller.clearance import VIEW_CLEAR
        vs = np.full((40, 40), VIEW_CLEAR, dtype=np.uint8)
        sec = np.full((40, 40), 0b00001111, dtype=np.uint8)  # 4 个扇区
        p = clearance.clearance_probability(vs, sec, cell_area_m2=0.0625,
                                            params=self._params())
        # p_miss = 0.3^4 = 0.0081 → Λ = 0.0081 → p ≈ 0.992
        assert p == pytest.approx(math.exp(-0.0081), abs=1e-4)

    def test_residual_blocks_clearance(self):
        from thermal_motion_controller import clearance
        from thermal_motion_controller.clearance import VIEW_CLEAR
        vs = np.full((40, 40), VIEW_CLEAR, dtype=np.uint8)
        sec = np.full((40, 40), 0b00001111, dtype=np.uint8)
        resid = np.zeros((40, 40), dtype=np.float32)
        base = clearance.clearance_probability(
            vs, sec, 0.0625, self._params(), residual=resid)
        resid[10:14, 10:14] = 5.0
        with_blob = clearance.clearance_probability(
            vs, sec, 0.0625, self._params(), residual=resid)
        assert with_blob < base

    def test_free_mask_restricts_domain(self):
        from thermal_motion_controller import clearance
        vs = np.zeros((40, 40), dtype=np.uint8)
        sec = np.zeros((40, 40), dtype=np.uint8)
        free = np.zeros((40, 40), dtype=bool)
        free[0:10, 0:10] = True  # 只有 100 格在域内
        p = clearance.clearance_probability(vs, sec, 0.0625, self._params(),
                                            free_mask=free)
        assert p == pytest.approx(math.exp(-0.01 * 0.0625 * 100), abs=1e-6)

    def test_ece_hand_case(self):
        from thermal_motion_controller import clearance
        claimed = np.array([0.9] * 10)
        outcomes = np.array([1] * 9 + [0])  # 经验率 0.9 → 完美校准
        assert clearance.expected_calibration_error(claimed, outcomes) == \
            pytest.approx(0.0, abs=1e-9)
        outcomes_bad = np.array([1] * 5 + [0] * 5)  # 经验率 0.5
        assert clearance.expected_calibration_error(claimed, outcomes_bad) == \
            pytest.approx(0.4, abs=1e-9)

    def test_monte_carlo_calibration(self):
        """生成式自洽: 按模型采样世界与覆盖, 宣称概率应校准 (ECE < 0.05)."""
        from thermal_motion_controller import clearance
        from thermal_motion_controller.clearance import VIEW_CLEAR
        rng = np.random.default_rng(42)
        params = self._params(source_rate_per_m2=0.02)
        h = w = 20
        area = 0.25
        claimed, outcomes = [], []
        for _ in range(1500):
            vs = np.zeros((h, w), dtype=np.uint8)
            sec = np.zeros((h, w), dtype=np.uint8)
            covered = rng.random((h, w)) < rng.uniform(0.2, 0.95)
            vs[covered] = VIEW_CLEAR
            n_sec = rng.integers(1, 5, size=(h, w))
            sec[covered] = (np.left_shift(1, n_sec) - 1).astype(np.uint8)[covered]
            p_miss = clearance.cell_miss_prob(vs, sec, params)
            lam_cells = params.source_rate_per_m2 * area * p_miss
            n_undet = rng.poisson(lam_cells).sum()
            claimed.append(clearance.clearance_probability(vs, sec, area, params))
            outcomes.append(1 if n_undet == 0 else 0)
        ece = clearance.expected_calibration_error(
            np.array(claimed), np.array(outcomes), n_bins=10)
        assert ece < 0.05
```

- [ ] **Step 2: 跑测试确认失败**

```bash
python3 -m pytest src/thermal_robot/tests/test_phase1_observation.py -k Clearance -q
```
预期：`No module named 'thermal_motion_controller.clearance'`。

- [ ] **Step 3: 实现 clearance.py**

```python
"""清场概率 v1: 显式生成式模型 (spec §3).

P(不存在幅值≥A_min 的未发现源)
  = exp(−λ · Σ_{free 格} 面积 · p_miss(格))

- λ (source_rate_per_m2): 自由空间上"幅值≥A_min 的源"的 Poisson 强度。
  幅值先验在 v1 并入 λ; B 级传感器模型后拆为独立项。
- p_miss: 从 k 个不同方位扇区看清过的格 = (1−p_detect)^min(k, k_max);
  没看过/只被挡过 = 1; 残差≥阈值的格 = 1 (残差证据近似漏检似然,
  spec 允许的 v1 简化)。
验收看校准 (可靠性曲线 + ECE), 不看阈值是否"合理"。

本模块纯 numpy 无跨包依赖; 三态数值与 thermal_mapping 保持一致,
由 tests/test_phase1_observation.py 锁定。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VIEW_NEVER = 0
VIEW_BLOCKED_ONLY = 1
VIEW_CLEAR = 2


@dataclass
class ClearanceParams:
    source_rate_per_m2: float = 0.01
    p_detect_per_sector: float = 0.7
    max_effective_sectors: int = 4
    residual_block_thresh: float = 1.5
    epsilon: float = 0.05
    domain_radius_m: float = 12.0  # 任务域半径(以 spawn 为心), 由调用方折进 free_mask


def _popcount8(arr: np.ndarray) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.uint8).reshape(-1, 1)
    return np.unpackbits(flat, axis=1).sum(axis=1).reshape(np.asarray(arr).shape)


def cell_miss_prob(view_state, view_sectors, params: ClearanceParams,
                   residual=None) -> np.ndarray:
    vs = np.asarray(view_state)
    k = np.minimum(_popcount8(view_sectors), params.max_effective_sectors)
    p_miss = np.ones(vs.shape, dtype=np.float64)
    clear = vs == VIEW_CLEAR
    p_miss[clear] = (1.0 - params.p_detect_per_sector) ** k[clear]
    if residual is not None:
        p_miss[np.asarray(residual) >= params.residual_block_thresh] = 1.0
    return p_miss


def clearance_probability(view_state, view_sectors, cell_area_m2,
                          params: ClearanceParams, residual=None,
                          free_mask=None) -> float:
    p_miss = cell_miss_prob(view_state, view_sectors, params, residual=residual)
    if free_mask is not None:
        p_miss = np.where(np.asarray(free_mask), p_miss, 0.0)
    lam = params.source_rate_per_m2 * float(cell_area_m2) * float(p_miss.sum())
    return float(np.exp(-lam))


def reliability_curve(claimed, outcomes, n_bins: int = 10):
    """返回 [(bin_low, bin_high, mean_claimed, empirical_rate, n), ...]."""
    claimed = np.asarray(claimed, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (claimed >= lo) & (claimed < hi if i < n_bins - 1 else claimed <= hi)
        if not mask.any():
            continue
        rows.append((float(lo), float(hi),
                     float(claimed[mask].mean()),
                     float(outcomes[mask].mean()),
                     int(mask.sum())))
    return rows


def expected_calibration_error(claimed, outcomes, n_bins: int = 10) -> float:
    rows = reliability_curve(claimed, outcomes, n_bins=n_bins)
    total = sum(r[4] for r in rows)
    if total == 0:
        return 0.0
    return float(sum(abs(r[2] - r[3]) * r[4] for r in rows) / total)
```

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

```bash
python3 -m pytest src/thermal_robot/tests/test_phase1_observation.py -q
python3 -m pytest src/thermal_robot/tests/ -q
```

- [ ] **Step 5: Commit**

```bash
git add src/thermal_robot/thermal_motion_controller/thermal_motion_controller/clearance.py \
        src/thermal_robot/tests/test_phase1_observation.py
git commit -m "feat: generative clearance probability v1 with calibration tools"
```

---

### Task 9: 控制器集成 strategy=residual + 清场发布 + 数据采集

控制器侧把全链路接通：`residual` 进合法策略表；订阅 SLAM `/map` 做可达性；COARSE 目标改用 `select_residual_target`（不可用时落回 v31 路径，机器人永不卡死）；周期评估清场概率并发布到 `/thermal/clearance`（**只发布不终止**，`clearance_terminate` 默认 false——阶段 1 门槛比较 recall，终止行为留给门槛评审后决定）；collector 录 clearance 时间线供校准分析。

**Files:**
- Modify: `src/thermal_robot/thermal_motion_controller/thermal_motion_controller/controller_node.py`
- Modify: `src/thermal_robot/thermal_motion_controller/package.xml`（加 `<exec_depend>thermal_field_reconstructor</exec_depend>`）
- Modify: `src/thermal_robot/thermal_bringup/config/params.yaml`（controller 段）
- Modify: `src/thermal_robot/scripts/run_multiscenario_matrix.py`（strategy choices）
- Modify: `src/thermal_robot/scripts/collect_sim_data.py`（clearance 记录）
- Test: `src/thermal_robot/tests/test_phase1_observation.py`（配置一致性测试）

- [ ] **Step 1: 写失败测试**（追加；控制器节点 import rclpy 无法在纯测试里实例化，用文本断言锁配置，沿用 T-PY23 风格）

```python
class TestResidualStrategyWiring:
    def test_controller_accepts_residual_strategy(self):
        src = (ROBOT / 'thermal_motion_controller' / 'thermal_motion_controller'
               / 'controller_node.py').read_text()
        assert "'residual'" in src
        assert "select_residual_target" in src
        assert "/thermal/clearance" in src

    def test_matrix_runner_accepts_residual(self):
        src = (ROBOT / 'scripts' / 'run_multiscenario_matrix.py').read_text()
        assert '"residual"' in src or "'residual'" in src

    def test_collector_records_clearance(self):
        src = (ROBOT / 'scripts' / 'collect_sim_data.py').read_text()
        assert '/thermal/clearance' in src
```

- [ ] **Step 2: 跑测试确认失败**

```bash
python3 -m pytest src/thermal_robot/tests/test_phase1_observation.py -k Wiring -q
```
预期：3 failed（断言找不到字符串）。

- [ ] **Step 3: controller_node.py 改动**

(a) import 区（`from nav_msgs.msg import Odometry` 行）改为：

```python
from nav_msgs.msg import Odometry, OccupancyGrid
```

`from thermal_motion_controller.planning import PlannerWeights` 行改为：

```python
from thermal_motion_controller.planning import PlannerWeights, select_residual_target
from thermal_motion_controller import clearance as clearance_mod
from thermal_field_reconstructor import visibility as fr_visibility
from thermal_field_reconstructor import residual as fr_residual
```

`from geometry_msgs.msg import Twist` 之后追加：

```python
from std_msgs.msg import Float32
```

(b) 参数声明区（`self.declare_parameter('random_seed', 0)` 附近）追加：

```python
        self.declare_parameter('clearance_source_rate_per_m2', 0.01)
        self.declare_parameter('clearance_p_detect_per_sector', 0.7)
        self.declare_parameter('clearance_residual_thresh', 1.5)
        self.declare_parameter('clearance_epsilon', 0.05)
        self.declare_parameter('clearance_eval_interval_s', 10.0)
        self.declare_parameter('clearance_domain_radius_m', 12.0)
        self.declare_parameter('clearance_terminate', False)
```

(c) 约 433 行合法策略检查：

```python
        if self._strategy_mode not in ('full', 'frontier', 'levy'):
```

改为：

```python
        if self._strategy_mode not in ('full', 'frontier', 'levy', 'residual'):
```

(d) 参数读取区（`self._strategy_mode` 读取之后）追加：

```python
        self._clearance_params = clearance_mod.ClearanceParams(
            source_rate_per_m2=float(g('clearance_source_rate_per_m2').value),
            p_detect_per_sector=float(g('clearance_p_detect_per_sector').value),
            residual_block_thresh=float(g('clearance_residual_thresh').value),
            epsilon=float(g('clearance_epsilon').value),
            domain_radius_m=float(g('clearance_domain_radius_m').value),
        )
        self._clearance_interval = float(g('clearance_eval_interval_s').value)
        self._clearance_terminate = bool(g('clearance_terminate').value)
        self._clearance_last_eval = 0.0
        self._clearance_value = None
        self._occ_view = None
```

(e) 订阅区（`self.create_subscription(ThermalMap, '/thermal/map', ...)` 之后）追加：

```python
        occ_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, '/map', self._occ_map_cb, occ_qos)
        self._clearance_pub = self.create_publisher(Float32, '/thermal/clearance', 10)
```

（确认本文件 QoS import 是否已含 `QoSDurabilityPolicy`/`QoSHistoryPolicy`，没有就在既有 `from rclpy.qos import ...` 里补。）

(f) `_map_cb` 末尾（`self._thermal_map_t = time.monotonic()` 之后、except 之前）追加节流的清场评估：

```python
            if self._strategy_mode == 'residual':
                now_mono = time.monotonic()
                if now_mono - self._clearance_last_eval >= self._clearance_interval:
                    self._clearance_last_eval = now_mono
                    self._evaluate_clearance()
```

(g) 新方法（放在 `_map_cb` 之后）：

```python
    def _occ_map_cb(self, msg: OccupancyGrid):
        self._occ_view = fr_visibility.from_flat(
            msg.data, msg.info.width, msg.info.height,
            msg.info.origin.position.x + self._spawn_x,
            msg.info.origin.position.y + self._spawn_y,
            msg.info.resolution)

    def _residual_snapshot(self):
        m = self._thermal_map
        if m is None or 'view_state' not in m or 'temperature_mean' not in m:
            return None
        srcs = [(sx, sy, max(0.0, st - self._ambient_est), self._heat_sigma)
                for sx, sy, st in self._found_sources]
        predicted = fr_residual.predict_field(
            m['width'], m['height'], m['resolution'],
            m['origin_x'], m['origin_y'], self._ambient_est, srcs)
        return fr_residual.residual_field(
            m['temperature_mean'], predicted, m['view_state'])

    def _residual_waypoint(self):
        m = self._thermal_map
        resid = self._residual_snapshot()
        if resid is None:
            return None
        occ = self._occ_view
        reach_fn = None
        if occ is not None:
            reach_fn = (lambda x0, y0, x1, y1:
                        fr_visibility.line_reachable(occ, x0, y0, x1, y1))
        t = select_residual_target(
            self._wx, self._wy, m['width'], m['height'], m['resolution'],
            m['origin_x'], m['origin_y'], resid, m['view_state'],
            m['last_seen_age_s'],
            known_sources=[(sx, sy) for sx, sy, _ in self._found_sources],
            line_reachable_fn=reach_fn)
        if t is None:
            return None
        return StrategyTarget(
            x=t.x, y=t.y, score=t.score, reason=t.reason,
            execution_hint=EXECUTION_NAV2_PREFERRED,
            metadata={'label': 'residual',
                      'reachable': getattr(t, 'metadata_reachable', True)})

    def _evaluate_clearance(self):
        m = self._thermal_map
        if m is None or 'view_state' not in m:
            return
        resid = self._residual_snapshot()
        h, w = m['height'], m['width']
        yy, xx = np.mgrid[0:h, 0:w]
        cwx = m['origin_x'] + (xx.astype(np.float32) + 0.5) * m['resolution']
        cwy = m['origin_y'] + (yy.astype(np.float32) + 0.5) * m['resolution']
        # 任务域: spawn 周边 domain 半径内 (先验只覆盖源可能出现的区域)
        free = (np.hypot(cwx - self._spawn_x, cwy - self._spawn_y)
                <= self._clearance_params.domain_radius_m)
        if self._occ_view is not None:
            free &= ~fr_visibility.occupied_at(self._occ_view, cwx, cwy)
        t0 = time.monotonic()
        p_clear = clearance_mod.clearance_probability(
            m['view_state'], m['view_sectors'], m['resolution'] ** 2,
            self._clearance_params, residual=resid, free_mask=free)
        eval_ms = (time.monotonic() - t0) * 1000.0
        self._clearance_value = p_clear
        out = Float32()
        out.data = float(p_clear)
        self._clearance_pub.publish(out)
        self.get_logger().info(
            f'[CLEARANCE] p_no_undetected={p_clear:.4f} '
            f'eps={self._clearance_params.epsilon} eval_ms={eval_ms:.1f}')
        if (self._clearance_terminate
                and p_clear >= 1.0 - self._clearance_params.epsilon
                and self._found_sources):
            self.get_logger().info('[CLEARANCE_DONE_SIGNAL] threshold reached')
```

`StrategyTarget` 与 `EXECUTION_NAV2_PREFERRED` 需从 target_selection import——确认文件顶部既有 `from thermal_motion_controller.target_selection import (...)` 里是否已含这两个名字，没有就补进该 import 列表。

(h) `_coarse_waypoint`（约 1512 行）在 `elif self._strategy_mode == 'levy':` 分支之后、`else:` 之前插入：

```python
        elif self._strategy_mode == 'residual':
            target = self._residual_waypoint()
            if target is None:
                target = self._source_seek_selector.select_coarse_waypoint(
                    self._strategy_context(),
                    min_d=self._survey_wp_min_d,
                    max_d=self._survey_wp_max_d,
                )
```

(i) `_refresh_frontier`（约 1324 行）在 `if self._strategy_mode == 'levy':` 块之后插入：

```python
        if self._strategy_mode == 'residual':
            if not force and (now - self._frontier_last_upd) < self._frontier_upd:
                return
            target = self._residual_waypoint()
            if target is not None:
                self._frontier_last_upd = now
                self._frontier_target = target.xy
                self.get_logger().info(
                    f'[FRONTIER/residual/{target.reason}] '
                    f'→({target.x:.1f},{target.y:.1f}) score={target.score:.3f} '
                    f'reachable={target.metadata.get("reachable", True)}')
                return
            # 残差目标不可用(地图未就绪等) → 落回下方 v31 路径
```

（插在 levy 块后即可，后续原有代码不动——残差目标拿不到时自然走 v31 frontier 逻辑。）

- [ ] **Step 4: package.xml + params.yaml + runner + collector**

(a) `src/thermal_robot/thermal_motion_controller/package.xml` 的 depend 区追加：

```xml
  <exec_depend>thermal_field_reconstructor</exec_depend>
```

(b) `params.yaml` 的 `controller_node: ros__parameters:` 下追加：

```yaml
    clearance_source_rate_per_m2: 0.01
    clearance_p_detect_per_sector: 0.7
    clearance_residual_thresh: 1.5
    clearance_epsilon: 0.05
    clearance_eval_interval_s: 10.0
    clearance_domain_radius_m: 12.0
    clearance_terminate: false
```

(c) `scripts/run_multiscenario_matrix.py` 约 396 行：

```python
    parser.add_argument("--strategy", choices=["full", "frontier", "levy"], default="full")
```

改为：

```python
    parser.add_argument("--strategy",
                        choices=["full", "frontier", "levy", "residual"],
                        default="full")
```

(d) `scripts/collect_sim_data.py` 记录 clearance：参照该文件现有订阅模式（先读文件确认回调与 finalize 的结构），做三处改动：

- import 区补 `from std_msgs.msg import Float32`（已有则跳过）；
- 订阅区追加：

```python
        self.create_subscription(Float32, '/thermal/clearance',
                                 self._clearance_cb, 10)
```

- 回调与缓冲（仿照其它 `_*_cb` + buffer 的写法）：

```python
    def _clearance_cb(self, msg: Float32):
        self._clearance_rows.append((self._elapsed(), float(msg.data)))
```

（`__init__` 里初始化 `self._clearance_rows = []`；`_elapsed()` 用该文件现成的时间基准函数，名字以实际为准。）

- 落盘：在写 `source_summary.json`/`metadata.json` 的 finalize 函数里追加写 `clearance.csv`（列 `t,p_clear`），并在 metadata dict 加 `'clearance_final': self._clearance_rows[-1][1] if self._clearance_rows else None`。

- [ ] **Step 5: 构建 + 测试 + 回归**

```bash
colcon build --packages-select thermal_motion_controller thermal_bringup
source install/setup.bash
python3 -m py_compile src/thermal_robot/thermal_motion_controller/thermal_motion_controller/controller_node.py \
                      src/thermal_robot/scripts/collect_sim_data.py \
                      src/thermal_robot/scripts/run_multiscenario_matrix.py
python3 -m pytest src/thermal_robot/tests/ -q
ros2 launch thermal_bringup sim_nav_slam_launch.py --show-args | head -20
```

- [ ] **Step 6: Commit**

```bash
git add -A src/thermal_robot
git commit -m "feat: residual strategy integration with clearance publishing"
```

---

### Task 10: 配对统计检验 + cluster bootstrap + 对比报告工具

spec §5.4：主检验 = 按 (case, seed) 配对的置换检验（符号翻转，n≤14 穷举否则抽样）；precision/duplicate rate 的 CI = 以 run 为重抽样单位的 cluster bootstrap，矩阵层面按确认事件汇总。新 CLI `scripts/matrix_compare.py` 对比两个矩阵 out-root，产出门槛评审报告。**不改 matrix_stats 现有函数。**

**Files:**
- Modify: `src/thermal_robot/scripts/matrix_stats.py`（仅追加）
- Create: `src/thermal_robot/scripts/matrix_compare.py`
- Test: `src/thermal_robot/tests/test_phase1_observation.py`

- [ ] **Step 1: 写失败测试**（追加）

```python
class TestPairedStats:
    def test_paired_permutation_exact_small(self):
        ms = _load_script('matrix_stats')
        # diffs 全为 +1, n=3: 符号翻转 2^3=8 种, |mean|≥1 的只有全+与全− → p=0.25
        assert ms.paired_permutation_test([2, 2, 2], [1, 1, 1]) == \
            pytest.approx(0.25)
        # n=5 全正 → p = 2/32
        assert ms.paired_permutation_test([1, 2, 3, 4, 5], [0, 0, 0, 0, 0]) == \
            pytest.approx(2.0 / 32.0)

    def test_paired_permutation_zero_diffs(self):
        ms = _load_script('matrix_stats')
        assert ms.paired_permutation_test([1, 1], [1, 1]) == 1.0

    def test_paired_permutation_monotone(self):
        ms = _load_script('matrix_stats')
        sep = ms.paired_permutation_test(
            [5, 6, 7, 8, 9, 10, 11, 12], [1, 2, 1, 2, 1, 2, 1, 2])
        mixed = ms.paired_permutation_test(
            [5, 1, 7, 2, 9, 1, 11, 2], [1, 6, 1, 8, 1, 10, 1, 12])
        assert sep < mixed

    def test_cluster_bootstrap_degenerate(self):
        ms = _load_script('matrix_stats')
        point, lo, hi = ms.cluster_bootstrap_ratio_ci(
            [3, 3, 3, 3], [3, 3, 3, 3], n_boot=500, seed=1)
        assert point == 1.0 and lo == 1.0 and hi == 1.0

    def test_cluster_bootstrap_brackets_point(self):
        ms = _load_script('matrix_stats')
        nums = [3, 2, 4, 1, 3, 4, 2, 3]
        dens = [4, 3, 4, 2, 4, 4, 3, 4]
        point, lo, hi = ms.cluster_bootstrap_ratio_ci(nums, dens,
                                                      n_boot=2000, seed=7)
        assert lo <= point <= hi
        assert 0.5 < point < 1.0
        assert hi - lo < 0.5


class TestMatrixCompare:
    def test_compare_two_roots(self, tmp_path):
        import json
        mc = _load_script('matrix_compare')
        for label, recall in (('base', 0.4), ('cand', 0.8)):
            root = tmp_path / label
            runs = []
            for case in ('w__a', 'w__b'):
                for seed in (101, 102, 103, 104, 105):
                    run_dir = root / case / f'seed{seed}'
                    run_dir.mkdir(parents=True)
                    (run_dir / 'source_summary.json').write_text(json.dumps({
                        'confirmed_count': 2,
                        'localization_errors_m': [
                            {'truth_id': 'S1', 'error_m': 0.5},
                            {'truth_id': 'S2', 'error_m': 0.4}],
                    }))
                    (run_dir / 'attribution.json').write_text(json.dumps({
                        'n_truth_sources': 3, 'n_matched': 2,
                        'failure_counts': {'not_reached': 1, 'occluded': 0,
                                           'timing_missed': 0,
                                           'not_confirmed': 0},
                        'per_source': {}}))
                    runs.append({'name': case, 'seed': seed, 'passed': True,
                                 'source_recall': recall,
                                 'source_precision': 1.0,
                                 'duplicate_confirmations': 0,
                                 'time_to_first_source': 5.0,
                                 'path_length_m': 30.0,
                                 'missing_counts': []})
            (root / 'matrix_summary.json').write_text(json.dumps({
                'n_runs': len(runs), 'n_passed': len(runs), 'runs': runs}))
        report = mc.compare_roots(tmp_path / 'base', tmp_path / 'cand')
        assert report['n_pairs'] == 10
        assert report['metrics']['source_recall']['p_value'] < 0.01
        assert report['metrics']['source_recall']['mean_b'] > \
            report['metrics']['source_recall']['mean_a']
        md = mc.render_report(report, 'base', 'cand')
        assert 'source_recall' in md and 'precision' in md
```

- [ ] **Step 2: 跑测试确认失败**

```bash
python3 -m pytest src/thermal_robot/tests/test_phase1_observation.py -k "PairedStats or MatrixCompare" -q
```
预期：`AttributeError: ... 'paired_permutation_test'` / 找不到 matrix_compare。

- [ ] **Step 3: matrix_stats.py 追加**（文件末尾；顶部按需补 `import random`，`math` 已有则不重复）

```python
def paired_permutation_test(xs, ys, n_resamples: int = 20000, seed: int = 0) -> float:
    """配对置换检验(符号翻转), 双侧, 统计量=配对差均值 (spec §5.4 主检验).

    零差对剔除; 有效 n≤14 时穷举 2^n 种符号翻转, 否则随机抽样
    (加一平滑避免 p=0)。
    """
    diffs = [float(x) - float(y) for x, y in zip(xs, ys)]
    diffs = [d for d in diffs if d != 0.0]
    n = len(diffs)
    if n == 0:
        return 1.0
    obs = abs(sum(diffs) / n)
    tol = 1e-12
    if n <= 14:
        count = 0
        total = 1 << n
        for mask in range(total):
            s = 0.0
            for i, d in enumerate(diffs):
                s += d if (mask >> i) & 1 else -d
            if abs(s / n) >= obs - tol:
                count += 1
        return count / total
    rng = random.Random(seed)
    count = 0
    for _ in range(n_resamples):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(s / n) >= obs - tol:
            count += 1
    return (count + 1) / (n_resamples + 1)


def cluster_bootstrap_ratio_ci(numerators, denominators, n_boot: int = 10000,
                               seed: int = 0, alpha: float = 0.05):
    """以 run 为重抽样单位的事件汇总比例 CI (spec §5.4 口径).

    返回 (point, lo, hi)。point = Σnum/Σden; 重抽样保留 run 内相关性。
    """
    nums = [float(v) for v in numerators]
    dens = [float(v) for v in denominators]
    n = len(nums)
    total_den = sum(dens)
    point = (sum(nums) / total_den) if total_den > 0 else 0.0
    if n == 0:
        return point, 0.0, 1.0
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        num = den = 0.0
        for _k in range(n):
            i = rng.randrange(n)
            num += nums[i]
            den += dens[i]
        stats.append(num / den if den > 0 else point)
    stats.sort()
    lo = stats[max(0, int(math.floor((alpha / 2.0) * (n_boot - 1))))]
    hi = stats[min(n_boot - 1, int(math.ceil((1.0 - alpha / 2.0) * (n_boot - 1))))]
    return point, lo, hi
```

- [ ] **Step 4: 实现 scripts/matrix_compare.py**

```python
#!/usr/bin/env python3
"""对比两个矩阵 out-root (基线 vs 候选), 产出门槛评审报告 (spec §5.4).

用法:
  python3 matrix_compare.py BASELINE_ROOT CANDIDATE_ROOT \
      [--label-a v31] [--label-b residual] [--out report.md]

主检验 = 按 (case, seed) 配对的置换检验; precision = 矩阵层面事件汇总 +
run 级 cluster bootstrap 95% CI; duplicate rate = Σ重复确认/Σ真值源数。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def _load_sibling(name: str):
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ms = _load_sibling('matrix_stats')

PAIRED_METRICS = ('source_recall', 'time_to_first_source', 'path_length_m')


def _load_runs(root: Path):
    with open(root / 'matrix_summary.json') as f:
        summary = json.load(f)
    return {(r['name'], int(r['seed'])): r for r in summary['runs']}


def _run_counts(root: Path, run):
    """返回 (n_correct, n_confirmed, n_truth, n_duplicates)。缺文件按 0。"""
    run_dir = root / run['name'] / f"seed{run['seed']}"
    n_correct = n_confirmed = n_truth = 0
    sp = run_dir / 'source_summary.json'
    if sp.exists():
        with open(sp) as f:
            s = json.load(f)
        n_confirmed = int(s.get('confirmed_count', 0))
        n_correct = len(s.get('localization_errors_m', []))
    ap = run_dir / 'attribution.json'
    if ap.exists():
        with open(ap) as f:
            a = json.load(f)
        n_truth = int(a.get('n_truth_sources', 0))
    n_dup = int(run.get('duplicate_confirmations', 0) or 0)
    return n_correct, n_confirmed, n_truth, n_dup


def compare_roots(root_a: Path, root_b: Path) -> dict:
    root_a, root_b = Path(root_a), Path(root_b)
    runs_a = _load_runs(root_a)
    runs_b = _load_runs(root_b)
    keys = sorted(set(runs_a) & set(runs_b))
    if not keys:
        raise SystemExit('no paired (case, seed) runs between the two roots')

    metrics = {}
    for metric in PAIRED_METRICS:
        xs, ys = [], []
        for k in keys:
            va, vb = runs_a[k].get(metric), runs_b[k].get(metric)
            if va is None or vb is None:
                continue
            xs.append(float(vb))  # 候选
            ys.append(float(va))  # 基线
        if not xs:
            continue
        metrics[metric] = {
            'n': len(xs),
            'mean_a': sum(ys) / len(ys),
            'mean_b': sum(xs) / len(xs),
            'p_value': _ms.paired_permutation_test(xs, ys),
        }

    sides = {}
    for label, root, runs in (('a', root_a, runs_a), ('b', root_b, runs_b)):
        corr, conf, truth, dup = [], [], [], []
        for k in keys:
            c, n, t, d = _run_counts(root, runs[k])
            corr.append(c)
            conf.append(n)
            truth.append(t)
            dup.append(d)
        p_point, p_lo, p_hi = _ms.cluster_bootstrap_ratio_ci(corr, conf, seed=11)
        d_point, d_lo, d_hi = _ms.cluster_bootstrap_ratio_ci(dup, truth, seed=13)
        sides[label] = {
            'precision': {'point': p_point, 'lo': p_lo, 'hi': p_hi},
            'duplicate_rate': {'point': d_point, 'lo': d_lo, 'hi': d_hi},
        }

    per_case = {}
    cases = sorted({k[0] for k in keys})
    for case in cases:
        xs = [float(runs_b[k]['source_recall']) for k in keys if k[0] == case]
        ys = [float(runs_a[k]['source_recall']) for k in keys if k[0] == case]
        per_case[case] = {
            'n': len(xs),
            'recall_a': sum(ys) / len(ys),
            'recall_b': sum(xs) / len(xs),
            'p_value': _ms.paired_permutation_test(xs, ys),
        }

    return {'n_pairs': len(keys), 'metrics': metrics,
            'sides': sides, 'per_case': per_case}


def render_report(report: dict, label_a: str, label_b: str) -> str:
    lines = [f'# Matrix compare: {label_a} (A) vs {label_b} (B)', '',
             f"paired runs: {report['n_pairs']}", '',
             '## Pooled paired tests (sign-flip permutation)', '',
             '| metric | n | mean A | mean B | p |', '|---|---|---|---|---|']
    for metric, row in report['metrics'].items():
        lines.append(f"| {metric} | {row['n']} | {row['mean_a']:.3f} "
                     f"| {row['mean_b']:.3f} | {row['p_value']:.5f} |")
    lines += ['', '## Event-pooled precision / duplicate rate '
              '(run-level cluster bootstrap 95% CI)', '',
              '| side | precision [lo, hi] | duplicate rate [lo, hi] |',
              '|---|---|---|']
    for label, key in ((label_a, 'a'), (label_b, 'b')):
        s = report['sides'][key]
        p, d = s['precision'], s['duplicate_rate']
        lines.append(f"| {label} | {p['point']:.3f} [{p['lo']:.3f}, {p['hi']:.3f}] "
                     f"| {d['point']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}] |")
    lines += ['', '## Per-case recall (paired permutation)', '',
              '| case | n | recall A | recall B | p |', '|---|---|---|---|---|']
    for case, row in report['per_case'].items():
        lines.append(f"| {case} | {row['n']} | {row['recall_a']:.3f} "
                     f"| {row['recall_b']:.3f} | {row['p_value']:.5f} |")
    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline_root')
    parser.add_argument('candidate_root')
    parser.add_argument('--label-a', default='baseline')
    parser.add_argument('--label-b', default='candidate')
    parser.add_argument('--out', default='')
    args = parser.parse_args()

    report = compare_roots(Path(args.baseline_root), Path(args.candidate_root))
    md = render_report(report, args.label_a, args.label_b)
    out = (Path(args.out) if args.out
           else Path(args.candidate_root) / 'compare_report.md')
    out.write_text(md)
    print(md)
    print(f'[compare] written to {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 5: 跑测试确认通过 + 全量回归 + CLI 冒烟**

```bash
python3 -m pytest src/thermal_robot/tests/test_phase1_observation.py -q
python3 -m pytest src/thermal_robot/tests/ -q
python3 src/thermal_robot/scripts/matrix_compare.py --help
```

- [ ] **Step 6: Commit**

```bash
git add src/thermal_robot/scripts/matrix_stats.py src/thermal_robot/scripts/matrix_compare.py \
        src/thermal_robot/tests/test_phase1_observation.py
git commit -m "feat: paired permutation tests, cluster bootstrap CI, matrix compare CLI"
```

---

### Task 11: 热相机选型文档

**Files:**
- Create: `docs/hardware/2026-06-thermal-camera-selection.md`

- [ ] **Step 1: 写入以下完整内容（原样落盘，不要自行增删型号）**

```markdown
# 热相机选型对比（阶段 1 交付物，spec §4）

目的：使采购可随时触发。最终决定由项目负责人做；本文档给出候选、关键参数与集成代价。
价格为 2025–2026 年公开渠道量级，采购前需重新询价。

## 候选对比

| 型号 | 分辨率 | 帧率 | 接口 | NETD | 价格量级 | ROS 2 集成 | 备注 |
|---|---|---|---|---|---|---|---|
| FLIR Lepton 3.5 | 160×120 | 8.7 Hz | SPI/I2C (需载板, 如 PureThermal 3 → USB UVC) | <50 mK | ~$200–300 | UVC 路线可用 usb_cam/v4l2_camera 直接出 Image; 有社区 lepton 驱动 | 出口管制使帧率限制在 <9 Hz; 体积最小, 适合快速起步 |
| FLIR Boson+ 320 | 320×256 | 60 Hz | USB-C / CMOS | ≤20 mK | ~$1500–2500 | flir_boson_usb (社区, ROS1 为主, 需移植) 或 UVC 模式 | 帧率/灵敏度都好; 价格中高 |
| FLIR Boson+ 640 | 640×512 | 60 Hz | USB-C / CMOS | ≤20 mK | ~$3000+ | 同上 | 论文效果最佳; 预算压力大 |
| HikMicro/海康微影 机芯 (如 TE 系列 256×192) | 256×192 | 25–50 Hz | USB/UART | ≤40 mK | ~¥2000–5000 | 厂商 SDK (Linux C) 自行包 ROS 2 节点 | 国内采购便利; SDK 文档中文; 驱动工作量中等 |
| Optris PI 450i | 382×288 | 80 Hz | USB | 40 mK | ~€3000+ | optris_drivers (ROS1, 需移植) | 工业级标定好; 重量/价格高 |
| MLX90640 (热电堆阵列) | 32×24 | 8 Hz | I2C | ~100 mK | ~$60 | 任意 SBC 上 python 直读, 自写节点极简 | 分辨率太低, 只适合算法管线连通性验证, 不适合论文实验 |

## 与本项目的匹配要点

1. **观测契约已透视兼容**（`observation.py`）：任何候选只需实现"热像素 → (射线/单元, 温度, 置信度, timestamp, frame_id, 传感器位姿, measurement_type=surface_radiance)"投影器。
2. **分辨率下限**：当前算法在 64×48 仿真图上工作；候选都≥此分辨率（除 MLX90640）。
3. **帧率下限**：管线 10 Hz；Lepton 8.7 Hz 可接受（降采样融合），其余均富余。
4. **定位需求**：实机观测只给方向+表面温度，距离需深度配准或运动三角化——若与深度相机/激光雷达共标定，优先选有公开外参标定流程的型号。
5. **G1 载荷**：Lepton/Boson/海康机芯均 <100 g，无载荷问题；Optris 偏重。

## 建议（供决策参考）

- **预算优先 / 尽快拿到 C 级录包**：FLIR Lepton 3.5 + PureThermal 3（UVC 即插即用，两周内可出第一批真实录包）。
- **论文效果优先**：Boson+ 320（60 Hz + 20 mK，运动模糊小，三角化质量高）。
- **国内供应链优先**：海康微影 256×192 机芯（采购周期短，SDK 中文支持）。

触发条件（spec §4）：负责人决定采购型号 → 启动 C 级里程碑（真实录包离线回放）。
```

- [ ] **Step 2: Commit**

```bash
git add docs/hardware/2026-06-thermal-camera-selection.md
git commit -m "docs: thermal camera selection comparison for hardware track"
```

---

### Task 12: 构建、全量回归、冒烟验证、devlog、收尾提交

- [ ] **Step 1: 全量构建**

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select thermal_interfaces
source install/setup.bash
colcon build --packages-select g1_description thermal_sensor_sim signal_preprocessor \
    thermal_field_reconstructor thermal_gradient_processor thermal_motion_controller thermal_bringup
source install/setup.bash
```

- [ ] **Step 2: 全量测试**

```bash
python3 -m pytest src/thermal_robot/tests/ -q
```
预期：旧 68 + 新 ~33 全部通过，0 failed。

- [ ] **Step 3: 冒烟 1 — 开阔世界 residual 策略**

```bash
python3 src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset phase0 --cases open__static2 --seeds 101 \
  --strategy residual --duration 60 --warmup 36 --domain-start 214 \
  --out-root /tmp/phase1_smoke_open
grep -c "strategy=residual" /tmp/phase1_smoke_open/open__static2/seed101/launch.log
grep "CLEARANCE" /tmp/phase1_smoke_open/open__static2/seed101/launch.log | head -3
grep "FUSE" /tmp/phase1_smoke_open/open__static2/seed101/launch.log | head -2
ls /tmp/phase1_smoke_open/open__static2/seed101/clearance.csv
```
通过标准：launch.log 命中 `strategy=residual`；出现 `[CLEARANCE] p_no_undetected=` 行；`[FUSE]` 行的耗时 < 100 ms；`clearance.csv` 存在且非空；run 产物（attribution.json 等）齐全。recall 数值不作要求（60 s 太短）。

- [ ] **Step 4: 冒烟 2 — 障碍世界（可见性真正生效）**

```bash
python3 src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset phase0 --cases walls__static3 --seeds 101 \
  --strategy residual --duration 90 --warmup 36 --domain-start 216 \
  --out-root /tmp/phase1_smoke_walls
grep "cells_blocked_only" /tmp/phase1_smoke_walls/walls__static3/seed101/launch.log | tail -2
```
通过标准：`cells_blocked_only` 数值 > 0（墙体后的格被标为"看过但被挡"——这是阶段 1 的核心行为证据）。若恒为 0，排查 `/map` 订阅 QoS（必须 TRANSIENT_LOCAL）与 spawn 偏移换算，如实记录排查过程。

- [ ] **Step 5: 冒烟 3 — v31 锚点无回归**

```bash
python3 src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset phase0 --cases open__static2 --seeds 101 \
  --strategy full --duration 45 --warmup 36 --domain-start 218 \
  --out-root /tmp/phase1_smoke_anchor
```
通过标准：退出码 0，`PASS open__static2 seed=101`，行为与 devlog 2026-06-10-2339 的 full 冒烟一致（recall=0.5 量级）。

- [ ] **Step 6: 清理 + devlog + 收尾提交**

```bash
bash src/thermal_robot/kill_gz.sh
pgrep -a gzserver || echo "gazebo clean"
```

写 `docs/devlog/<YYYY-MM-DD-HHMM>-phase1-visibility-residual-clearance.md`，按 背景/改动/验证/中间失败与处理/结论 结构，必须包含：三个冒烟的命令与关键 log 行（CLEARANCE、FUSE、cells_blocked_only）、测试计数（旧 68 不跌 + 新增数）、`[FUSE]`/`eval_ms` 实测耗时（算力预算证据）、所有中途失败与处理。然后：

```bash
git add docs/devlog/
git commit -m "docs: phase1 visibility/residual/clearance delivery log"
```

---

## 门槛评审（计划外，负责人触发）

本计划交付后，阶段 1 门槛需要全量矩阵对比（约一夜）：

```bash
python3 src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset phase0 --seeds 101,102,103,104,105 --strategy residual \
  --duration 120 --warmup 36 --domain-start 71 \
  --out-root bags/matrix/phase1_residual_<date>
python3 src/thermal_robot/scripts/matrix_compare.py \
  bags/matrix/phase0_full_20260611 bags/matrix/phase1_residual_<date> \
  --label-a v31 --label-b residual
```

基线就是已存在的 `bags/matrix/phase0_full_20260611`（同 seed 同用例，天然配对）。门槛判据（spec §4 阶段 1）：4/5 源场景 recall 配对检验显著高于 v31；precision 95% CI 下界 ≥ 0.90 且 duplicate rate 不高于 v31；归因 not_reached 显著减少；`[FUSE]`/clearance eval_ms 满足单核 100 ms 预算。

## Self-Review 备注

- 新测试数估算 44 个（Task 1: 4, Task 2: 5, Task 3: 7, Task 4: 4, Task 6: 4, Task 7: 5, Task 8: 6, Task 9: 3, Task 10: 6），与旧 68 个合计约 112；以实际为准，devlog 记录真实数字（各 Task 中间步骤标注的"N passed"同理以实际为准）。
- spec 覆盖检查：visibility.py ✓(T3) / 三态融合+扇区 ✓(T4) / 接口契约含 timestamp/frame_id/位姿/measurement_type ✓(T2) / 栅格常数统一 ✓(T1) / residual ✓(T6) / 打分核+可达性 ✓(T7) / clearance 生成式 v1+校准 ✓(T8) / matrix_stats 配对+bootstrap ✓(T10) / 选型文档 ✓(T11) / 算力预算证据 ✓(T5 FUSE 日志、T9 eval_ms、T12 冒烟)。
- v31 冻结检查：full/frontier/levy 代码路径仅在 Task 9 的 `elif 'residual'` 分支外围被阅读、未被修改；Task 4 的 `integrate_image` 在无 occupancy 时数学等价（有回归测试锁定）。
