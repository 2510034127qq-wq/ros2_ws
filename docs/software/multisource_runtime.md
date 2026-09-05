# 多热源软件运行与 UGV 接入

本轮交付范围以 `docs/superpowers/plans/2026-09-05-software-completion.md` 为准。原总体设计保留为研究背景。软件功能验证与论文统计门槛分开：本轮不要求全矩阵显著优势、两周影子期或真实硬件成绩。

## 策略与观测

主入口仍是 `sim_nav_slam_launch.py`，新增选项：

| 参数 | 值 | 行为 |
|---|---|---|
| `strategy` | `fast` | Kalman 快层、残差探索、预测重访 |
| | `dual`（默认） | 快层加可用的慢层后验；慢层不可用时走快层 |
| | `gp_ucb` | 有界 GP-UCB 对照规划 |
| | `full/frontier/levy/residual` | 保留原策略入口 |
| `sensor_model` | `a`（默认） | 原理想俯视场观测 |
| | `b` | 三维表面前视热图和深度，160×120、8.6 Hz |
| `belief_mode` | `online/shadow/off` | 慢层参与决策 / 仅输出 / 停止估计 |
| `software_params` | YAML 路径 | 覆盖新模块的参数，默认 `config/multisource.yaml` |

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch thermal_bringup sim_nav_slam_launch.py \
  use_rviz:=false use_gzclient:=false strategy:=dual sensor_model:=b
```

B 级从当前 world 的碰撞几何构造遮挡场景，同时在 Gazebo 临时 world 中加入热源对应的盒体或柱体。前视热图通过最近表面射线交点生成；深度为 optical-z 米。源生灭改变表面发热，物体本身仍存在；动态源同时更新热表面位置及 Gazebo 物体位置。指定发热面由 `surface_hot_faces` 控制，`[-1]` 为全部表面。盒体面编号依次是 -x/+x/-y/+y/-z/+z，柱体为侧面/底面/顶面。

温度扰动参数包括噪声、偏置、漂移、emissivity 和深度噪声/失效。辐射采用灰体 T^4 混合近似；不宣称是完整 LWIR 光谱、热传导或反射模拟。算法只接收热图、深度、CameraInfo、里程计和 SLAM 地图，`/sim/thermal_sources_truth` 仅供评测。

## 快慢层

- 快层：有界常速 Gaussian filter、一对一预测关联、stale 预测和重捕获统计；默认 2 Hz。位置发布的是当前预测位置，`age_s` 单独表示离上次检测多久。
- 慢层：每个带标签源簇承载运动条件高斯、强度/尺度估计与 Bernoulli 存在概率。源数分布由 Bernoulli 卷积得到。残差支持 birth/split birth，当前可见区域的漏检支持 death；遮挡仅降低长期生存先验。关联采用有门控的近似匹配，非精确联合多目标后验。
- 已确认且具有不同身份的源不因一次近距离交叉而合并；未充分支持的重叠簇需持续 merge 证据。
- 慢层独立进程，发布 `/thermal/belief` 的 health、revision、compute_ms、cardinality_pmf。失败/超预算不输出可用新估计；控制器按健康状态及时间戳拒绝过期后验。
- online 模式将期望存在性熵降与定位信息收益用于选点，并通过保守协方差交集给快层先验校正。shadow 不影响规划或快层；off 时快层独立运行。
- 清场概率使用空间 Poisson、幅值先验、扇区漏检及历史衰减，双层另计尚未确认源簇。`clearance_calibrated: false`，默认仅输出，不自动停机。只有用户有标定证据并明确启用时才可用清场触发 DONE。

## UGV 与 Lepton/PT3

底盘驱动、Nav2、雷达、轮式里程计及其 TF 由 UGV 原有软件负责。本仓库提供热处理与自主任务覆盖层，不启动 Gazebo、不伪造真实机器人里程计。

必需输入：

- `/odom`、`/scan`、`/map`、`map -> odom -> base_link`，以及 Nav2 的 `navigate_to_pose` action。
- 热测温输入：`/thermal/radiometric_input`。配置为 TLinear 时要求 `16UC1/mono16` 和正确 Kelvin 比例；配置为 Celsius 时要求 `32FC1`。
- `/thermal/camera_info` 与 `/depth/camera_info`，必须是实际标定的内参和去畸变图像。
- `/depth/image`：`32FC1` 米或 `16UC1` 毫米。
- 相机 optical 坐标系的已标定 TF。光学轴约定 x 向右、y 向下、z 向前。

```bash
ros2 launch thermal_bringup ugv_thermal_launch.py
```

默认 `enable_motion:=false`，只启动传感与估计供录包/回放验证。完成底盘与标定检查后通过 `enable_motion:=true` 启动控制器。该参数是实机操作选项，不代表本次执行过实机验证。

PT3 可选 `input_mode: uvc`，使用 OpenCV V4L2 Y16 采集。相机必须事先配置正确的 radiometric/TLinear 输出；驱动拒绝 RGB/伪彩图，未将普通原始 DN 当作温度。实际固件/USB/ARM 组合需用户在设备上验证，源码中未假定 UVC 即插即用就等于测温已正确。ROS 输入模式也可接用户已有的测温驱动。

已对齐的深度可直接发布 `/thermal/depth` 并设置 `register_depth:=false`。未对齐深度由 `depth_registration_node` 使用标定 TF 重投影，深度孔洞保持无效，不填造距离。

录包应包括上述图像、CameraInfo、TF、里程计、雷达与地图。回放用 `ros2 bag play BAG --clock`，覆盖层加 `use_sim_time:=true enable_motion:=false`。如果录包已包含 `/thermal/raw`，再加 `start_input:=false`。真实测温标定、深度误差、车体速度/足迹配置和 Pi 5 负载由用户实机验证。

## 软件验收

```bash
python3 -m pytest src/thermal_robot/tests/ -q
python3 src/thermal_robot/scripts/run_software_validation.py \
  --out bags/software_validation/my_b_dual --sensor-model b --strategy dual \
  --duration 90 --domain 151
```

每次使用新输出目录，保留启动日志、probe JSON、轨迹摘要、源估计和图像快照。软件 probe 检查数据链、有效深度、地图观测、运动输出和慢层状态，不以它替代全矩阵 recall/precision 验收。实际结果与剩余限制在最终交付审计中记录。

消融可复制 `multisource.yaml` 后改 `revisit_enabled`、`residual_enabled`、mapper 的 `visibility_enabled`，再用 `software_params` 传入；比较 fast/dual 或 shadow/off 可隔离慢层作用。B 级图像的物理遮挡不应因“去算法可见性”而消失。
