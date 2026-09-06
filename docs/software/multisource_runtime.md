# 多热源软件运行与 UGV 接入

当前执行范围以 [2026-09-06 静态持续热源巡检规划](../superpowers/specs/2026-09-06-static-thermal-inspection-design.md) 为准。精简代码：删除移动、生灭事件和自动判断搜完的专用实现；有助于基础能力的观测、去重、地图融合、规划和慢层保留。历史动态研究与验证不代表当前静态任务已达标。

## 策略与观测

主入口仍是 `sim_nav_slam_launch.py`，新增选项：

仿真启动默认 `use_sim_time:=true`，热链路、SLAM 和 Nav2 使用仿真时钟。真实 UGV 覆盖层默认 `false`；录包回放显式设置 `true`。原矩阵脚本对历史策略显式保留旧时钟配置，新策略使用仿真时钟；历史实验结果不混入本次软件验收。

| 参数 | 值 | 行为 |
|---|---|---|
| `strategy` | `fast` | 二维 Kalman 快层、残差探索 |
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

B 级从当前 world 的碰撞几何构造遮挡场景，同时在 Gazebo 临时 world 中加入热源对应的盒体或柱体。前视热图通过最近表面射线交点生成；深度为 optical-z 米。目标固定位置、持续发热；动态场景生成和实体移动同步已删除。指定发热面由 `surface_hot_faces` 控制，`[-1]` 为全部表面。盒体面编号依次是 -x/+x/-y/+y/-z/+z，柱体为侧面/底面/顶面。

温度扰动参数包括噪声、偏置、漂移、emissivity 和深度噪声/失效。辐射采用灰体 T^4 混合近似；不宣称是完整 LWIR 光谱、热传导或反射模拟。算法只接收热图、深度、CameraInfo、里程计和 SLAM 地图，`/sim/thermal_sources_truth` 仅供评测。

## 快慢层

- 快层默认 2 Hz：二维位置 Gaussian filter、一对一关联、多帧确认。位置在缺测时保持不变，协方差允许地图/配准漂移。关联保留协方差归一化项和实测位置门控，防止不确定的旧标签抢占新观测。
- 当前冷观测可以降低错误热点假设的置信度；中心和邻域必须获得新观测且没有热支持。未观测、过期和遮挡不等于冷证据，重复缓存帧不重复计数。被反证的假设需要多帧重新确认。
- B 级控制在启动、移动 4 米或上次扫描完成 60 秒后进行一周相机转向扫描，完成条件使用里程计累计转角。探索目标到达半径为 0.6 米，正常推进时保持目标；持续 12 秒无位移或 45 秒超时会重选，并临时排除失败目标附近 2 米。地图尚未准备好时等待/扫描，不发起长距离随机退路。参数均在 `multisource.yaml`，需要按底盘和场景调整。
- 慢层保留带标签的位置高斯、强度/尺度与存在置信度，源数分布由 Bernoulli 卷积得到。残差支持新候选，当前可见却未检测到的证据用于清理错误候选；遮挡只让置信度老化。关联仍是有门控的近似匹配，非精确联合后验。
- 未充分支持的重叠候选需持续证据才合并；不同已确认标签不因一次接近而合并。删除速度条件、运动外推和生灭事件记录。
- 慢层独立进程，发布 `/thermal/belief` 的 health、revision、compute_ms、cardinality_pmf。失败/超预算不输出可用新估计；控制器按健康状态及时间戳拒绝过期后验。
- online 模式将期望存在性熵降与定位信息收益用于选点，并通过保守协方差交集给快层先验校正。shadow 不影响规划或快层；off 时快层独立运行。
- `fast` 和 `gp_ucb` 不接受慢层先验，即使慢层仍在 online 估计。`belief_mode` 是启动选项，切换模式时重新启动覆盖层。`off` 在 launch 中显式按字符串传递。
- 新控制路径按世界坐标接近源并保持停靠距离；直达被挡时尝试已知空闲停靠点与 Nav2 绕行，无可用停靠点或导航停滞时暂缓该源并继续探索，冷却后可重试。Nav2 探索停滞有原有受扫描保护的直接控制回退。
- 预测重访和搜完概率模块已删除，也不按预设源数或“很久没找到新源”自动结束。雷达避障间距、相机扫描、失败点冷却仍保留。

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

**当前采用深度定位路线。Lepton/PT3 自身不输出深度；需要额外提供已标定的深度输入。** 本轮未实现无深度的运动三角化路线。硬件模式缺少 CameraInfo、TF，或热图与深度坐标系不一致时跳过该帧。

录包应包括上述图像、CameraInfo、TF、里程计、雷达与地图。回放用 `ros2 bag play BAG --clock`，覆盖层加 `use_sim_time:=true enable_motion:=false`。如果录包已包含 `/thermal/raw`，再加 `start_input:=false`。真实测温标定、深度误差、车体速度/足迹配置和 Pi 5 负载由用户实机验证。

## 软件验收

```bash
python3 -m pytest src/thermal_robot/tests/ -q
python3 src/thermal_robot/scripts/run_software_validation.py \
  --out bags/software_validation/my_b_dual --sensor-model b --strategy dual \
  --duration 90 --domain 151
```

每次使用新输出目录，保留启动日志、probe JSON、轨迹摘要、源估计和图像快照。软件 probe 检查数据链、有效深度、地图观测、运动输出和慢层状态，不以它替代全矩阵 recall/precision 验收。实际结果与剩余限制在最终交付审计中记录。

验证器在启动前比较源码与 install，发现未重新构建的文件直接退出；执行期间检查节点崩溃、话题停止及实际位移。Gazebo Classic 用同一个 master 端口，本工具通过锁防止自身并发运行；请逐个运行验证用例。`--seed`、`--scenario`、`--world`、`--duration` 可选择缩减矩阵。慢层预算注入可通过覆盖 YAML 和 `--expected-health over_budget` 运行。覆盖 YAML 的副本保留在输出目录。

每次运行另存 `detection_metrics.json`：在采样窗口内，将 age ≤ 1.5 秒的 confirmed 轨迹与 1 米内物理真值配对，至少 3 次有效命中才计为发现一个活跃源。串 ID 统计包含熄灭后的同一物体；活跃源发现数不包含冷物体。该诊断仍依赖距离门限和采样窗口，不是全任务 recall、precision 或全量验收。可对旧结果只读执行 `python3 src/thermal_robot/scripts/evaluate_detection_tracks.py RUN_DIRECTORY`。

```bash
# 不需要硬件，验证 UGV 覆盖层的真实 ROS 消息与 TF 接线
python3 src/thermal_robot/scripts/validate_hardware_contract.py \
  --out bags/software_validation/my_hardware_contract

# 从保留的数据生成独立图像
python3 src/thermal_robot/scripts/plot_software_validation.py \
  bags/software_validation/my_b_dual --out /tmp/b_dual.png
```

`SourceEstimate` 保留位置、协方差、强度、确认状态和 `age_s`，删除速度、速度方差、重捕获计数与间隔。重新构建接口包和所有依赖；外部消费者需要更新，旧派生消息 bag 不保证兼容。优先回放原始热图、深度和 TF。

`ThermalMap` 进一步删除无人消费的 `view_sectors`、`last_view_distance_m`；保留 `view_state` 和观测年龄，继续区分未观察、被遮挡、已观察。对应的方向历史缓存和 `sector_memory_s` 配置已删除；外部消息消费者同样需要重新构建。mapper 的 `occupied_threshold` 现在实际传入可见性判断，默认仍为 65。

无效控制参数 `plateau_thresh`、`peak_confirm_s`、`peak_confirm_lin_vel`、`converge_circle_radius`、`levy_post_confirm_step` 已删除。B 级慢层直接使用表面候选，不再计算随后被丢弃的高斯预测场；A 级仍保留残差候选计算。信息收益仅保留控制器调用的共享实现。详见 [冗余清理记录](../devlog/2026-09-06-redundancy-cleanup.md)。

静态对照可固定场景、seed 和预算，比较 fast/dual 或 online/shadow/off；也可通过 `software_params` 改 `residual_enabled` 或 mapper 的 `visibility_enabled`。B 级物理遮挡不因关闭算法可见性而消失。`posterior_*` 假设应与 `belief_node` 参数保持一致。

当前精简与实测证据见 [精简验证记录](../devlog/2026-09-06-static-inspection-simplification.md)。[旧软件交付审计](software_completion_audit.md) 仅对应精简前版本。
