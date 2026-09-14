# thermal_robot 项目现状与交接报告

核对日期：2026-09-14。运行代码基线：`244d067`。本报告说明现有实现；历史开发日志和改版计划保留原文，不作为当前功能清单。

## 当前结论

项目是 ROS 2 Humble + Gazebo Classic 的**静态、持续发热目标巡检**工作区。机器人探索环境、把热观测投影到地图、登记与去重热源，并尝试绕障接近。默认启动 `sensor_model:=a strategy:=dual belief_mode:=online`；B 级提供前视三维热表面与深度观测。运动源、生灭调度、预测重访和自动清场判断已从当前实现删除。

仿真使用 `g1_nav.urdf` 差速模型，完整 G1 URDF/mesh 只是保留资产。已有 `ugv_thermal_launch.py` 热处理覆盖层，依赖外部底盘、导航和相机标定；未实现 G1 SDK，也未证明真实 UGV 可自主完成任务。

## 架构与实现

| 层 | 当前责任 | 关键入口 |
|---|---|---|
| 几何导航 | Gazebo 的 odom/scan、SLAM 地图与 TF、Nav2 路径执行 | `sim_nav_slam_launch.py`、`nav2_params.yaml` |
| 热观测 | A 理想俯视场；B 前视热表面/深度；实机测温输入和深度配准 | `sensor_node.py`、`radiometric_input_node.py`、`depth_registration_node.py` |
| 地图 | 图像预处理、场/梯度输出、带可见性及年龄的世界热图 | `thermal_mapper_node.py`、`thermal_mapping.py` |
| 身份 | 快层是唯一源身份入口，二维位置估计、多帧确认、重复候选抑制 | `source_tracker_node.py`、`source_tracking.py` |
| 慢层 | 按登记 ID 更新后验；online 可反馈，shadow 仅估计，off 停止估计 | `belief_node.py`、`belief.py` |
| 决策 | 默认世界坐标接近、B 级扫视、残差探索、受阻绕行/延期 | `controller_node.py`、`runtime_policy.py`、`planning.py` |

`/thermal/field -> /thermal/gradient` 与 `/thermal/filtered -> /thermal/map -> /thermal/sources` 是并行支路，不能把整个系统画成只有梯度上升的一条链。几何地图 `/map` 不等于热覆盖；所有源位置估计来自观测，真值仅用于模拟和离线评测。

已确认静态源在会话内保留身份和登记位置；`age_s` 表示最后观测年龄，不代表还在画面中。候选默认 12 秒未观察即清理。慢层不独立新建身份。误确认不会自动撤销，源登记和已处理集合均不跨进程重启持久化。源数后验只描述已登记假设，不是环境总源数或搜完概率。

## 默认值与控制边界

- A 级 64×48、10 Hz；B 级 160×120、8.6 Hz。B 默认 passthrough 且关闭空间平滑，避免移动相机像素滤波拖影与深度错配。
- B 默认热物体宽 0.3 m、高 0.8 m，绝对表面温度 `[28.0, 37.0]` ℃按源 ID 和种子一次抽取后恒定；环境为 22℃。B 范围覆盖场景 amplitude，A 仍用场景温升。图像读数还受发射率与噪声影响。
- 仿真 odom 采用 Gazebo 绝对世界位姿，出生点 `(-6, 0)`；launch 将 mapper/controller 偏移清零。这验证坐标一致性，不验证轮速漂移。
- 仿真 launch 默认 ROS 仿真时钟；控制器部分策略超时仍用 `time.monotonic()`，不能把所有超时视作仿真秒。矩阵脚本另对旧策略使用墙钟配置。
- 当前接近保持目标，仅当另一个目标明显更近时切换；受阻立即停车，持续 0.5 秒后选择已知安全停靠/中间点或延期。中间推进默认上限 2 m。
- 默认停距 1.2 m，控制确认还检查距离容差、朝向、停留与登记状态。Nav2 action 结束不等于物理接近成功。全部已知源处理后仍探索，由操作者或外部预算结束。

参数实际优先级是节点缺省、`params.yaml`、`multisource.yaml`（或替换文件）、launch 内联覆盖；UGV 再叠加 `ugv_thermal.yaml`（或 `hardware_params`）。单独运行节点的默认值不等于完整 launch。

## 验证与尚未完成的能力

本次文档核对运行 `python3 -m pytest src/thermal_robot/tests/ -q`：**211 passed**。本次未重新构建或启动 ROS/Gazebo，也未重跑性能实验。以下闭环数值引自仓库保留的 [最近接近策略记录](docs/devlog/2026-09-07-static-approach-strategy.md)，并非本次实测：

| 条件 | 保留的证据 | 能支持的结论 |
|---|---|---|
| B 五源障碍场，3 组 180 秒采样 | 均发现 5/5，物理接近 4/5 | 所测条件下接近推进改善，仍未在短预算内全部完成 |
| B 同类场景，单次 300 秒采样 | 发现/物理接近 5/5 | 单次长预算可完成，不能推广到全部种子/世界 |
| A 双源，60 秒采样，两次同配置 | 发现 0/2、1/2 | 稳定搜齐能力尚未通过；接近只有控制日志，未独立采集机器人真值 |

当前没有全场景多种子性能保证、论文统计验收、真实测温/深度标定、Pi 5 负载或实机运动验收。旧动态实验、旧清场校准工具和历史 phase 1 门槛不属于当前功能或当前达标证据。

## 阅读入口与记录范围

- [项目 README](src/thermal_robot/README.md)：构建、仿真、参数与验证命令。
- [交接总览](docs/handover/00-总览与导读.md)：架构、文件、控制器、评测和排错分册。
- [多源运行与 UGV 接入](docs/software/multisource_runtime.md)：接口和模式约束。
- [热相机接入要求](docs/hardware/2026-06-thermal-camera-selection.md)：当前代码要求与未验证部分。
- `docs/devlog/`、`docs/superpowers/plans/`、`docs/superpowers/specs/` 保留历史过程和改版设计。
- `docs/software/detection_recovery.md`、`extended_detection_validation.md`、`software_completion_audit.md` 及其 JSON/图像是旧版本实验记录，原文保留。

生成目录 `build/ install/ log/ bags/` 不提交。详细运行原始数据通常在被忽略的 `bags/` 中；仅有文档摘要不等于拥有可完整复查的实验归档。
