# 热相机与 UGV 接入要求

核对日期：2026-09-14。保留原文件名便于旧链接访问；本页按当前代码说明接入要求，不再把旧报价、驱动推测或采购工期当作现状。

## 当前软件支持什么

项目已有 `ugv_thermal_launch.py`、`radiometric_input_node.py`、`depth_registration_node.py` 和 B 级透视 mapper。UGV 覆盖层默认只运行感知与估计（`enable_motion=false`），不启动底盘、雷达、SLAM 或 Nav2。

`ugv_thermal.yaml` 的输入缺省是 ROS 模式、160×120、8.6 Hz、TLinear、Kelvin 比例 0.01。它们是**项目配置值**，不是对任意相机型号、固件或输出模式的保证。Lepton/PureThermal 的接入路径已经预留，但未做本项目真实设备验收。

| 输入 | 当前契约 |
|---|---|
| `/thermal/radiometric_input` | TLinear 模式接受 16UC1/mono16；Celsius 模式接受 32FC1；必须配置正确单位 |
| `/thermal/camera_info` | 实测热相机内参；图像需符合代码的去畸变投影假设 |
| `/depth/image`、`/depth/camera_info` | 深度为 32FC1 米或 16UC1 毫米，配准至热相机 |
| `/thermal/depth` | 已配准时可直接提供，并关闭 register_depth |
| 相机 optical TF | 实测外参；x 右、y 下、z 前 |
| `/odom`、`/scan`、`/map` 与导航 TF/action | 外部底盘/导航栈负责 |

可选 `input_mode=uvc` 通过 OpenCV V4L2 Y16 采集；相机必须已配置有效测温输出。普通 DN 或 RGB/伪彩图不能因为通过 USB/UVC 读取就视为摄氏温度。当前深度定位要求额外深度来源，Lepton/PureThermal 本身不提供深度，代码没有无深度运动三角化后备路线。

## 选型时与代码逐项对齐

确认设备提供可解析的辐射测温数值、单位/缩放说明、时间戳和内参标定能力；再确认 Linux/目标计算平台上能稳定采集，以及深度源、热相机的视场/同步/外参可以满足配准需求。分辨率和标称帧率本身不能保证小目标发现率。

本仓库没有验证任何候选型号的最新价格、库存、重量、ROS 驱动维护状态或采购周期，不据此承诺某个型号“两周可用”或“论文效果最佳”。具体型号比较应另查厂商当前资料并实测。本页不把仿真 B 级默认参数写成物理相机性能。

## 已有验证和缺口

`validate_hardware_contract.py` 用合成 ROS 图像、CameraInfo 与 TF 检查温度/深度单位、配准、热地图、源确认及默认不运动；它不连接真实相机。当前缺少真实测温误差、同步/配准误差、深度孔洞、小目标与低温差场景、Pi 5 完整负载及实际车体运动验证。

构建、话题、回放与参数用法见 [运行/UGV 文档](../software/multisource_runtime.md) 和 [操作手册](../handover/05-构建运行调试手册.md)。仿真用简化 G1 差速模型，保留完整 G1 资产不等于已有 G1 SDK 实机集成。
