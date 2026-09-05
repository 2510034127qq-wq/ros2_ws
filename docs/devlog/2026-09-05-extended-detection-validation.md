# 扩展探测验证

用户要求“多跑些测试看看”。固定 `7bfaf5f`，新增 8 次串行 ROS/Gazebo 闭环，累计 1050 秒采样，所有运行使用同一源码指纹。覆盖 A 级静态长跑、两个随机种子的 B 级直线运动、圆周、航点/随机、五源、障碍生灭及障碍五源。未修改运行算法或配置。

8/8 通过运行检查；6/8 在采样窗口发现全部活跃源；4/8 出现单物理源多个已确认 ID，其中 3 个用例存在同一消息中的重复确认。三种位置门限均未观察到跨源串 ID。A 级 180 秒仍只有 1/2；障碍生灭的活跃源窗口发现数为 2/3。

原始日志与分析脚本保留于 `bags/software_validation/20260905_extended_detection/`。输入、哈希、指标及失败边界见 [扩展报告](../software/extended_detection_validation.md) 和 [结果快照](../software/extended_detection_validation_snapshot.json)。本次不将运行检查通过等同于探测质量全部达标，未实施报告建议的后续修复。
