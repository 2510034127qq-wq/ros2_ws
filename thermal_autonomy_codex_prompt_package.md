# thermal_robot Autonav Codex Prompt Package

这个文件包含给 Codex CLI 使用的完整任务 prompt、推荐运行命令、恢复命令，以及一个可选的 `AGENTS.md` 辅助说明模板。

使用方式：

1. 保存本文件到任意位置，例如 `/tmp/thermal_autonomy_codex_prompt_package.md`。
2. 进入 ROS2 工作区。
3. 把下面“主 Prompt”部分单独喂给 Codex，或者直接用后面的“一键写入并运行命令”。

---

## 一键写入并运行命令

```bash
cd /home/hanchen/ros2_ws

cat > /tmp/thermal_autonomy_codex_prompt.md <<'EOF'
你是 Codex CLI，当前工作目录是 /home/hanchen/ros2_ws。你要在现有 thermal_robot ROS 2 Humble 项目基础上，设计并实现一套“可长期自动运行、可中断恢复、可自动迭代算法、可自动生成新仿真世界、可研究论文并记录依据”的完整实验与优化流程。目标是在不进行真实机器人实机运行的前提下，尽可能完成动态/静态多热源热导航算法的仿真侧全部研发工作，并让系统能长时间无人值守运行。

一、当前项目事实与硬约束

1. 当前主线架构：
   Gazebo g1_nav 简化差速模型
     -> /odom + /scan
     -> slam_toolbox
     -> /map + map->odom TF
     -> Nav2 NavigateToPose

   /sim/thermal_raw
     -> /thermal/filtered
     -> /thermal/field + /thermal/get_field_info
     -> /thermal/gradient
     -> controller_node
          FINE: 直接 /cmd_vel
          COARSE: Nav2 优先，直接 /cmd_vel 兜底

2. 当前唯一主线 launch：
   src/thermal_robot/thermal_bringup/launch/sim_nav_slam_launch.py

3. 当前核心包：
   thermal_interfaces
   g1_description
   thermal_sensor_sim
   signal_preprocessor
   thermal_field_reconstructor
   thermal_gradient_processor
   thermal_motion_controller
   thermal_bringup

4. 当前 Config-B 场景：
   SA_left  world=(-1.0,  3.5), amplitude=35.0, sigma=1.1, peak≈57C
   SB_far   world=( 6.0, -3.0), amplitude=22.0, sigma=0.9, peak≈44C
   SC_weak  world=(-5.0, -5.5), amplitude=16.0, sigma=0.8, peak≈38C
   robot spawn = (-6.0, 0.0)
   thermal camera = 64x48, FOV 4.0m x 3.0m, ambient 22C, noise_std 0.5C, 10Hz

5. 当前控制器是 controller_node v31。核心原则：
   - 局部热信号强时用梯度/局部行为；
   - 大范围探索时 Nav2 优先；
   - Nav2 不可用或失败时必须有 /cmd_vel 兜底；
   - 控制器禁止使用热源真实坐标作为导航先验；
   - 热源真实坐标只能用于仿真生成、ground truth、评估、图表和报告。

6. 当前已知问题要优先处理：
   - Nav2 costmap/map 边界与规划失败问题，例如 robot out of bounds、start position off global costmap；
   - nav2_params.yaml 中 BT XML 绝对路径问题；
   - use_sim_time 当前默认 false，不要单独改某一个节点。若切换 use_sim_time，必须整体验证 /clock、TF、SLAM、Nav2 lifecycle、Gazebo 插件时间戳；
   - reconstructor_node shutdown 风格不一致，可能污染退出日志；
   - 当前系统已知能找到 SA_left，但后续 SB_far 与 SC_weak 的发现率仍需重点优化。

7. 不允许：
   - 不允许执行真实机器人或真实硬件控制；
   - 不允许把 ground truth 热源坐标注入 controller、planner、gradient processor 或任何在线导航决策模块；
   - 不允许通过降低 num_sources、缩短场景、调低成功标准、直接读取场景 YAML 热源位置等方式作弊；
   - 不允许删除完整 G1 资产，除非只做文档建议；
   - 不允许把大量 bags、实验输出直接纳入 git 版本控制；
   - 不允许无检查地无限循环。长时间运行也必须有 watchdog、checkpoint、日志和可恢复状态。

二、最终目标

实现一个长期自主研发系统，使它可以：

1. 自动构建、测试、启动 headless 仿真；
2. 自动采集 ROS 话题、实验日志、控制器状态、地图/路径/热场数据；
3. 自动计算指标、生成图表和报告；
4. 自动评估当前算法；
5. 自动提出候选改动，包括参数调优和算法结构改进；
6. 自动运行多轮实验，对比候选算法；
7. 在当前 Config-B 世界达到高标准后，自动生成新的不同仿真世界；
8. 新世界要覆盖：
   - 热源静态/动态；
   - 热源数量变化；
   - 热源位置变化；
   - 热源强弱变化；
   - sigma 变化；
   - 噪声变化；
   - 机器人初始位置和朝向变化；
   - 可选障碍物/地图布局变化；
9. 自动进行 curriculum：
   当前世界 -> 当前世界多 seed 稳定通过 -> 随机场景训练集 -> 随机场景验证集 -> 随机场景 holdout 测试集；
10. 自动搜索论文和资料，整理文献库，并把有价值算法转化为可测试候选；
11. 任何时候我停止进程后，都可以在停止前基础上恢复；
12. 最终生成“实机前准备包”，但不做实机运行。

三、必须优先实现的工程化目录

在不破坏现有 ROS 包结构的前提下，新增以下目录和文件。具体文件名可微调，但必须有同等功能：

experiments/autonav/
  README.md
  config/
    experiment_defaults.yaml
    metrics_thresholds.yaml
    optimizer_space.yaml
    scenario_schema.yaml
  scenarios/
    config_b_baseline.yaml
    generated/
  runs/
    <run_id>/
      run_config.yaml
      scenario.yaml
      candidate.yaml
      command_log.jsonl
      ros_launch_stdout.log
      ros_launch_stderr.log
      collector_stdout.log
      collector_stderr.log
      metrics.json
      metrics_timeseries.csv
      events.jsonl
      source_detections.csv
      ground_truth.csv
      nav2_health.json
      tf_health.json
      figures/
      summary.md
      git_diff.patch
  state/
    autonav.sqlite
    latest_state.json
    locks/
  literature/
    literature.bib
    paper_index.json
    notes/
    research_log.md
  candidates/
    baseline_v31.yaml
    generated/
  reports/
    leaderboard.csv
    current_best.md
    final_sim2real_readiness.md

四、必须实现的自动化组件

1. Scenario system

把当前 sensor_node.py 中硬编码 WORLD_SOURCES 的方式重构为“可选场景文件驱动”：

- 新增参数 scenario_file；
- scenario_file 为空时保持当前 Config-B 行为，确保兼容；
- scenario_file 有值时从 YAML 加载：
  world bounds
  robot spawn pose
  ambient temp
  noise_std
  source list
  source id
  initial position
  amplitude
  sigma_m
  motion type:
    static
    linear
    circular
    random_waypoint
    sinusoidal
  motion parameters:
    velocity
    radius
    period
    phase
    bounds
    seed
  optional obstacles / visual markers

动态源必须通过 source.position(t) 计算。评估时 ground truth 也要用同一套模型输出每个时间点的热源真实位置。

Gazebo world 与 sensor_node 场景必须一致。至少要做到：
- 当前 Config-B 可由 YAML 完整复现；
- 新场景可自动生成；
- world 文件中可以生成静态视觉标记/障碍物；
- 动态热源即使 Gazebo 里没有动态可视模型，sensor、metrics、plots 必须正确。

2. Launch integration

扩展 sim_nav_slam_launch.py：
- 新增 launch 参数 scenario_file；
- 新增 launch 参数 experiment_run_id；
- 新增 launch 参数 headless 或继续使用 use_rviz:=false use_gzclient:=false；
- 把 scenario_file 传入 sensor_node；
- 不要破坏原 launch 命令；
- 修复 BT XML 绝对路径问题，用 launch 动态路径注入；
- 统一日志输出到 experiment run 目录；
- 继续支持当前 use_sim_time=false 默认行为。

3. Experiment runner

实现一个可恢复的实验管理器，例如：

python3 experiments/autonav/run_manager.py run \
  --resume \
  --phase current_world_mastery \
  --headless \
  --max-hours 12

run_manager 必须做到：

- 自动 preflight：
  cd /home/hanchen/ros2_ws
  source /opt/ros/humble/setup.bash
  source install/setup.bash 或在缺失时构建
  rosdep check --from-paths src/thermal_robot --ignore-src
  colcon list
  python3 -m pytest src/thermal_robot/tests/test_thermal_system.py -q
  ros2 launch thermal_bringup sim_nav_slam_launch.py --show-args

- 自动构建：
  colcon build --packages-select thermal_interfaces
  source install/setup.bash
  colcon build --packages-select g1_description thermal_sensor_sim signal_preprocessor thermal_field_reconstructor thermal_gradient_processor thermal_motion_controller thermal_bringup
  source install/setup.bash

- 自动清理残留 Gazebo：
  bash src/thermal_robot/kill_gz.sh

- 自动启动 headless 仿真：
  ros2 launch thermal_bringup sim_nav_slam_launch.py use_rviz:=false use_gzclient:=false scenario_file:=...

- 自动 readiness check：
  ros2 node list
  ros2 topic list
  ros2 action info /navigate_to_pose
  ros2 lifecycle get /planner_server
  ros2 lifecycle get /controller_server
  ros2 lifecycle get /bt_navigator
  ros2 topic hz /sim/thermal_raw
  ros2 topic hz /thermal/gradient
  ros2 topic hz /odom
  ros2 topic hz /scan
  ros2 service call /thermal/get_field_info ...

- 自动采集：
  优先扩展现有 collect_sim_data.py；
  必须采集 odom、cmd_vel、thermal raw/filtered/field/gradient、map、plan、controller 状态、Nav2 action/状态、TF 健康、ground truth source positions；
  可以使用 CSV/JSONL/rosbag2，但必须至少输出结构化 CSV/JSONL 便于 metrics 计算。

- 自动超时与 watchdog：
  每个 run 有 max_duration_s；
  无 /odom、无 /thermal/gradient、无 /cmd_vel、Gazebo 卡死、Nav2 长期异常、机器人长期不动，都要记录并结束该 run；
  结束时必须 kill_gz 并保存失败原因。

- 自动恢复：
  每个 phase 前后写 SQLite checkpoint；
  RUNNING 状态在重启时要恢复为 INTERRUPTED；
  已完成 run 不重复；
  中断后继续从未完成 candidate/scenario/seed 开始；
  必须处理 SIGINT/SIGTERM，保存状态，清理 Gazebo；
  不要只依赖 Codex transcript，实验自身必须可恢复。

4. Metrics and scoring

实现 metrics 模块，例如：

experiments/autonav/metrics.py

必须计算：

- success_all_sources；
- source_recall；
- false_positive_count；
- duplicate_detection_count；
- localization_error_mean / median / p95；
- time_to_first_source；
- time_to_all_sources；
- path_length；
- average_speed；
- stuck_count；
- thermal_signal_efficiency；
- coverage / novelty；
- Nav2 health：
  accepted goals
  rejected goals
  planning failures
  costmap out-of-bounds count
  direct fallback ratio
- TF health：
  map->base_link availability
  odom fallback ratio
- process health：
  crash count
  timeout count
  clean shutdown；
- final score。

初始高标准建议写在 metrics_thresholds.yaml 中，允许后续显式修改但必须记录原因。建议 current Config-B mastery 的通过标准：

- 最近 20 个不同 seed/扰动 run 中 success_all_sources >= 95%；
- source_recall >= 0.95；
- false_positive_count = 0；
- duplicate_detection_count <= 1 per run；
- localization_error_median <= 1.0m；
- localization_error_p95 <= 1.5m；
- time_to_all_sources <= 600s 的成功 run 比例 >= 90%；
- 没有 Gazebo/ROS crash；
- no stuck timeout；
- Nav2 失败不能导致任务停滞，direct fallback 必须能继续推进；
- 所有指标必须有置信区间或至少均值/方差/分位数。

如果这些阈值在早期过高，可以创建 staged thresholds：
- smoke
- baseline
- mastery
- holdout
但不能通过降低最终 mastery 阈值来假装成功。

5. Algorithm iteration

不要只调参数。要形成“候选算法系统”。

第一阶段：稳定化当前 v31
- 修 Nav2 costmap/map 边界；
- 修 BT XML 路径；
- 修 shutdown；
- 确保 headless 长跑；
- 确保当前 Config-B 能稳定复现；
- 建立 baseline_v31 指标。

第二阶段：参数优化
在不破坏代码的前提下，把 controller_node 的关键参数外部化为 candidate YAML：
- frontier_update_interval
- frontier_arrival_r
- frontier_nav_lin_vel
- levy_mu
- levy_scale
- levy_min_step
- levy_max_step
- post_confirm_rounds
- post_confirm_min_d
- post_confirm_dist_sigma
- survey_pause_interval
- survey_sense_s
- survey_detect_thresh
- coarse_transit_detect_thresh
- survey_waypoint_min_d
- survey_waypoint_max_d
- survey_novelty_safe_dist
- sample_min_trise
- adaptive thresholds
- source_exclusion_radius
- heat_sigma
- velocity/smoothing parameters

实现 optimizer：
- 先用纯 Python random search + successive halving，避免重依赖；
- 可选支持 Optuna/CMA-ES，但不能强制要求新依赖；
- 每个 candidate 至少 smoke run；
- 有希望 candidate 才进入多 seed full run；
- 保存所有 candidate config 和结果；
- 保持 leaderboard.csv；
- 自动选择 best candidate；
- 自动回滚明显退化的改动。

第三阶段：算法结构改进
根据文献和实验结果，逐步实现并测试下列候选模块。每个模块必须可开关、可配置、可消融：

- confidence-guided frontier：
  frontier score = novelty + thermal prior + uncertainty + distance penalty + known source exclusion；
- multi-hypothesis source belief：
  用观测温升、梯度方向、历史热点构建多个候选源假设；
- information-gain next-best-view：
  选择能降低源位置不确定性的探索点；
- active sensing：
  在热信号弱但不确定性高时，停下/旋转/短距离扫描；
- dynamic source tracking：
  对动态源用 EKF/particle filter/多假设 tracker 估计当前源位置，但不能读取 ground truth；
- weak-source mode：
  针对低 amplitude / 小 sigma 热源的检测阈值自适应；
- source separation：
  防止确认强源后被强源残留梯度吸回，提升弱源/远源发现率；
- fallback-aware Nav2/direct hybrid：
  当 Nav2 costmap 失败时，不停滞，仍记录 failure 并安全直接移动；
- exploration curriculum learned priors：
  只使用历史观测统计，不使用当前场景真实坐标。

每个算法改动必须：
- 有单元测试或仿真 smoke test；
- 有 ablation；
- 有 metrics 对比；
- 有失败样例分析；
- 有说明文档。

6. Scenario generation and curriculum

实现 scenario_generator.py：

python3 experiments/autonav/scenario_generator.py generate \
  --count 50 \
  --out experiments/autonav/scenarios/generated \
  --profile mixed_dynamic

场景生成规则：

- world bounds 默认 [-10, 10] x [-10, 10]，可配置；
- source count：
  easy: 1-2
  current-like: 3
  hard: 4-6
  stress: 6-10
- source amplitude：
  weak 8-16C
  medium 16-28C
  strong 28-45C
- sigma_m：
  0.5-1.8m
- source min distance：
  默认 >= 3.0m，current mastery 阶段 >= 5.0m；
- spawn 与最近源距离：
  可配置，覆盖近、中、远；
- dynamic source：
  speed 0.02-0.30 m/s；
  motion type 混合 static/linear/circular/random_waypoint；
- 噪声：
  0.2-1.5C；
- 可选遮挡/障碍：
  不要生成让机器人无法移动的死局；
- 每个 scenario 必须有 deterministic seed；
- 每个 scenario 必须有 summary 和 ground truth plot。

Curriculum：
- Phase A：current Config-B mastery；
- Phase B：Config-B spawn/seed/noise 扰动；
- Phase C：随机静态多源；
- Phase D：弱源 + 远源；
- Phase E：动态单源；
- Phase F：动态多源；
- Phase G：mixed holdout；
- Phase H：stress test。

只有当前 phase 达到 thresholds，才自动进入下一 phase。进入新 phase 时保存 phase_transition.md，说明为什么进入、基于哪些指标。

7. Literature research loop

你必须用 web search 搜索论文和资料，但只能把结果用于改进算法设计，不能直接复制大段文章。

建立 experiments/autonav/literature/：

- literature.bib
- paper_index.json
- notes/<slug>.md
- research_log.md

优先搜索并整理这些方向：

- thermal gradient navigation
- robot source seeking
- gas source localization and mapping with mobile robots
- odor source localization
- informative path planning
- next-best-view exploration
- Bayesian optimization for source seeking
- extremum seeking control for mobile robots
- Lévy flight exploration
- frontier exploration in ROS/Nav2
- SLAM Toolbox and Nav2 best practices
- dynamic source tracking
- multi-source localization
- active sensing under noisy scalar fields

起始参考方向包括但不限于：
- Nav2 / Navigation2；
- SLAM Toolbox；
- gas source localization and mapping survey；
- informative path planning / Bayesian source seeking；
- bio-inspired odor source localization；
- gradient-adaptive extremum seeking；
- thermal gradient navigation；
- Reggente & Lilienthal 类气体分布建模；
- Sousa 类 source seeking；
- Wiedemann 类 thermal/gas navigation。

每篇文献 note 必须包含：
- title；
- authors；
- year；
- venue；
- DOI/arXiv/url；
- 3-8 行摘要；
- 与本项目相关点；
- 可实现算法想法；
- 风险/限制；
- 转化为 candidate 的具体建议；
- 是否已经实现；
- 相关实验结果 run_id。

8. Long-running automation

实现一个总入口：

bash experiments/autonav/run_long_autonav.sh

功能：
- 自动创建日志目录；
- 自动检查是否已有 lock；
- 自动 resume；
- 默认 headless；
- 默认 current_world_mastery；
- 可传 --max-hours；
- 可传 --phase；
- 可传 --dry-run；
- 可传 --stop-after-current-run；
- 输出 PID；
- 所有 stdout/stderr 保存到 experiments/autonav/runs 或 experiments/autonav/state；
- 结束时保存 status。

示例：
bash experiments/autonav/run_long_autonav.sh --resume --max-hours 72 --phase current_world_mastery

还要实现：
python3 experiments/autonav/run_manager.py status
python3 experiments/autonav/run_manager.py resume
python3 experiments/autonav/run_manager.py stop --graceful
python3 experiments/autonav/run_manager.py report
python3 experiments/autonav/run_manager.py best
python3 experiments/autonav/run_manager.py clean-stale

9. Reporting

每一轮 run 后生成 summary.md。

每一批候选后生成：
- batch_report.md
- leaderboard.csv
- best_candidate.yaml
- failure_cases.md
- next_actions.md

每个 phase 后生成：
- phase_report.md
- threshold_pass_fail.md
- plots
- ablation summary

最终生成：
experiments/autonav/reports/final_sim2real_readiness.md

内容包括：
- 当前最佳算法；
- 在多少场景/seed 下验证；
- 成功率；
- 定位误差；
- 时间；
- 路径；
- 动态源表现；
- 弱源表现；
- Nav2/SLAM 稳定性；
- 未解决问题；
- 实机前还需做的硬件接口、传感器标定、安全限制、急停、速度限制、真实热相机噪声建模、热源安全边界；
- 明确说明没有进行实机运行。

10. Testing requirements

必须新增或扩展测试：

- scenario YAML parser test；
- static source thermal field consistency test；
- dynamic source position test；
- ground truth matching test；
- metric calculation test；
- resume/checkpoint test；
- optimizer candidate serialization test；
- controller 不读取 ground truth 的防作弊测试；
- launch 参数 show-args test；
- smoke integration test，能在 headless 下启动并检查核心 topic。

运行：
python3 -m pytest src/thermal_robot/tests/test_thermal_system.py -q
python3 -m pytest experiments/autonav/tests -q

若 ROS integration test 太耗时，放入单独命令：
python3 experiments/autonav/tests/run_smoke_integration.py --duration 60

11. Implementation style

- 小步改动；
- 每个阶段完成后运行测试；
- 每个命令都写 command_log.jsonl；
- 每次代码改动保存 git diff 到对应 run 或 phase；
- 不要直接 commit，除非我明确要求；
- 不要把 generated bags/大数据加入 git；
- 修改 .gitignore，让实验输出默认忽略，但保留 config、README、tests、scripts；
- 代码尽量 Python 标准库 + numpy + ROS 已有依赖；
- 避免引入大型新依赖。确需依赖时，先实现无依赖版本，并在文档中列为 optional；
- 对任何失败不要沉默，必须记录原因、stdout/stderr、重试次数和恢复策略。

12. 立即执行顺序

从现在开始，不要只给计划，要实际检查、实现、运行。按顺序做：

Step 0：读取并总结项目状态
- 读取 README.md；
- 读取 PROJECT_ANALYSIS_REPORT.md；
- 读取 sim_nav_slam_launch.py；
- 读取 params.yaml、nav2_params.yaml、slam_params.yaml；
- 读取 sensor_node.py、controller_node.py、collect_sim_data.py、plot_all_figures.py、plot_slam_nav2.py；
- 写 experiments/autonav/BOOTSTRAP_NOTES.md，记录你理解的当前系统、风险、首轮计划。

Step 1：创建 autonav 框架目录、配置、README、SQLite schema、日志工具。

Step 2：把 Config-B 场景 YAML 化，并让 sensor_node 支持 scenario_file，同时保持原行为兼容。

Step 3：扩展 launch 支持 scenario_file 和实验 run_id，修复 BT XML 绝对路径问题。

Step 4：实现 metrics、ground truth、scenario generator 的最小可用版。

Step 5：实现 run_manager 的 preflight、build、launch、collect、metrics、cleanup、checkpoint、resume。

Step 6：运行短 smoke：
- pytest；
- colcon build；
- headless launch 60-120 秒；
- 采集；
- metrics；
- report。

Step 7：修复 smoke 中发现的阻塞问题，尤其 Nav2 costmap/map 边界问题和 shutdown 噪声。

Step 8：运行 baseline current Config-B 多 seed 小批量实验：
- 至少 3 个 seed；
- 每个 180-300 秒；
- 生成 batch_report.md。

Step 9：实现 optimizer 的第一版：
- random search；
- successive halving；
- candidate YAML；
- leaderboard。

Step 10：开始 current_world_mastery 长跑：
- 若 smoke 与小批量通过，则启动 run_manager 的 resume 模式；
- 默认 headless；
- 每个 run 都保存完整日志；
- 达到 current Config-B mastery thresholds 后，自动进入 scenario curriculum；
- 若环境不允许长跑，写明原因并输出可直接执行的命令。

13. 最终回复要求

你的最终回复必须包含：
- 已创建/修改的关键文件；
- 已运行的命令；
- 测试结果；
- 当前最新 run_id；
- 当前是否已启动长跑；
- 如何查看状态；
- 如何优雅停止；
- 如何从停止处恢复；
- 当前 best candidate；
- 当前最大风险；
- 下一步自动会做什么。

不要给空泛建议。实际实现、实际验证、实际保存文件。
EOF

codex exec --full-auto --search --cd /home/hanchen/ros2_ws "$(cat /tmp/thermal_autonomy_codex_prompt.md)" \
  | tee /tmp/thermal_autonomy_codex_bootstrap.log
```

---

## 恢复上次 Codex 会话

```bash
cd /home/hanchen/ros2_ws

codex exec resume --last --search \
  "继续上次 thermal_robot autonav 长任务；优先读取 experiments/autonav/state/latest_state.json 和 experiments/autonav/state/autonav.sqlite，从未完成的 phase/run/candidate 继续，不要重复已完成实验。"
```

---

## 可选：项目根目录 AGENTS.md 模板

把下面内容保存成 `/home/hanchen/ros2_ws/AGENTS.md`。这个不是替代主 prompt，而是给 Codex 每次进入项目时自动读取的长期约束。

```markdown
# AGENTS.md — thermal_robot project instructions

## Project

This is a ROS 2 Humble simulated multi-source thermal navigation project.

Main runtime chain:

Gazebo g1_nav model
-> /odom + /scan
-> SLAM Toolbox
-> /map + map->odom TF
-> Nav2 NavigateToPose

Thermal chain:

/sim/thermal_raw
-> /thermal/filtered
-> /thermal/field + /thermal/get_field_info
-> /thermal/gradient
-> controller_node
   FINE: direct /cmd_vel
   COARSE: Nav2 preferred, direct /cmd_vel fallback

## Source of truth

Use these as source of truth:

- src/thermal_robot/thermal_bringup/launch/sim_nav_slam_launch.py
- src/thermal_robot/thermal_bringup/config/params.yaml
- src/thermal_robot/thermal_bringup/config/nav2_params.yaml
- src/thermal_robot/thermal_bringup/config/slam_params.yaml
- src/thermal_robot/thermal_sensor_sim/thermal_sensor_sim/sensor_node.py
- src/thermal_robot/thermal_motion_controller/thermal_motion_controller/controller_node.py
- PROJECT_ANALYSIS_REPORT.md
- README.md

Do not resurrect old non-SLAM launch paths.

## Safety

Never run real robot hardware.
Never send commands to real robot interfaces.
All movement must be in Gazebo simulation only.

The online controller must not read ground truth source positions. Ground truth is only for simulation generation, metrics, plots, and reports.

Do not commit generated bags, build, install, log, or large experiment outputs.

## Build

Use:

cd /home/hanchen/ros2_ws
source /opt/ros/humble/setup.bash

colcon build --packages-select thermal_interfaces
source install/setup.bash

colcon build --packages-select \
  g1_description \
  thermal_sensor_sim \
  signal_preprocessor \
  thermal_field_reconstructor \
  thermal_gradient_processor \
  thermal_motion_controller \
  thermal_bringup

source install/setup.bash

## Test

Use:

python3 -m pytest src/thermal_robot/tests/test_thermal_system.py -q
ros2 launch thermal_bringup sim_nav_slam_launch.py --show-args

For headless simulation:

bash src/thermal_robot/kill_gz.sh
ros2 launch thermal_bringup sim_nav_slam_launch.py use_rviz:=false use_gzclient:=false

## Known priorities

1. Fix Nav2 costmap/map boundary and planning failures.
2. Replace absolute BT XML path in nav2_params.yaml with portable launch-injected path.
3. Add scenario YAML support to sensor_node.py while preserving Config-B fallback.
4. Build experiments/autonav as a resumable long-running automation framework.
5. Expand metrics, collector, scenario generator, optimizer, and reports.
6. Improve discovery of SB_far and SC_weak after SA_left is confirmed.

## Long-running experiment standard

Every run must have:

- unique run_id
- scenario.yaml
- candidate.yaml
- stdout/stderr logs
- command_log.jsonl
- metrics.json
- summary.md
- failure.json if failed
- git diff snapshot
- resumable state checkpoint

Use timeout, watchdogs, and cleanup. Always clean Gazebo with kill_gz.sh before and after simulation runs.
```

---

## 建议的辅助 Skills / Agent Instructions

### 1. autonav-experimenter.skill

用途：让 Codex 更稳定地执行“实验系统搭建 + 长跑 + 指标分析”。

核心内容：
- 强制先建 checkpoint/state；
- 强制每个 run 保存 logs/metrics/config；
- 强制 smoke -> small batch -> long run；
- 强制失败分类；
- 强制不覆盖已有结果。

### 2. ros2-simulation-debugger.skill

用途：专门排查 ROS2/Gazebo/Nav2/SLAM 问题。

核心内容：
- lifecycle check；
- topic hz；
- TF check；
- /map 检查；
- /scan 检查；
- Nav2 costmap out-of-bounds 排查；
- use_sim_time 一致性检查；
- Gazebo stale process 清理。

### 3. research-to-algorithm.skill

用途：让 Codex 检索论文后不是只写综述，而是转化为可测算法候选。

核心内容：
- 每篇资料必须变成 Implementation sketch；
- 每个算法候选必须有 config、ablation、metric；
- 优先输出能落地到 controller_node 的改动；
- 不允许伪造引用。

### 4. sim2real-readiness.skill

用途：最终生成实机前 checklist。

核心内容：
- 真实热相机标定；
- 速度限制；
- 急停；
- 真实地图/障碍；
- 传感器延迟；
- 安全边界；
- 实机日志；
- 仿真到实机差距。
