# aim_for_laser_detection_module

工业级无人机激光检测模块识别骨架（C++20, no ROS2），面向 RoboMaster 雷达视角场景。

## 设计目标
- 复用 `wust_vl` 底层能力：
  - `wust_vl::video::Camera`
  - `wust_vl::algorithm::PnPSolver`
  - `wust_vl::common::utils::{ParamGroup, Parameter, ParameterManager}`
  - `wust_vl` 日志与并发线程池
- 复用项目内 `KalmanHyLib` 做运动状态估计
- 沿用 `wust_vision` 的配置热更新与 debug 输出风格（`/debug_frame`, `/dev/shm/cmd_log.json`）

## 目录
- `include/laser_aim/common`: 核心数据结构
- `include/laser_aim/modules/config`: 参数组定义与配置中心
- `include/laser_aim/modules/perception`: 多阶段识别 pipeline 框架
- `include/laser_aim/modules/tracking`: Kalman 跟踪封装
- `include/laser_aim/app`: 节点生命周期与入口接口
- `include/laser_aim/utils`: debug_utils / env 展开 / 周期任务工具
- `src/modules/*`: 业务模块实现
- `src/app/*`: 应用层调度与进程入口
- `config/*`: 配置模板

## 当前 pipeline 骨架映射
1. 输入与状态初始化：`SystemConfig -> TeamPolicy`
2. 采集与预处理：`LaserVisionNode::frameCallback/processFrame`
3. 双路候选生成：`LaserPipeline::runDualCandidateGeneration`（传统路已接，学习路预留）
4. 候选颜色判别：`LaserPipeline::classifyColor`
5. 激光模块 vs 装甲板区分：`LaserPipeline::classifyLaserModule`（双模板PnP）
6. 关键点/PnP/中心点：`refineAndSolvePose + solveAimCenter`
7. 时序跟踪（颜色状态）：`tracking::LaserTrackFilter`
8. 锁定阶段与缩窗：`updateLockStage + currentStageScale`（支持 `stage_ref` 规则直驱）
9. 控制与发射门控：`evaluateGate`（enemy/laser/PnP/EKF稳定/预测点窗口 五重门控）
10. 失效保护：`TeamPolicy::isSafeTrackOnly` + `recapture_mode`（低置信切传统路快捕）

## 构建与运行
```bash
./run.sh build
./run.sh run 1
```
- `run 1` 表示开启 debug 模式（输出共享内存调试图 + `/dev/shm/cmd_log.json`）

## 工程状态
这是“工业级骨架”版本：
- 架构、配置、调试链路、并发和跟踪层已就位
- 具体模型推理（YOLO11-pose）与传统视觉规则可在已预留接口中逐步替换为正式实现
