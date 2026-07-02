# YOPO AirSim ROS Bridge

## 项目简介

该项目是一个面向科研实验的轻量桥接器，用于将 YOPO / Ego-Planner 的位置速度指令稳定映射到 AirSim ROS 速度控制话题。

当前版本目标是最小可运行与稳定可复现：

- 有上游指令时：进入自动接管并跟踪 `/planning/pos_cmd`。
- 无上游指令时：进入悬停保持（Hover Hold）。
- GUI 仅提供生命周期按钮与状态显示，不提供飞行路径操控能力。

## 模块结构

- [main_airsim_ros.py](main_airsim_ros.py)
  - 主控制循环。
  - 读取 AirSim 状态，执行 YOPO 接管与悬停保持切换。
  - 发布速度命令前进行平滑与斜率限制。

- [ros_topic_manager.py](ros_topic_manager.py)
  - ROS 话题收发层。
  - 管理速度发布、里程计订阅、`pos_cmd` 订阅。
  - 提供话题类型自适应（VelCmd/Twist）与重映射发现。

- [pid_controller.py](pid_controller.py)
  - 通用 PID 控制器。
  - 支持外部 dt、微分低通与抗积分饱和。

- [gui.py](gui.py)
  - 只读状态面板。
  - 提供 Takeoff / Land / Reset 生命周期按钮。
  - 显示模式、速度、控制状态。

- [run.sh](run.sh)
  - 启动脚本。
  - 自动 source 依赖工作空间后启动主程序。

## ROS 交互

### 订阅

- `/planning/pos_cmd`（`quadrotor_msgs/PositionCommand`）
- `/airsim_node/drone_1/odom_local_ned`（`nav_msgs/Odometry`）

### 发布

- `/airsim_node/drone_1/vel_cmd_world_frame`（优先 `airsim_ros_pkgs/VelCmd`，兼容 `geometry_msgs/Twist`）
- `/airsim_node/drone_1/odom_local_enu`（转换后的 ENU 里程计）
- `/pcl_render_node/camera_pose`（相机位姿）

## 运行

本仓库运行时需要三个组件配合：

### 1. 启动桥接主程序

```bash
python3 main_airsim_ros.py
```

可以在 `main_airsim_ros.py` 中修改 `OBSTACLE_PUBLISH_DT` 等参数以适应不同场景。

### 2. 启动 YOPO 规划器（放入 YOPO 原仓库运行）

```bash
python test_yopo_ros.py --trial=1 --epoch=50
```

### 3. 启动 RViz 可视化

```bash
rviz -d yopo-airsim.rviz
```
