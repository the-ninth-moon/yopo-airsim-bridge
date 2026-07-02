import rospy
import std_msgs.msg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from threading import Lock
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs import point_cloud2

import cv2
import os
import time
import torch
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

from config.config import cfg
from control_msg import PositionCommand
from policy.yopo_network import YopoNetwork
from policy.poly_solver import *
from policy.state_transform import *

try:
    from torch2trt import TRTModule
except ImportError:
    print("tensorrt not found.")


class YopoNet:
    """
    YOPO 在线推理与控制节点（AirSim 版本）。

    核心流程：
    1) 订阅里程计和深度图。
    2) 组合观测并调用网络推理。
    3) 将网络预测的终端状态转为五次多项式轨迹。
    4) 定时发布 PositionCommand 给下游控制器。
    """

    def __init__(self, config, weight):
        # 外部传入运行配置（topic、目标点、是否可视化、是否 TensorRT 等）。
        self.config = config
        rospy.init_node('yopo_net', anonymous=False)

        # load params
        # 切到测试模式：primitive.py 会根据 train 标志做速度/时间比例切换。
        cfg["train"] = False

        # 网络输入尺寸与深度图有效距离范围。
        self.height = cfg['image_height']
        self.width = cfg['image_width']
        self.min_dis, self.max_dis = 0.04, 20.0

        # 任务与运行开关。
        self.goal = np.array(self.config['goal'])
        self.plan_from_reference = self.config['plan_from_reference']
        self.use_trt = self.config['use_tensorrt']
        self.verbose = self.config['verbose']
        self.visualize = self.config['visualize']

        # 相机相对机体的旋转（这里仅用 pitch），用于 body<->camera 坐标换算。
        self.Rotation_bc = R.from_euler('ZYX', [0, self.config['pitch_angle_deg'], 0], degrees=True).as_matrix()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # variables
        # 里程计与控制状态缓存。
        self.odom = Odometry()
        self.odom_init = False
        self.last_yaw = 0.0
        self.ctrl_dt = 0.02
        self.ctrl_time = None
        self.desire_init = False
        self.arrive = False
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.optimal_poly_x = None
        self.optimal_poly_y = None
        self.optimal_poly_z = None
        self.lock = Lock()
        self.last_control_msg = None

        # 状态变换器与离散 primitive 网格。
        self.state_transform = StateTransform()
        self.lattice_primitive = LatticePrimitive.get_instance()
        self.traj_time = self.lattice_primitive.segment_time

        # eval
        # 各阶段耗时统计，用于实时性分析。
        self.time_forward = 0.0
        self.time_process = 0.0
        self.time_prepare = 0.0
        self.time_interpolation = 0.0
        self.time_visualize = 0.0
        self.count = 0
        self.depth_fps = 30  # used only as processing time tolerance for printing logs

        # Load Network
        # 两种后端：PyTorch / TensorRT。接口都保持 self.policy(depth, obs)。
        if self.use_trt:
            self.policy = TRTModule()
            self.policy.load_state_dict(torch.load(weight))
        else:
            state_dict = torch.load(weight, weights_only=True)
            self.policy = YopoNetwork()
            self.policy.load_state_dict(state_dict)
            self.policy = self.policy.to(self.device)
            self.policy.eval()

        # 预热一次，减少首帧冷启动抖动。
        self.warm_up()

        # ros publisher
        # 轨迹可视化 + 控制指令发布。
        self.lattice_traj_pub = rospy.Publisher("/yopo_net/lattice_trajs_visual", PointCloud2, queue_size=1)
        self.best_traj_pub = rospy.Publisher("/yopo_net/best_traj_visual", PointCloud2, queue_size=1)
        self.all_trajs_pub = rospy.Publisher("/yopo_net/trajs_visual", PointCloud2, queue_size=1)
        self.ctrl_pub = rospy.Publisher(self.config["ctrl_topic"], PositionCommand, queue_size=1)

        # ros subscriber
        # 深度图触发主规划流程；里程计更新状态；RViz 点击可动态改目标。
        self.odom_sub = rospy.Subscriber(self.config['odom_topic'], Odometry, self.callback_odometry, queue_size=1, tcp_nodelay=True)
        self.depth_sub = rospy.Subscriber(self.config['depth_topic'], Image, self.callback_depth, queue_size=1, tcp_nodelay=True)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.callback_set_goal, queue_size=1)

        # ros timer
        # 控制频率独立于深度帧率，定时采样当前最优多项式并发布。
        rospy.sleep(1.0)  # wait connection...
        self.timer_ctrl = rospy.Timer(rospy.Duration(self.ctrl_dt), self.control_pub)
        print("YOPO Net Node Ready!")
        rospy.spin()

    def callback_set_goal(self, data):
        # 接收 RViz 2D Nav Goal，z 固定为 2m（可按实际场景改造）。
        self.goal = np.asarray([data.pose.position.x, data.pose.position.y, 2])
        self.arrive = False
        print(f"New Goal: ({data.pose.position.x:.1f}, {data.pose.position.y:.1f})")

    # the first frame
    def callback_odometry(self, data):
        # 更新最新里程计。
        self.odom = data

        # 第一次收到里程计时初始化参考状态。
        # yopo只会使用速度、位置、姿态。加速度不直接来自传感器，而是通过期望状态更新（在 control_pub 里），因此这里初始化为 0。
        if not self.desire_init:
            self.desire_pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            self.desire_vel = np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
            self.desire_acc = np.array((0.0, 0.0, 0.0))
            ypr = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                               self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_euler('ZYX', degrees=False)
            self.last_yaw = ypr[0]
        self.odom_init = True

        # 到达判定：仅用于停止继续发布 READY 轨迹。
        pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        if np.linalg.norm(pos - self.goal) < 5 and not self.arrive:
            print("Arrive!")
            self.arrive = True

    def process_odom(self):
        """
        将里程计与目标信息整理为网络状态输入 obs。

        输出：
        - obs_norm: [1, 9]，依次为 vxyz, axyz, goal_xyz（相机坐标系，且已归一化）。
        """

        # Rwb -> Rwc -> Rcw
        # Rwb: body->world；Rbc: camera->body；Rwc: camera->world。
        Rotation_wb = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                                   self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_matrix()
        self.Rotation_wc = np.dot(Rotation_wb, self.Rotation_bc)
        Rotation_cw = self.Rotation_wc.T

        # vel and acc
        # plan_from_reference=True 时使用上一周期期望状态，更平滑。
        vel_w = self.desire_vel if self.plan_from_reference else np.array([self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z])
        vel_c = np.dot(Rotation_cw, vel_w)

        # 当前实现中加速度来自期望状态（而非直接传感器估计）。
        acc_w = self.desire_acc
        acc_c = np.dot(Rotation_cw, acc_w)

        # goal_dir
        # 目标方向向量同样转到相机坐标系，与深度图视角保持一致。
        goal_w = self.goal - self.desire_pos
        goal_c = np.dot(Rotation_cw, goal_w)

        # 拼接成 9 维观测并做归一化。
        obs = np.concatenate((vel_c, acc_c, goal_c), axis=0).astype(np.float32)
        obs_norm = self.state_transform.normalize_obs(torch.from_numpy(obs[None, :]))
        return obs_norm

    @torch.inference_mode()
    def callback_depth(self, data):
        # 还没拿到里程计时，无法完成坐标变换，直接跳过。
        if not self.odom_init: return

        # 1. Depth Image Process (Be careful with the depth units in your application)
        time0 = time.time()
        if data.encoding == "32FC1":    # Simulator, meter
            depth = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width)
        elif data.encoding == "16UC1":  # RealSense, millimeter
            depth = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width).astype(np.float32) / 1000.0
        else:
            raise ValueError(f"Unsupported depth encoding: {data.encoding}. Expected '32FC1' or '16UC1'.")

        # 输入尺寸对齐到网络期望分辨率。
        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        # 裁剪并归一化到 [0,1]。
        depth = np.minimum(depth, self.max_dis) / self.max_dis

        # interpolated the nan value (experiment shows that treating nan directly as 0 produces similar results)
        # 将 NaN / 过近异常值掩膜后做 inpaint，减小无效深度对网络的干扰。
        nan_mask = np.isnan(depth) | (depth < self.min_dis / self.max_dis)
        
        interpolated_image = cv2.inpaint(np.uint8(depth * 255), np.uint8(nan_mask), 1, cv2.INPAINT_NS)
        interpolated_image = interpolated_image.astype(np.float32) / 255.0

        # 网络输入形状：[B,C,H,W] = [1,1,H,W]。
        depth = interpolated_image.reshape([1, 1, self.height, self.width])
        # cv2.imshow("1", depth[0][0])
        # cv2.waitKey(1)

        # 2. YOPO Network Inference
        # input prepare
        time1 = time.time()
        depth_input = torch.from_numpy(depth).to(self.device, non_blocking=True)  # (non_blocking: copying speed 3x)
        obs_norm = self.process_odom().to(self.device, non_blocking=True)

        # 将 [1,9] 展开到 primitive 维度，得到 [1,9,V,H]。
        obs_input = self.state_transform.prepare_input(obs_norm)
        # torch.cuda.synchronize()

        time2 = time.time()
        # Forward (TensorRT: inference speed increased by 5x)
        endstate_pred, score_pred = self.policy(depth_input, obs_input)

        # 后处理主要在 numpy 上完成（CPU），减少小张量频繁 CUDA 操作开销。
        endstate_pred, score_pred = endstate_pred.cpu().numpy(), score_pred.cpu().numpy()
        time3 = time.time()

        # 3. Post-Processing
        # Replacing PyTorch operation on CUDA with NumPy operation on CPU (speed increased by 10x)
        endstate, score = self.process_output(endstate_pred, score_pred, return_all_preds=self.visualize)

        # Vectorization: transform the prediction(P V A in body frame) to the world frame with the attitude (without the position)
        # endstate_c: [N,3,3]，每条轨迹按 [p,v,a] 组织。
        endstate_c = endstate.reshape(-1, 3, 3).transpose(0, 2, 1)  # [N, 9] -> [N, 3, 3] -> [px vx ax, py vy ay, pz vz az]
        endstate_w = np.matmul(self.Rotation_wc, endstate_c)

        # 若需要可视化全部预测，则按分数最小选最优；否则默认取第一条（已在 process_output 做筛选）。
        action_id = np.argmin(score) if self.visualize else 0
        with self.lock:  # Python3.8: threads are scheduled using time slices, add the lock to ensure safety
            # 轨迹起点可选择“参考状态”或“当前实测状态”。
            start_pos = self.desire_pos if self.plan_from_reference else np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            start_vel = self.desire_vel if self.plan_from_reference else np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))

            # 分轴构造五次多项式轨迹（位置、速度、加速度在段内连续）。
            self.optimal_poly_x = Poly5Solver(start_pos[0], start_vel[0], self.desire_acc[0], endstate_w[action_id, 0, 0] + start_pos[0],
                                              endstate_w[action_id, 0, 1], endstate_w[action_id, 0, 2], self.traj_time)
            self.optimal_poly_y = Poly5Solver(start_pos[1], start_vel[1], self.desire_acc[1], endstate_w[action_id, 1, 0] + start_pos[1],
                                              endstate_w[action_id, 1, 1], endstate_w[action_id, 1, 2], self.traj_time)
            self.optimal_poly_z = Poly5Solver(start_pos[2], start_vel[2], self.desire_acc[2], endstate_w[action_id, 2, 0] + start_pos[2],
                                              endstate_w[action_id, 2, 1], endstate_w[action_id, 2, 2], self.traj_time)

            # 重置控制轨迹时间，从新段开始发布。
            self.ctrl_time = 0.0
        time4 = time.time()

        # 发布可视化点云（仅在有订阅连接时执行，避免额外开销）。
        self.visualize_trajectory(score_pred, endstate_w)
        time5 = time.time()

        # 打印与统计各阶段耗时。
        self.print_time(time0, time1, time2, time3, time4, time5)

    def control_pub(self, _timer):
        # 还没有可执行轨迹或轨迹段已发布结束时，直接返回。
        if self.ctrl_time is None or self.ctrl_time > self.traj_time:
            return

        # 达到目标后发送 EMPTY 标记，通知下游控制器当前轨迹结束。
        if self.arrive and self.last_control_msg is not None:
            self.desire_init = False   # ready for next rollout
            self.last_control_msg.trajectory_flag = self.last_control_msg.TRAJECTORY_STATUS_EMPTY
            self.ctrl_pub.publish(self.last_control_msg)
            return

        with self.lock:  # Python3.8: threads are scheduled using time slices, add the lock to ensure safety and publish frequency
            # 按固定控制周期推进轨迹时间。
            self.ctrl_time += self.ctrl_dt
            control_msg = PositionCommand()
            control_msg.header.stamp = rospy.Time.now()
            control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_READY

            # 从五次多项式采样当前时刻的 p/v/a，填入控制消息。
            control_msg.position.x = self.optimal_poly_x.get_position(self.ctrl_time)
            control_msg.position.y = self.optimal_poly_y.get_position(self.ctrl_time)
            control_msg.position.z = self.optimal_poly_z.get_position(self.ctrl_time)
            control_msg.velocity.x = self.optimal_poly_x.get_velocity(self.ctrl_time)
            control_msg.velocity.y = self.optimal_poly_y.get_velocity(self.ctrl_time)
            control_msg.velocity.z = self.optimal_poly_z.get_velocity(self.ctrl_time)
            control_msg.acceleration.x = self.optimal_poly_x.get_acceleration(self.ctrl_time)
            control_msg.acceleration.y = self.optimal_poly_y.get_acceleration(self.ctrl_time)
            control_msg.acceleration.z = self.optimal_poly_z.get_acceleration(self.ctrl_time)

            # 将本次发布值回灌为下一次规划参考状态。
            self.desire_pos = np.array([control_msg.position.x, control_msg.position.y, control_msg.position.z])
            self.desire_vel = np.array([control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z])
            self.desire_acc = np.array([control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z])

            # 根据速度方向与目标方向计算平滑 yaw 与 yaw_rate。
            goal_dir = self.goal - self.desire_pos
            yaw, yaw_dot = calculate_yaw(self.desire_vel, goal_dir, self.last_yaw, self.ctrl_dt)
            self.last_yaw = yaw
            control_msg.yaw = yaw
            control_msg.yaw_dot = yaw_dot

            # 标记参考状态已初始化，并缓存最后一条控制消息。
            self.desire_init = True
            self.last_control_msg = control_msg
            self.ctrl_pub.publish(control_msg)

    def process_output(self, endstate_pred, score_pred, return_all_preds=False):
        """
        将网络输出张量重排并解码为物理终端状态。

        输入：
        - endstate_pred: 网络预测终端状态（归一化）
        - score_pred: 网络预测代价分数
        - return_all_preds: 是否返回全部候选（用于可视化）
        """

        # [9,V,H] -> [traj_num, 9]；score 对齐为 [traj_num]。
        endstate_pred = endstate_pred.reshape(9, self.lattice_primitive.traj_num).T
        score_pred = score_pred.reshape(self.lattice_primitive.traj_num)

        if not return_all_preds:
            # 推理模式：仅保留最优候选。
            action_id = np.argmin(score_pred)
            lattice_id = self.lattice_primitive.traj_num - 1 - action_id
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred[action_id, :][np.newaxis, :], lattice_id)
            score = score_pred[action_id]
        else:
            # 可视化模式：返回全部候选，便于在 RViz 展示全轨迹簇。
            score = score_pred
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred, torch.arange(self.lattice_primitive.traj_num-1, -1, -1))

        return endstate, score

    def visualize_trajectory(self, pred_score, pred_endstate):
        """发布三类可视化点云：最优轨迹、lattice 轨迹、全部预测轨迹。"""

        # 将一段轨迹离散为 20 个采样点用于可视化。
        dt = self.traj_time / 20.0
        start_pos = self.desire_pos if self.plan_from_reference else np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        start_vel = self.desire_vel if self.plan_from_reference else np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))

        # best predicted trajectory
        if self.best_traj_pub.get_num_connections() > 0:
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                self.optimal_poly_x.get_position(t_values),
                self.optimal_poly_y.get_position(t_values),
                self.optimal_poly_z.get_position(t_values)
            ), axis=-1)
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world_enu'
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array)
            self.best_traj_pub.publish(point_cloud_msg)

        # lattice primitive
        if self.visualize and self.lattice_traj_pub.get_num_connections() > 0:
            # primitive 节点从相机系旋转到世界系，再构造基准轨迹供对照。
            lattice_endstate = self.lattice_primitive.lattice_pos_node.cpu().numpy()
            lattice_endstate = np.dot(lattice_endstate, self.Rotation_wc.T)
            zero_state = np.zeros_like(lattice_endstate)
            lattice_poly_x = Polys5Solver(start_pos[0], start_vel[0], self.desire_acc[0],
                                          lattice_endstate[:, 0] + start_pos[0], zero_state[:, 0], zero_state[:, 0], self.traj_time)
            lattice_poly_y = Polys5Solver(start_pos[1], start_vel[1], self.desire_acc[1],
                                          lattice_endstate[:, 1] + start_pos[1], zero_state[:, 1], zero_state[:, 1], self.traj_time)
            lattice_poly_z = Polys5Solver(start_pos[2], start_vel[2], self.desire_acc[2],
                                          lattice_endstate[:, 2] + start_pos[2], zero_state[:, 2], zero_state[:, 2], self.traj_time)
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                lattice_poly_x.get_position(t_values),
                lattice_poly_y.get_position(t_values),
                lattice_poly_z.get_position(t_values)
            ), axis=-1)
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world_enu'
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array)
            self.lattice_traj_pub.publish(point_cloud_msg)

        # all predicted trajectories
        if self.visualize and self.all_trajs_pub.get_num_connections() > 0:
            # 将网络给出的所有终端状态展开为轨迹簇，并把 score 作为 intensity 方便着色。
            all_poly_x = Polys5Solver(start_pos[0], start_vel[0], self.desire_acc[0],
                                      pred_endstate[:, 0, 0] + start_pos[0], pred_endstate[:, 0, 1], pred_endstate[:, 0, 2], self.traj_time)
            all_poly_y = Polys5Solver(start_pos[1], start_vel[1], self.desire_acc[1],
                                      pred_endstate[:, 1, 0] + start_pos[1], pred_endstate[:, 1, 1], pred_endstate[:, 1, 2], self.traj_time)
            all_poly_z = Polys5Solver(start_pos[2], start_vel[2], self.desire_acc[2],
                                      pred_endstate[:, 2, 0] + start_pos[2], pred_endstate[:, 2, 1], pred_endstate[:, 2, 2], self.traj_time)
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                all_poly_x.get_position(t_values),
                all_poly_y.get_position(t_values),
                all_poly_z.get_position(t_values)
            ), axis=-1)
            scores = np.repeat(pred_score, t_values.size)
            points_array = np.column_stack((points_array, scores))
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world_enu'
            fields = [PointField('x', 0, PointField.FLOAT32, 1), PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1), PointField('intensity', 12, PointField.FLOAT32, 1)]
            point_cloud_msg = point_cloud2.create_cloud(header, fields, points_array)
            self.all_trajs_pub.publish(point_cloud_msg)

    def print_time(self, time0, time1, time2, time3, time4, time5):
        """
        Performance reference: PyTorch model should take < 5 ms; TensorRT model should take < 1 ms

        Notes:
        - Running program and enabling RViz under WSL greatly increase processing time, and Ubuntu does not have these issues
        - Even with queue_size=1, it may cause message accumulation and lag when processing time exceeds the image frequency
        """
        # 累计各阶段耗时（用于平均值统计）。
        self.time_interpolation = self.time_interpolation + (time1 - time0)
        self.time_prepare = self.time_prepare + (time2 - time1)
        self.time_forward = self.time_forward + (time3 - time2)
        self.time_process = self.time_process + (time4 - time3)
        self.time_visualize = self.time_visualize + (time5 - time4)
        self.count = self.count + 1

        # 以深度帧率推算实时预算（ms）。
        total_time = (time5 - time0) * 1000
        tolerance = 1000.0 / self.depth_fps

        # 超预算时打印告警与当前分解耗时。
        if total_time > tolerance:
            rospy.logwarn(f"Warn: Processing time {(time5 - time0) * 1000:.2f} ms exceeds {tolerance:.2f} ms, may cause message lag!")
            print(f"\033[34mCurrent Time Consuming:\033[0m "
                  f"depth-interpolation: \033[32m{1000 * (time1 - time0):.2f} ms\033[0m; "
                  f"data-prepare: \033[32m{1000 * (time2 - time1):.2f} ms\033[0m; "
                  f"network-inference: \033[32m{1000 * (time3 - time2):.2f} ms\033[0m; "
                  f"post-process: \033[32m{1000 * (time4 - time3):.2f} ms\033[0m; "
                  f"visualize-trajectory: \033[32m{1000 * (time5 - time4):.2f} ms\033[0m")

        # verbose 模式或超预算时，输出平均耗时概览。
        if self.verbose or (total_time > tolerance):
            print(f"\033[34mAverage Time Consuming:\033[0m "
                  f"depth-interpolation: \033[32m{1000 * self.time_interpolation / self.count:.2f} ms\033[0m; "
                  f"data-prepare: \033[32m{1000 * self.time_prepare / self.count:.2f} ms\033[0m; "
                  f"network-inference: \033[32m{1000 * self.time_forward / self.count:.2f} ms\033[0m; "
                  f"post-process: \033[32m{1000 * self.time_process / self.count:.2f} ms\033[0m; "
                  f"visualize-trajectory: \033[32m{1000 * self.time_visualize / self.count:.2f} ms\033[0m")

    def warm_up(self):
        # 用全零输入跑一遍，初始化 CUDA/TensorRT 内部缓存，降低首帧时延。
        depth = torch.zeros((1, 1, self.height, self.width), dtype=torch.float32, device=self.device)
        obs = torch.zeros((1, 9), dtype=torch.float32, device=self.device)
        obs = self.state_transform.prepare_input(obs)
        endstate_pred, score_pred = self.policy(depth, obs)
        _ = self.state_transform.pred_to_endstate(endstate_pred)


def parser():
    # 运行参数：是否使用 TensorRT、加载哪个 trial/epoch 权重。
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tensorrt", type=int, default=0, help="use tensorrt or not")
    parser.add_argument("--trial", type=int, default=1, help="trial number")
    parser.add_argument("--epoch", type=int, default=50, help="epoch number")
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 权重选择：TensorRT 固定文件名；PyTorch 按 trial/epoch 读取。
    weight = "yopo_trt.pth" if args.use_tensorrt else base_dir + "/saved/YOPO_{}/epoch{}.pth".format(args.trial, args.epoch)
    print("load weight from:", weight)

    # settings = {'use_tensorrt': args.use_tensorrt,
    #             'goal': [50, 0, 2],      # 目标点位置
    #             'pitch_angle_deg': -0,   # 相机俯仰角(仰为负)
    #             'odom_topic': '/sim/odom',                   # 里程计话题
    #             'depth_topic': '/depth_image',               # 深度图话题
    #             'ctrl_topic': '/so3_control/pos_cmd',        # 控制器话题
    #             'plan_from_reference': False,   # 从参考状态规划？位置控制器: True, 神经网络直接控制: False
    #             'verbose': False,               # 打印耗时？
    #             'visualize': True               # 可视化所有轨迹？(实飞改为False节省计算)
    #             }
    settings = {'use_tensorrt': args.use_tensorrt,
                'goal': [0, 0, 3],      # 目标点位置 (可以按需修改)
                'pitch_angle_deg': -0,   # 相机俯仰角(仰为负)
                
                # 1. 替换为你的 AirSim ENU 里程计话题
                'odom_topic': '/airsim_node/drone_1/odom_local_enu',                   
                
                # 2. 替换为你的 AirSim 深度图话题
                'depth_topic': '/airsim_node/drone_1/front_center_custom/DepthPlanar',               
                
                # 3. 替换为你控制器订阅的规划指令话题
                'ctrl_topic': '/planning/pos_cmd',        
                
                # 4. 【核心细节】改为 True！
                # 注释写了"位置控制器: True"。你的 main_airsim_ros.py 是典型的 PID 位置速度控制器，
                # 所以从期望参考状态（desire_state）规划会比直接从当前里程计规划更平滑。
                'plan_from_reference': True,
                
                'verbose': False,               
                'visualize': True               
                }

    # 启动节点（构造函数内部会 spin）。
    YopoNet(settings, weight)
