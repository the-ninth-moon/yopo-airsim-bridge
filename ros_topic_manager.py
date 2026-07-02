import time
import sys
import numpy as np
from glob import glob
from pathlib import Path

import rospy
import roslib.message
import rostopic
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import Odometry, Path as NavPath

try:
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray
    _HAS_MARKER_MSGS = True
except Exception:
    ColorRGBA = None
    Marker = None
    MarkerArray = None
    _HAS_MARKER_MSGS = False

POS_CMD_DEBUG_LOG = False
POS_CMD_DEBUG_LOG_INTERVAL_S = 5.0

try:
    from quadrotor_msgs.msg import PositionCommand  # type: ignore[import-not-found]
except ImportError:
    PositionCommand = None
    print("[WARN] 未找到 quadrotor_msgs，Ego-Planner 自动接管功能将被禁用。请确保已 source 你的工作空间。")


class RosTopicManager:
    """ROS 话题收发管理器：负责订阅状态话题并发布速度指令。""" 

    def __init__(self, drone_name="drone_1", node_name="pid_airsim_ros_bridge"):
        self.drone_name = drone_name

        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True, disable_signals=True)

        self.vel_topic = f"/airsim_node/{self.drone_name}/vel_cmd_world_frame"
        self.odom_topic = f"/airsim_node/{self.drone_name}/odom_local_ned"
        self.pos_cmd_topic = "/planning/pos_cmd"

        self.latest_odom = None
        self.latest_odom_stamp = None
        self.latest_pos_cmd = None
        self.latest_pos_cmd_stamp = 0.0
        self.pos_cmd_rx_count = 0

        # ========== [核心新增] 定义 ENU 里程计和相机位姿的发布者 ==========
        self.odom_enu_pub = rospy.Publisher(
            f"/airsim_node/{self.drone_name}/odom_local_enu", Odometry, queue_size=10
        )
        self.cam_pose_pub = rospy.Publisher('/pcl_render_node/camera_pose', PoseStamped, queue_size=10)
        # ====================================================================

        # --- RViz 可视化发布器 ---
        self.debug_frame = "world_enu"

        # Marker 发布器（用于障碍物可视化）
        self.marker_pub = None
        if _HAS_MARKER_MSGS:
            self.marker_pub = rospy.Publisher(
                f"/rl_debug/{self.drone_name}/markers", MarkerArray, queue_size=5, latch=True
            )
            rospy.loginfo(f"RViz marker publisher ready: /rl_debug/{self.drone_name}/markers")
        else:
            rospy.logwarn("visualization_msgs not available, marker topic disabled")
        # ---

        self._vel_msg_cls = self._resolve_velocity_msg_class(self.vel_topic)
        self.vel_pub = rospy.Publisher(self.vel_topic, self._vel_msg_cls, queue_size=10)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._on_odom, queue_size=10)
        if PositionCommand is not None:
            self.pos_cmd_sub = rospy.Subscriber(
                self.pos_cmd_topic,
                PositionCommand,
                self._on_pos_cmd,
                queue_size=1,
            )

        # 等待 publisher 注册到 master，避免刚启动时首条消息丢失。
        timeout_s = 2.0
        start = time.time()
        while self.vel_pub.get_num_connections() == 0 and (time.time() - start) < timeout_s:
            rospy.sleep(0.05)

    def _resolve_velocity_msg_class(self, topic_name):
        """根据运行时话题类型选择消息类，兼容 Twist 与 VelCmd。"""
        # 优先尝试加载 VelCmd，解决未 source catkin 环境导致的模块不可见问题。
        velcmd_cls = self._try_import_velcmd_class()
        if velcmd_cls is not None:
            return velcmd_cls

        wait_timeout_s = 10.0
        start = time.time()
        discovered_topic = None
        while (time.time() - start) < wait_timeout_s:
            # 先用 rostopic API 直接向 master 查询，通常比 get_published_topics 更稳。
            try:
                msg_cls, real_topic, _ = rostopic.get_topic_class(topic_name, blocking=False)
            except Exception:
                msg_cls, real_topic = None, None

            if msg_cls is not None:
                discovered_topic = real_topic or topic_name
                if discovered_topic != self.vel_topic:
                    rospy.logwarn(
                        f"velocity topic remapped: configured={self.vel_topic}, discovered={discovered_topic}"
                    )
                    self.vel_topic = discovered_topic
                return msg_cls

            published = dict(rospy.get_published_topics())
            if topic_name in published:
                discovered_topic = topic_name
                topic_type = published[topic_name]
            else:
                # 兜底：允许命名空间变化，按后缀自动发现。
                suffix_matches = [
                    t for t in published.keys() if t.endswith("/vel_cmd_world_frame")
                ]
                if not suffix_matches:
                    rospy.sleep(0.05)
                    continue
                discovered_topic = suffix_matches[0]
                topic_type = published[discovered_topic]

            if discovered_topic != self.vel_topic:
                rospy.logwarn(
                    f"velocity topic remapped: configured={self.vel_topic}, discovered={discovered_topic}"
                )
                self.vel_topic = discovered_topic

            if topic_type == "geometry_msgs/Twist":
                return Twist

            cls = roslib.message.get_message_class(topic_type)
            if cls is not None:
                return cls
            raise RuntimeError(
                f"检测到 {self.vel_topic} 类型为 {topic_type}，"
                "但当前 Python 环境无法加载该消息模块。"
            )

        rospy.logwarn(
            f"等待超时：未发现速度话题 {topic_name}，回退为 geometry_msgs/Twist 发布。"
            "如果 airsim_node 端实际是 VelCmd，请检查 ROS 环境后再运行。"
        )
        return Twist

    def _try_import_velcmd_class(self):
        """尝试导入 airsim_ros_pkgs.msg.VelCmd；必要时自动补充常见 catkin 路径。"""
        try:
            from airsim_ros_pkgs.msg import VelCmd  # type: ignore[import-not-found]

            return VelCmd
        except Exception:
            pass

        candidate_paths = []

        # 常见 AirSim ROS 工作区位置
        candidate_paths.extend(glob("/home/*/AirSim/ros/devel/lib/python3/dist-packages"))
        candidate_paths.extend(
            glob("/home/*/AirSim/ros/devel/.private/airsim_ros_pkgs/lib/python3/dist-packages")
        )

        # 当前用户主目录下的常见 catkin 工作区
        home = str(Path.home())
        candidate_paths.extend(glob(f"{home}/*/devel/lib/python3/dist-packages"))
        candidate_paths.extend(glob(f"{home}/catkin_ws/devel/lib/python3/dist-packages"))

        for p in candidate_paths:
            if p not in sys.path:
                sys.path.append(p)

        try:
            from airsim_ros_pkgs.msg import VelCmd  # type: ignore[import-not-found]

            rospy.loginfo("Loaded airsim_ros_pkgs.msg.VelCmd from discovered catkin path")
            return VelCmd
        except Exception:
            return None

    def _on_odom(self, msg):
        import math

        self.latest_odom = msg
        self.latest_odom_stamp = time.time()

        # ==========================================
        # 1. 位置和速度转换 (NED -> ENU) 完全正确
        # ==========================================
        x_enu = msg.pose.pose.position.y
        y_enu = msg.pose.pose.position.x
        z_enu = -msg.pose.pose.position.z

        vx_enu = msg.twist.twist.linear.y
        vy_enu = msg.twist.twist.linear.x
        vz_enu = -msg.twist.twist.linear.z

        # ==========================================
        # 2. 【核心修复】姿态四元数严谨转换 (NED -> ENU)
        # ==========================================
        w = msg.pose.pose.orientation.w
        x = msg.pose.pose.orientation.x
        y = msg.pose.pose.orientation.y
        z = msg.pose.pose.orientation.z

        # 步骤 A: 从 NED 四元数提取真实欧拉角 (Roll, Pitch, Yaw)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw_ned = math.atan2(siny_cosp, cosy_cosp)

        sinp = 2 * (w * y - z * x)
        pitch_ned = math.asin(sinp) if abs(sinp) <= 1 else math.copysign(math.pi / 2, sinp)
        
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll_ned = math.atan2(sinr_cosp, cosr_cosp)

        # 步骤 B: 严谨映射到 ENU 欧拉角
        yaw_enu = math.pi / 2.0 - yaw_ned
        pitch_enu = -pitch_ned
        roll_enu = roll_ned

        # 步骤 C: 将修正后的 ENU 欧拉角重新打包成纯正的四元数
        cy = math.cos(yaw_enu * 0.5)
        sy = math.sin(yaw_enu * 0.5)
        cp = math.cos(pitch_enu * 0.5)
        sp = math.sin(pitch_enu * 0.5)
        cr = math.cos(roll_enu * 0.5)
        sr = math.sin(roll_enu * 0.5)

        qw_enu = cr * cp * cy + sr * sp * sy
        qx_enu = sr * cp * cy - cr * sp * sy
        qy_enu = cr * sp * cy + sr * cp * sy
        qz_enu = cr * cp * sy - sr * sp * cy

        # 发布彻底修正后的 ENU 里程计
        enu_odom = Odometry()
        enu_odom.header.stamp = msg.header.stamp
        enu_odom.header.frame_id = "world_enu"
        
        enu_odom.pose.pose.position.x = x_enu
        enu_odom.pose.pose.position.y = y_enu
        enu_odom.pose.pose.position.z = z_enu
        
        enu_odom.pose.pose.orientation.x = qx_enu
        enu_odom.pose.pose.orientation.y = qy_enu
        enu_odom.pose.pose.orientation.z = qz_enu
        enu_odom.pose.pose.orientation.w = qw_enu
        
        enu_odom.twist.twist.linear.x = vx_enu
        enu_odom.twist.twist.linear.y = vy_enu
        enu_odom.twist.twist.linear.z = vz_enu

        self.odom_enu_pub.publish(enu_odom)

        # ==========================================
        # 3. 计算真实的相机位姿并发布
        # ==========================================
        # 利用三角函数，把相机的坐标顺着机头方向往前推 0.5 米
        CAMERA_OFFSET = 0.5
        cam_x_enu = x_enu + CAMERA_OFFSET * math.cos(yaw_enu)
        cam_y_enu = y_enu + CAMERA_OFFSET * math.sin(yaw_enu)
        cam_z_enu = z_enu

        cam_pose = PoseStamped()
        cam_pose.header.stamp = msg.header.stamp
        cam_pose.header.frame_id = "world_enu"

        cam_pose.pose.position.x = cam_x_enu
        cam_pose.pose.position.y = cam_y_enu
        cam_pose.pose.position.z = cam_z_enu

        # 固定的相机下洗旋转四元数 (r=-pi/2, p=0, y=-pi/2)
        x2, y2, z2, w2 = 0.5, -0.5, -0.5, 0.5 

        # 四元数乘法 q_enu * q_rot
        cam_pose.pose.orientation.x = qw_enu*x2 + qx_enu*w2 + qy_enu*z2 - qz_enu*y2
        cam_pose.pose.orientation.y = qw_enu*y2 - qx_enu*z2 + qy_enu*w2 + qz_enu*x2
        cam_pose.pose.orientation.z = qw_enu*z2 + qx_enu*y2 - qy_enu*x2 + qz_enu*w2
        cam_pose.pose.orientation.w = qw_enu*w2 - qx_enu*x2 - qy_enu*y2 - qz_enu*z2

        self.cam_pose_pub.publish(cam_pose)

    def _on_pos_cmd(self, msg):
        self.latest_pos_cmd = msg
        self.latest_pos_cmd_stamp = time.time()
        self.pos_cmd_rx_count += 1

        if POS_CMD_DEBUG_LOG:
            rospy.logdebug_throttle(
                POS_CMD_DEBUG_LOG_INTERVAL_S,
                "[pos_cmd rx #%d] p=(%.3f, %.3f, %.3f) v=(%.3f, %.3f, %.3f) yaw=%.3f yaw_dot=%.3f",
                self.pos_cmd_rx_count,
                msg.position.x,
                msg.position.y,
                msg.position.z,
                msg.velocity.x,
                msg.velocity.y,
                msg.velocity.z,
                msg.yaw,
                msg.yaw_dot,
            )

    def get_latest_odom(self):
        return self.latest_odom

    def get_latest_pos_cmd(self, timeout_s=0.5):
        if self.latest_pos_cmd and (time.time() - self.latest_pos_cmd_stamp) < timeout_s:
            return self.latest_pos_cmd
        return None

    def get_pos_cmd_debug_status(self):
        """返回 pos_cmd 接收诊断信息。"""
        age = None
        if self.latest_pos_cmd_stamp > 0.0:
            age = time.time() - self.latest_pos_cmd_stamp
        return {
            "rx_count": self.pos_cmd_rx_count,
            "last_age_s": age,
        }

    def publish_velocity_world(self, vx, vy, vz=0.0, yaw_rate=0.0):
        """发布世界系速度指令。"""
        msg = self._vel_msg_cls()

        # airsim_ros_pkgs/VelCmd: 包含 twist 字段
        if hasattr(msg, "twist"):
            msg.twist.linear.x = float(vx)
            msg.twist.linear.y = float(vy)
            msg.twist.linear.z = float(vz)
            msg.twist.angular.z = float(yaw_rate)

            # 常见 VelCmd 字段，存在时赋值，不存在则跳过
            if hasattr(msg, "drivetrain"):
                msg.drivetrain = 0
            if hasattr(msg, "yaw_mode") and hasattr(msg.yaw_mode, "is_rate"):
                msg.yaw_mode.is_rate = True
                msg.yaw_mode.yaw_or_rate = float(yaw_rate)

        # geometry_msgs/Twist
        else:
            msg.linear.x = float(vx)
            msg.linear.y = float(vy)
            msg.linear.z = float(vz)
            msg.angular.z = float(yaw_rate)

        self.vel_pub.publish(msg)

    def publish_hover(self):
        self.publish_velocity_world(0.0, 0.0, 0.0, 0.0)

    def reset_debug_visualization(self):
        """清除所有 RViz Marker。"""
        if _HAS_MARKER_MSGS and self.marker_pub is not None:
            marker_array = MarkerArray()
            clear_marker = Marker()
            clear_marker.header.stamp = rospy.Time.now()
            clear_marker.header.frame_id = self.debug_frame
            clear_marker.action = Marker.DELETEALL
            marker_array.markers.append(clear_marker)
            self.marker_pub.publish(marker_array)

    def publish_scene_objects(self, objects_dict, object_radius_m=0.5):
        """发布场景物体（障碍物）到 RViz 进行可视化。

        将障碍物坐标以 CYLINDER marker 的形式发布到 RViz，
        每个物体会同时附带一个名称标签（TEXT_VIEW_FACING）。

        Args:
            objects_dict: 字典，格式 {"Obstacle1": [x, y, z], "Obstacle2": [x, y, z], ...}
                          坐标为 ENU 世界坐标系（与本模块的 world_enu 一致）。
            object_radius_m: 物体可视化半径（米），默认 0.5m。
        """
        if not (_HAS_MARKER_MSGS and self.marker_pub is not None):
            rospy.logwarn_throttle(10.0, "Marker publisher not available, skip scene objects visualization.")
            return

        if not objects_dict:
            return

        marker_array = MarkerArray()
        stamp = rospy.Time.now()
        marker_id = 100  # 从 100 开始，避免与其他 debug marker id 冲突

        for obj_name, pos_xyz in objects_dict.items():
            # 物体位置 marker（CYLINDER）
            obj_marker = Marker()
            obj_marker.header.stamp = stamp
            obj_marker.header.frame_id = self.debug_frame
            obj_marker.ns = "scene_objects"
            obj_marker.id = marker_id
            obj_marker.type = Marker.CYLINDER
            obj_marker.action = Marker.ADD
            obj_marker.pose.position.x = float(pos_xyz[0])
            obj_marker.pose.position.y = float(pos_xyz[1])
            obj_marker.pose.position.z = float(pos_xyz[2]) if len(pos_xyz) > 2 else 0.0
            obj_marker.pose.orientation.w = 1.0
            obj_marker.scale.x = float(2.0 * max(0.05, object_radius_m))
            obj_marker.scale.y = float(2.0 * max(0.05, object_radius_m))
            obj_marker.scale.z = 2.0  # 高度 2 米，可根据实际障碍物高度调整
            obj_marker.color = ColorRGBA(0.2, 0.8, 0.2, 0.8)  # 绿色半透明
            marker_array.markers.append(obj_marker)

            # 物体名字标签
            text_marker = Marker()
            text_marker.header.stamp = stamp
            text_marker.header.frame_id = self.debug_frame
            text_marker.ns = "scene_objects_text"
            text_marker.id = marker_id + 1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = float(pos_xyz[0])
            text_marker.pose.position.y = float(pos_xyz[1])
            text_marker.pose.position.z = float(pos_xyz[2]) + 0.6 if len(pos_xyz) > 2 else 0.6
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.3  # 文字高度
            text_marker.text = str(obj_name)
            text_marker.color = ColorRGBA(0.2, 0.8, 0.2, 0.95)
            marker_array.markers.append(text_marker)

            marker_id += 1

        self.marker_pub.publish(marker_array)

    def shutdown(self):
        self.publish_hover()
        self.odom_sub.unregister()
        if hasattr(self, "pos_cmd_sub"):
            self.pos_cmd_sub.unregister()
        self.vel_pub.unregister()
