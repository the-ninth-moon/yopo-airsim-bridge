import math
import threading
import time

import airsim
import matplotlib.pyplot as plt
import numpy as np

from gui import DroneGUI
from pid_controller import PIDController
from ros_topic_manager import RosTopicManager

# --- 配置参数 ---
HEIGHT = -3.0
MAX_VELOCITY = 10.0
MAX_Z_VELOCITY = 1.2
MAIN_LOOP_DT = 0.02
GUI_UPDATE_DT = 0.10
TAKEOFF_SPEED = 1.0
LAND_TIMEOUT = 5.0
EGO_USE_ENU_Z = True
TAKEOFF_ON_START = False
POS_CMD_TIMEOUT_S = 0.25
OBSTACLE_PUBLISH_DT = 2.0  # 障碍物可视化刷新间隔（秒），0.5Hz

# 速度平滑与斜率限制，抑制物理引擎中的“卡顿”和姿态点头
CMD_LPF_ALPHA_XY = 0.35
CMD_LPF_ALPHA_Z = 0.25
CMD_LPF_ALPHA_YAW = 0.30
MAX_ACC_XY = 4.0
MAX_ACC_Z = 2.0
MAX_ACC_YAW = 2.0
CMD_DEADBAND_XY = 0.02
CMD_DEADBAND_Z = 0.03
CMD_DEADBAND_YAW = 0.03


def _normalize_angle(angle_rad):
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


def _arm_and_enable_api(client):
    client.enableApiControl(True)
    client.armDisarm(True)


def _rate_limit(prev_value, target_value, max_rate, dt_s):
    if dt_s <= 0.0:
        return target_value
    max_delta = max_rate * dt_s
    delta = target_value - prev_value
    if delta > max_delta:
        return prev_value + max_delta
    if delta < -max_delta:
        return prev_value - max_delta
    return target_value


def _smooth_velocity_command(prev_cmd, raw_cmd, dt_s):
    prev_vx, prev_vy, prev_vz, prev_yaw = prev_cmd
    raw_vx, raw_vy, raw_vz, raw_yaw = raw_cmd

    # 先做低通，再做斜率限制，减少控制突变导致的机体抖动
    lpf_vx = prev_vx + CMD_LPF_ALPHA_XY * (raw_vx - prev_vx)
    lpf_vy = prev_vy + CMD_LPF_ALPHA_XY * (raw_vy - prev_vy)
    lpf_vz = prev_vz + CMD_LPF_ALPHA_Z * (raw_vz - prev_vz)
    lpf_yaw = prev_yaw + CMD_LPF_ALPHA_YAW * (raw_yaw - prev_yaw)

    vx = _rate_limit(prev_vx, lpf_vx, MAX_ACC_XY, dt_s)
    vy = _rate_limit(prev_vy, lpf_vy, MAX_ACC_XY, dt_s)
    vz = _rate_limit(prev_vz, lpf_vz, MAX_ACC_Z, dt_s)
    yaw_rate = _rate_limit(prev_yaw, lpf_yaw, MAX_ACC_YAW, dt_s)

    # 微小命令死区，抑制悬停时细碎抖动
    if abs(vx) < CMD_DEADBAND_XY:
        vx = 0.0
    if abs(vy) < CMD_DEADBAND_XY:
        vy = 0.0
    if abs(vz) < CMD_DEADBAND_Z:
        vz = 0.0
    if abs(yaw_rate) < CMD_DEADBAND_YAW:
        yaw_rate = 0.0
    return vx, vy, vz, yaw_rate


def _apply_gui_command(command, client, ros_io, pid_x, pid_y, gui, pid_z=None, pid_yaw=None):
    if command is None:
        return

    try:
        if command == "takeoff":
            print("[SDK] takeoff command")
            _arm_and_enable_api(client)
            ros_io.publish_hover()
            client.takeoffAsync().join()
            client.moveToZAsync(HEIGHT, TAKEOFF_SPEED).join()
            gui.set_control_status("takeoff-done")

        elif command == "land":
            print("[SDK] land command")
            ros_io.publish_hover()
            client.landAsync(timeout_sec=LAND_TIMEOUT).join()
            client.armDisarm(False)
            gui.set_control_status("land-done")

        elif command == "reset":
            print("[SDK] reset command")
            ros_io.publish_hover()
            client.reset()
            _arm_and_enable_api(client)

            # reset 后清空 PID 状态，防止使用旧积分项
            pid_x.reset()
            pid_y.reset()
            if pid_z is not None:
                pid_z.reset()
            if pid_yaw is not None:
                pid_yaw.reset()
            gui.set_control_status("reset-done")

    except Exception as exc:
        gui.set_control_status(f"{command}-failed")
        print(f"[GUI Command Error] {command}: {exc}")


def _compute_ego_velocity_command(curr_x, curr_y, curr_z, curr_yaw, pos_cmd, pid_x, pid_y, pid_z, pid_yaw, dt_s):
    # 1. 提取 Ego-Planner 的 ENU 指令 (X=东, Y=北, Z=天)
    tx_enu = pos_cmd.position.x
    ty_enu = pos_cmd.position.y
    tz_enu = pos_cmd.position.z
    vx_enu = pos_cmd.velocity.x
    vy_enu = pos_cmd.velocity.y
    vz_enu = pos_cmd.velocity.z
    target_yaw_enu = pos_cmd.yaw
    yaw_dot_enu = pos_cmd.yaw_dot

    # 2. 坐标系映射：ENU -> NED (X=北, Y=东, Z=下)
    tx_ned = ty_enu       # ENU的北(Y) -> NED的北(X)
    ty_ned = tx_enu       # ENU的东(X) -> NED的东(Y)
    tz_ned = -tz_enu if EGO_USE_ENU_Z else curr_z

    vx_ff_ned = vy_enu
    vy_ff_ned = vx_enu
    vz_ff_ned = -vz_enu if EGO_USE_ENU_Z else 0.0

    # Yaw角映射：ENU的0度是东(逆时针)，NED的0度是北(顺时针)，相差90度(pi/2)
    target_yaw_ned = math.pi / 2.0 - target_yaw_enu
    yaw_dot_ff_ned = -yaw_dot_enu

    # 加速度前馈：ENU -> NED，预补偿下一帧的速度变化量
    ax_enu = pos_cmd.acceleration.x
    ay_enu = pos_cmd.acceleration.y
    az_enu = pos_cmd.acceleration.z
    ax_ff_ned = ay_enu      # ENU 北(Y) -> NED 北(X)
    ay_ff_ned = ax_enu      # ENU 东(X) -> NED 东(Y)
    az_ff_ned = -az_enu     # ENU 上(Z) -> NED 下(Z)

    # 3. 在完全统一的 NED 坐标系下计算误差
    err_x = tx_ned - curr_x
    err_y = ty_ned - curr_y
    err_z = tz_ned - curr_z
    yaw_diff = _normalize_angle(target_yaw_ned - curr_yaw)

    # 放开 XY 的限幅限制，让 Ego-Planner 完全接管速度
    pid_x.output_limits = (-MAX_VELOCITY, MAX_VELOCITY)
    pid_y.output_limits = (-MAX_VELOCITY, MAX_VELOCITY)

    # 4. 计算最终指令 (速度前馈 + 加速度前馈 + PID 位置补偿)
    vx_cmd = vx_ff_ned + ax_ff_ned * dt_s + pid_x.compute(err_x, dt=dt_s)
    vy_cmd = vy_ff_ned + ay_ff_ned * dt_s + pid_y.compute(err_y, dt=dt_s)
    vz_cmd = vz_ff_ned + az_ff_ned * dt_s + pid_z.compute(err_z, dt=dt_s)
    yaw_rate_cmd = yaw_dot_ff_ned + pid_yaw.compute(yaw_diff, dt=dt_s)

    # 5. 安全限幅
    vx_cmd = float(np.clip(vx_cmd, -MAX_VELOCITY, MAX_VELOCITY))
    vy_cmd = float(np.clip(vy_cmd, -MAX_VELOCITY, MAX_VELOCITY))
    vz_cmd = float(np.clip(vz_cmd, -MAX_Z_VELOCITY, MAX_Z_VELOCITY))
    yaw_rate_cmd = float(np.clip(yaw_rate_cmd, -1.5, 1.5))

    return vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd


def _compute_altitude_hold_velocity(curr_z, target_z):
    z_error = target_z - curr_z
    vz_cmd = float(np.clip(1.2 * z_error, -MAX_Z_VELOCITY, MAX_Z_VELOCITY))
    # 进入小误差带时减小抖动
    if abs(z_error) < 0.05:
        return 0.0
    return vz_cmd


def _obstacle_fetch_thread(shared_state, interval_s, stop_event):
    """后台线程：定时从 AirSim 获取场景物体坐标，使用独立 RPC 连接避免与主线程冲突。"""
    # 创建独立的 AirSim 客户端，不和主线程共享 TCP 连接
    client = airsim.MultirotorClient()
    try:
        client.confirmConnection()
    except Exception:
        print("[ObstacleThread] Failed to connect, retrying...")
        client = None

    while not stop_event.is_set():
        if client is not None:
            try:
                scene_names = client.simListSceneObjects()
                if scene_names:
                    obstacles = {}
                    for name in scene_names:
                        if "drone" in name.lower():
                            continue
                        try:
                            pose = client.simGetObjectPose(name)
                            pos = pose.position
                            obstacles[name] = [pos.x_val, pos.y_val, pos.z_val]
                        except Exception:
                            pass
                    with shared_state["lock"]:
                        shared_state["obstacles"] = obstacles
            except Exception as exc:
                print(f"[ObstacleThread] Error: {exc}")
                # 连接断开时尝试重连
                try:
                    client = airsim.MultirotorClient()
                    client.confirmConnection()
                except Exception:
                    client = None
        else:
            # 首次连接失败后的重试
            try:
                client = airsim.MultirotorClient()
                client.confirmConnection()
            except Exception:
                client = None
        stop_event.wait(interval_s)


def control_loop():
    # AirSim SDK 仅用于生命周期命令；主控制由 ROS 话题链路驱动。
    client = airsim.MultirotorClient()
    client.confirmConnection()
    if TAKEOFF_ON_START:
        _arm_and_enable_api(client)
        print("Taking off...")
        client.takeoffAsync().join()
        client.moveToZAsync(HEIGHT, TAKEOFF_SPEED).join()

    print("Ready to fly with ROS velocity topic control.")

    gui = DroneGUI()
    ros_io = RosTopicManager(drone_name="drone_1")

    # 后台线程：定时查询 AirSim 场景物体，避免阻塞主控制循环
    obstacle_shared = {"obstacles": {}, "lock": threading.Lock()}
    obstacle_stop_event = threading.Event()
    obstacle_thread = threading.Thread(
        target=_obstacle_fetch_thread,
        args=(obstacle_shared, OBSTACLE_PUBLISH_DT, obstacle_stop_event),
        daemon=True,
    )
    obstacle_thread.start()

    pid_x = PIDController(kp=1.1, ki=0.0, kd=0.15, output_limits=(-MAX_VELOCITY, MAX_VELOCITY))
    pid_y = PIDController(kp=1.1, ki=0.0, kd=0.15, output_limits=(-MAX_VELOCITY, MAX_VELOCITY))
    pid_z = PIDController(kp=0.7, ki=0.0, kd=0.0, output_limits=(-MAX_Z_VELOCITY, MAX_Z_VELOCITY))
    pid_yaw = PIDController(kp=1.0, ki=0.0, kd=0.05, output_limits=(-1.5, 1.5))

    plt.ion()
    was_ego_mode = False
    hover_z_setpoint = HEIGHT
    last_pos_cmd_stale_log_time = 0.0
    last_cmd = (0.0, 0.0, 0.0, 0.0)
    last_gui_update_time = 0.0
    last_obstacle_publish_time = 0.0
    prev_loop_time = time.monotonic()
    mode_text_is_ego = False
    mode_text_is_hold = False

    try:
        while True:
            loop_now = time.monotonic()
            loop_dt = max(0.001, min(0.1, loop_now - prev_loop_time))
            prev_loop_time = loop_now

            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            curr_x, curr_y, curr_z = pos.x_val, pos.y_val, pos.z_val

            vel = state.kinematics_estimated.linear_velocity
            curr_speed = math.sqrt(vel.x_val ** 2 + vel.y_val ** 2)

            orientation = state.kinematics_estimated.orientation
            curr_yaw = airsim.to_eularian_angles(orientation)[2]

            # GUI 生命周期控制命令（仍走 AirSim Python SDK）
            _apply_gui_command(
                gui.pop_control_command(),
                client,
                ros_io,
                pid_x,
                pid_y,
                gui,
                pid_z,
                pid_yaw,
            )

            pos_cmd = ros_io.get_latest_pos_cmd(timeout_s=POS_CMD_TIMEOUT_S)

            if pos_cmd is not None:
                if not was_ego_mode:
                    print("Switching to Ego-Planner Auto Mode!")
                    pid_x.reset()
                    pid_y.reset()
                    pid_z.reset()
                    pid_yaw.reset()
                    was_ego_mode = True
                    mode_text_is_ego = False

                # 持续刷新悬停高度基准，Ego 停止后可在当前高度平稳接管。
                hover_z_setpoint = curr_z
                mode_text_is_hold = False

                if not mode_text_is_ego:
                    gui.set_mode_text('Ego-Planner Auto [Override]')
                    mode_text_is_ego = True
                vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd = _compute_ego_velocity_command(
                    curr_x,
                    curr_y,
                    curr_z,
                    curr_yaw,
                    pos_cmd,
                    pid_x,
                    pid_y,
                    pid_z,
                    pid_yaw,
                    loop_dt,
                )
                smoothed_cmd = _smooth_velocity_command(last_cmd, (vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd), loop_dt)
                last_cmd = smoothed_cmd
                ros_io.publish_velocity_world(*smoothed_cmd)

                if (loop_now - last_gui_update_time) >= GUI_UPDATE_DT:
                    gui.update_plot(curr_speed)
                    last_gui_update_time = loop_now
            else:
                pos_cmd_dbg = ros_io.get_pos_cmd_debug_status()
                # 若确实收到了 pos_cmd 但当前未接管，输出原因（通常是消息中断超过 timeout）。
                if pos_cmd_dbg["rx_count"] > 0 and pos_cmd_dbg["last_age_s"] is not None:
                    now = time.time()
                    if now - last_pos_cmd_stale_log_time > 1.0:
                        print(
                            "[EGO Debug] pos_cmd seen but inactive: "
                            f"last_age={pos_cmd_dbg['last_age_s']:.3f}s, "
                            f"timeout={POS_CMD_TIMEOUT_S:.3f}s, "
                            f"rx_count={pos_cmd_dbg['rx_count']}"
                        )
                        last_pos_cmd_stale_log_time = now

                if was_ego_mode:
                    print("Ego-Planner timeout/stopped. Switching to Hover Hold.")
                    pid_x.reset()
                    pid_y.reset()
                    pid_z.reset()
                    pid_yaw.reset()
                    was_ego_mode = False
                    hover_z_setpoint = curr_z
                    last_cmd = (0.0, 0.0, 0.0, 0.0)
                    mode_text_is_ego = False
                    mode_text_is_hold = False

                # 无 YOPO 指令时进入悬停保持，不再提供 GUI 路径控制。
                pid_x.reset()
                pid_y.reset()
                vz_hold = _compute_altitude_hold_velocity(curr_z, hover_z_setpoint)
                smoothed_cmd = _smooth_velocity_command(last_cmd, (0.0, 0.0, vz_hold, 0.0), loop_dt)
                last_cmd = smoothed_cmd
                ros_io.publish_velocity_world(*smoothed_cmd)

                if not mode_text_is_hold:
                    gui.set_mode_text('Hover Hold (No YOPO pos_cmd)')
                    mode_text_is_hold = True

                if (loop_now - last_gui_update_time) >= GUI_UPDATE_DT:
                    gui.update_plot(curr_speed)
                    last_gui_update_time = loop_now

            # 保留 ROS 订阅入口供后续扩展
            _ = ros_io.get_latest_odom()

            # --- 障碍物可视化：从后台线程读取坐标并发布 MarkerArray ---
            if (loop_now - last_obstacle_publish_time) >= OBSTACLE_PUBLISH_DT:
                last_obstacle_publish_time = loop_now
                with obstacle_shared["lock"]:
                    obstacles = dict(obstacle_shared["obstacles"])
                if obstacles:
                    ros_io.publish_scene_objects(obstacles)

            plt.pause(0.001)

            elapsed = time.monotonic() - loop_now
            sleep_s = MAIN_LOOP_DT - elapsed
            if sleep_s > 0.0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        obstacle_stop_event.set()
        ros_io.publish_hover()
        ros_io.shutdown()
        # 不在退出时强制接管/降落，避免干扰 Ego-Planner 与 airsim_node 的运行态。


if __name__ == "__main__":
    control_loop()
