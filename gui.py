import matplotlib.pyplot as plt
from collections import deque
from matplotlib.widgets import Button

class DroneGUI:
    """轻量级飞行状态面板。

    设计目标：
    - 仅作为状态展示与生命周期按钮入口。
    - 不承担路径可视化与手动打点控制职责。
    """

    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.manager.set_window_title('Drone Control Panel')
        self.fig.subplots_adjust(bottom=0.16)

        # 速度平滑
        self.speed_history = deque(maxlen=20)

        # GUI 控制命令队列（由主循环消费）
        self.control_commands = deque(maxlen=20)

        # 面板区域（仅保留文本状态，不显示轨迹/无人机位置）
        self.ax.set_title('YOPO / ROS Bridge Status')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_frame_on(False)

        # 添加速度文本显示
        self.text_speed = self.ax.text(
            0.02,
            0.95,
            'Speed: 0.00 m/s',
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
        )
        self.text_mode = self.ax.text(
            0.02,
            0.89,
            'Mode: Waiting for YOPO pos_cmd',
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
        )
        self.text_ctrl = self.ax.text(0.02, 0.83, 'Ctrl: Idle', transform=self.ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # 控制按钮
        self.btn_takeoff = Button(self.fig.add_axes([0.12, 0.04, 0.2, 0.07]), 'Takeoff')
        self.btn_land = Button(self.fig.add_axes([0.4, 0.04, 0.2, 0.07]), 'Land')
        self.btn_reset = Button(self.fig.add_axes([0.68, 0.04, 0.2, 0.07]), 'Reset')
        self.btn_takeoff.on_clicked(self.on_takeoff)
        self.btn_land.on_clicked(self.on_land)
        self.btn_reset.on_clicked(self.on_reset)

    def on_takeoff(self, _event):
        self.enqueue_control_command('takeoff')

    def on_land(self, _event):
        self.enqueue_control_command('land')

    def on_reset(self, _event):
        self.enqueue_control_command('reset')

    def enqueue_control_command(self, cmd):
        self.control_commands.append(cmd)
        self.text_ctrl.set_text(f'Ctrl: {cmd}')

    def pop_control_command(self):
        if self.control_commands:
            return self.control_commands.popleft()
        return None

    def set_control_status(self, status):
        self.text_ctrl.set_text(f'Ctrl: {status}')

    def set_mode_text(self, mode_text):
        self.text_mode.set_text(f'Mode: {mode_text}')

    def update_plot(self, velocity=0.0):
        """
        更新状态面板。
        :param velocity: 当前速度 (m/s)
        """
        # 更新速度文本 (使用滑动平均)
        self.speed_history.append(velocity)
        if self.speed_history:
            avg_speed = sum(self.speed_history) / len(self.speed_history)
        else:
            avg_speed = 0.0
        self.text_speed.set_text(f'Speed: {avg_speed:.2f} m/s')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
