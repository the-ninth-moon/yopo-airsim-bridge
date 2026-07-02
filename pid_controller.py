import time

class PIDController:
    def __init__(self, kp, ki, kd, output_limits=None, derivative_tau=0.05):
        """
        初始化 PID 控制器
        :param kp: 比例系数
        :param ki: 积分系数
        :param kd: 微分系数
        :param output_limits: 输出限制 (min, max)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.derivative_tau = max(0.0, derivative_tau)
        
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None
        self.derivative_state = 0.0

    def compute(self, error, dt=None):
        current_time = time.time()
        
        if self.last_time is None:
            self.last_time = current_time
            # 第一次调用没有 dt，返回纯 P 控制或 0
            return self.kp * error

        if dt is None:
            dt = current_time - self.last_time

        self.last_time = current_time
        
        # 防止 dt 过大或过小导致数值问题
        if dt <= 0.0:
            dt = 1e-3
        dt = min(dt, 0.2)
        
        # 积分项
        self.integral += error * dt

        # 微分项
        derivative_raw = (error - self.prev_error) / dt
        if self.derivative_tau > 0.0:
            alpha = dt / (self.derivative_tau + dt)
            self.derivative_state += alpha * (derivative_raw - self.derivative_state)
            derivative = self.derivative_state
        else:
            derivative = derivative_raw
        self.prev_error = error

        # 计算输出
        output_unsat = self.kp * error + self.ki * self.integral + self.kd * derivative

        # 输出限幅
        output = output_unsat
        if self.output_limits:
            low, high = self.output_limits
            if output_unsat > high:
                output = high
                # 抗积分饱和：只在误差会推动更饱和时回退积分
                if self.ki > 0.0 and error > 0.0:
                    self.integral -= error * dt
            elif output_unsat < low:
                output = low
                if self.ki > 0.0 and error < 0.0:
                    self.integral -= error * dt
            
        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None
        self.derivative_state = 0.0
