#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

# 1) source AirSim ROS 工作空间
if [ -f "/home/user/AirSim/ros/devel/setup.bash" ]; then
	# shellcheck disable=SC1091
	source /home/user/AirSim/ros/devel/setup.bash
else
	echo "[WARN] AirSim ROS setup.bash not found"
fi

# 2) source YOPO/Ego 工作空间（若存在）
if [ -f "/home/user/ego-planner/devel/setup.bash" ]; then
	# shellcheck disable=SC1091
	source /home/user/ego-planner/devel/setup.bash
fi

# 3) 启动桥接主程序
exec "$PYTHON_BIN" "$SCRIPT_DIR/main_airsim_ros.py"