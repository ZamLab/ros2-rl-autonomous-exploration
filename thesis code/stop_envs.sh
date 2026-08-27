#!/bin/bash
# Fault-recovery script: kill all ROS 2 / Gazebo processes and relaunch the
# environments. Invoked (via try_global_reset) when an environment becomes
# unhealthy, e.g. a simulation stalls past a watchdog timeout.

echo "[$(date)] Stopping all ROS2 and Gazebo processes..."

TERMINALS=$(pgrep -a gnome-terminal-server)
echo "GNOME Terminals running:"
echo "$TERMINALS"

# Kill ROS 2 and Gazebo.
pkill -f ros2 || true
pkill -f gazebo || true
pkill -f gzserver || true

echo "[$(date)] All processes stopped."

# Wait to ensure everything is down.
sleep 10

echo "[$(date)] Relaunching environments in a new terminal..."
nohup gnome-terminal -- bash -c "sleep 2; ./run_multi_envs.sh; exec bash" > relaunch.log 2>&1 &

echo "Going to sleep. Bye"

# Close remaining GNOME terminal windows opened by the launcher.
pkill -f gnome-terminal-server
