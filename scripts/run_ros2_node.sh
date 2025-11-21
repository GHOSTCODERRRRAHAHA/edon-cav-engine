#!/bin/bash
# Run EDON ROS2 Node

set -e

# Check if ROS2 is installed
if [ -z "$ROS_DISTRO" ]; then
    echo "ERROR: ROS2 environment not sourced. Please run:"
    echo "  source /opt/ros/foxy/setup.bash"
    echo "  # or for your ROS2 distribution"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Add project to Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run the node
cd "$PROJECT_ROOT"
python3 integrations/ros2/edon_ros2_node/edon_node.py

