#!/usr/bin/env python3
"""
EDON ROS2 Node - Foxy-compatible

Subscribes to:
  - /edon/sensors/physiology (sensor_msgs/msg/PointCloud2 or custom msg)
  - /edon/sensors/environment (sensor_msgs/msg/Temperature or custom msg)

Publishes:
  - /edon/state (std_msgs/msg/String: restorative | balanced | focus | overload)
  - /edon/controls (custom msg with speed, torque, safety scales)
"""

import sys
import os
from pathlib import Path

# Add parent directories to path to import engine
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from sensor_msgs.msg import PointCloud2, Temperature
import numpy as np
from typing import Dict, Optional
import time

# Import EDON engine directly
from app.engine import CAVEngine, WINDOW_LEN


class PhysiologySubscriber(Node):
    """Handles physiology sensor data (EDA, BVP, ACC, TEMP)."""
    
    def __init__(self, engine: CAVEngine):
        super().__init__('edon_physiology_subscriber')
        self.engine = engine
        self.buffer: Dict[str, list] = {
            'EDA': [],
            'TEMP': [],
            'BVP': [],
            'ACC_x': [],
            'ACC_y': [],
            'ACC_z': [],
        }
        self.subscription = self.create_subscription(
            PointCloud2,  # Using PointCloud2 as generic container
            '/edon/sensors/physiology',
            self.physiology_callback,
            10
        )
        self.get_logger().info('Physiology subscriber initialized')
    
    def physiology_callback(self, msg):
        """Process incoming physiology data."""
        # In real implementation, extract data from PointCloud2 or custom message
        # For now, we'll use a simple buffer approach
        # This is a placeholder - actual implementation would parse the message
        self.get_logger().debug(f'Received physiology data: {len(msg.data)} bytes')
        
        # TODO: Parse PointCloud2 or use custom message type
        # For now, assume data comes in as structured format
        pass


class EnvironmentSubscriber(Node):
    """Handles environment sensor data (temp_c, humidity, aqi)."""
    
    def __init__(self):
        super().__init__('edon_environment_subscriber')
        self.temp_c: Optional[float] = None
        self.humidity: Optional[float] = None
        self.aqi: Optional[int] = None
        self.subscription = self.create_subscription(
            Temperature,  # Using Temperature as placeholder
            '/edon/sensors/environment',
            self.environment_callback,
            10
        )
        self.get_logger().info('Environment subscriber initialized')
    
    def environment_callback(self, msg):
        """Process incoming environment data."""
        # In real implementation, extract temp, humidity, aqi from message
        # This is a placeholder
        self.get_logger().debug(f'Received environment data: temp={msg.temperature}')
        # TODO: Parse custom message with temp_c, humidity, aqi


class EdonStatePublisher(Node):
    """Publishes EDON state and control signals."""
    
    def __init__(self):
        super().__init__('edon_state_publisher')
        self.state_pub = self.create_publisher(String, '/edon/state', 10)
        self.controls_pub = self.create_publisher(Float32, '/edon/controls', 10)
        self.get_logger().info('State publisher initialized')
    
    def publish_state(self, state: str):
        """Publish current EDON state."""
        msg = String()
        msg.data = state
        self.state_pub.publish(msg)
        self.get_logger().info(f'Published state: {state}')
    
    def publish_controls(self, speed: float, torque: float, safety: float):
        """Publish control scales."""
        # Using Float32 array or custom message
        # For simplicity, publishing as separate topics or custom message
        # TODO: Create custom Controls message type
        self.get_logger().debug(f'Controls: speed={speed}, torque={torque}, safety={safety}')


class EdonRos2Node(Node):
    """
    Main EDON ROS2 Node.
    
    Integrates physiology and environment sensors to compute CAV state
    and publish control signals for robot behavior adaptation.
    """
    
    def __init__(self):
        super().__init__('edon_node')
        
        # Initialize CAV engine
        self.get_logger().info('Initializing CAV engine...')
        self.engine = CAVEngine()
        self.get_logger().info('CAV engine initialized')
        
        # Initialize subscribers and publishers
        self.physiology_sub = PhysiologySubscriber(self.engine)
        self.env_sub = EnvironmentSubscriber()
        self.state_pub = EdonStatePublisher()
        
        # Window buffer for accumulating sensor data
        self.window_buffer: Dict[str, list] = {
            'EDA': [],
            'TEMP': [],
            'BVP': [],
            'ACC_x': [],
            'ACC_y': [],
            'ACC_z': [],
        }
        
        # Timer for periodic CAV computation (every 60 seconds)
        self.timer = self.create_timer(60.0, self.compute_cav_callback)
        
        self.get_logger().info('EDON ROS2 Node started')
    
    def add_sensor_sample(self, signal_type: str, value: float):
        """Add a single sensor sample to the buffer."""
        if signal_type in self.window_buffer:
            self.window_buffer[signal_type].append(value)
            
            # Keep buffer at WINDOW_LEN
            if len(self.window_buffer[signal_type]) > WINDOW_LEN:
                self.window_buffer[signal_type].pop(0)
    
    def compute_cav_callback(self):
        """Periodically compute CAV from accumulated window."""
        # Check if we have enough data
        if not all(len(self.window_buffer[k]) == WINDOW_LEN for k in self.window_buffer):
            self.get_logger().warn('Insufficient data for CAV computation')
            return
        
        # Build window dict
        window = {k: self.window_buffer[k].copy() for k in self.window_buffer}
        
        # Get environment data
        temp_c = self.env_sub.temp_c
        humidity = self.env_sub.humidity
        aqi = self.env_sub.aqi
        local_hour = time.localtime().tm_hour
        
        # Compute CAV
        try:
            cav_raw, cav_smooth, state, parts = self.engine.cav_from_window(
                window=window,
                temp_c=temp_c,
                humidity=humidity,
                aqi=aqi,
                local_hour=local_hour,
            )
            
            # Publish state
            self.state_pub.publish_state(state)
            
            # Compute control scales based on state
            speed, torque, safety = self.compute_controls(state, parts)
            self.state_pub.publish_controls(speed, torque, safety)
            
            self.get_logger().info(
                f'CAV computed: state={state}, cav_smooth={cav_smooth}, '
                f'p_stress={parts.get("p_stress", 0.0):.3f}'
            )
            
        except Exception as e:
            self.get_logger().error(f'CAV computation failed: {e}', exc_info=True)
    
    def compute_controls(self, state: str, parts: Dict[str, float]) -> tuple:
        """
        Compute robot control scales based on EDON state.
        
        Returns:
            (speed, torque, safety) as float scales [0.0, 1.0]
        """
        p_stress = parts.get('p_stress', 0.0)
        
        if state == 'overload':
            # Reduce speed and torque, increase safety margins
            speed = 0.3
            torque = 0.3
            safety = 1.0
        elif state == 'focus':
            # Moderate speed, full torque, normal safety
            speed = 0.7
            torque = 1.0
            safety = 0.8
        elif state == 'balanced':
            # Normal operation
            speed = 1.0
            torque = 1.0
            safety = 0.7
        else:  # restorative
            # Can operate at full capacity
            speed = 1.0
            torque = 1.0
            safety = 0.5
        
        return (speed, torque, safety)


def main(args=None):
    """Main entry point for ROS2 node."""
    rclpy.init(args=args)
    
    node = EdonRos2Node()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

