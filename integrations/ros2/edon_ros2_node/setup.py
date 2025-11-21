"""Setup script for EDON ROS2 Node."""

from setuptools import setup, find_packages

setup(
    name='edon_ros2_node',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'rclpy',
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'lightgbm',
    ],
    entry_points={
        'console_scripts': [
            'edon_ros2_node=edon_ros2_node.edon_node:main',
        ],
    },
)

