from setuptools import setup
import os
from glob import glob

package_name = "humanoid_edon_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "msg"), glob("msg/*.msg")),
    ],
    install_requires=["setuptools", "requests"],
    zip_safe=True,
    maintainer="EDON Team",
    maintainer_email="edon@example.com",
    description="ROS 2 bridge for EDON CAV engine state to humanoid robot actuators",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "edon_humanoid_bridge = humanoid_edon_bridge.edon_humanoid_bridge:main",
        ],
    },
)

