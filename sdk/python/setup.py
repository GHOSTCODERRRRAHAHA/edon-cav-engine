"""Setup script for EDON Python SDK."""

from setuptools import setup, find_packages
from pathlib import Path

# Read version from __init__.py
version = "2.0.0"
init_file = Path(__file__).parent / "edon" / "__init__.py"
if init_file.exists():
    with open(init_file) as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip('"').strip("'")
                break

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="edon",
    version=version,
    description="Python SDK for EDON CAV Engine (adaptive state engine for physical AI)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="EDON Team",
    author_email="dev@edon.ai",
    url="https://github.com/edon-ai/edon-cav-engine",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "requests>=2.31.0",
        "urllib3>=2.0.0",
    ],
    extras_require={
        "grpc": [
            "grpcio>=1.60.0",
            "grpcio-tools>=1.60.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords=["edon", "cav", "physiological", "ai", "wearables", "robotics"],
)

