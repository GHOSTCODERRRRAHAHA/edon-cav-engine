#!/usr/bin/env python3
"""
EDON v2 Server Auto-Start Script

Auto-detects and starts the EDON v2 server on http://127.0.0.1:8001
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path


def check_server_running(url: str = "http://127.0.0.1:8001/health", timeout: float = 2.0) -> bool:
    """Check if EDON v2 server is already running."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            if data.get("mode") == "v2":
                return True
    except (requests.exceptions.RequestException, KeyError, ValueError):
        pass
    return False


def find_uvicorn() -> str:
    """Find uvicorn executable."""
    # Try python -m uvicorn first
    try:
        result = subprocess.run(
            [sys.executable, "-m", "uvicorn", "--version"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            return f"{sys.executable} -m uvicorn"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Try direct uvicorn
    try:
        result = subprocess.run(
            ["uvicorn", "--version"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            return "uvicorn"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return None


def check_models_exist():
    """Check if required model files exist."""
    missing = []
    models_dir = Path("models")
    
    # PCA is optional but recommended
    pca_path = models_dir / "pca.pkl"
    if not pca_path.exists():
        missing.append("pca.pkl (optional)")
    
    # Neural head is optional but recommended
    neural_path = models_dir / "neural_head.pt"
    if not neural_path.exists():
        missing.append("neural_head.pt (optional)")
    
    return len(missing) == 0, missing


def start_server(port: int = 8001, host: str = "127.0.0.1") -> subprocess.Popen:
    """Start the EDON v2 server."""
    # Set environment variables
    env = os.environ.copy()
    env["EDON_MODE"] = "v2"
    
    # Set optional model paths if they exist
    models_dir = Path("models")
    pca_path = models_dir / "pca.pkl"
    neural_path = models_dir / "neural_head.pt"
    
    if pca_path.exists():
        env["EDON_PCA_PATH"] = str(pca_path.absolute())
    else:
        env["EDON_PCA_PATH"] = "models/pca.pkl"  # Default path
    
    if neural_path.exists():
        env["EDON_NEURAL_WEIGHTS"] = str(neural_path.absolute())
    else:
        env["EDON_NEURAL_WEIGHTS"] = "models/neural_head.pt"  # Default path
    
    # Find uvicorn
    uvicorn_cmd = find_uvicorn()
    if not uvicorn_cmd:
        print("ERROR: uvicorn not found. Install with: pip install uvicorn")
        sys.exit(1)
    
    # Build command
    cmd = uvicorn_cmd.split() + [
        "app.main:app",
        "--host", host,
        "--port", str(port)
    ]
    
    print(f"Starting EDON v2 server on http://{host}:{port}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Environment: EDON_MODE=v2")
    if pca_path.exists():
        print(f"  EDON_PCA_PATH={env['EDON_PCA_PATH']}")
    if neural_path.exists():
        print(f"  EDON_NEURAL_WEIGHTS={env['EDON_NEURAL_WEIGHTS']}")
    print()
    
    # Start server
    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        return process
    except Exception as e:
        print(f"ERROR: Failed to start server: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    port = 8001
    host = "127.0.0.1"
    health_url = f"http://{host}:{port}/health"
    
    print("=" * 70)
    print("EDON v2 Server Auto-Start")
    print("=" * 70)
    print()
    
    # Check if server is already running
    if check_server_running(health_url):
        print(f"✓ EDON v2 server is already running at {health_url}")
        print("  Server is ready to accept requests.")
        return
    
    # Check if app.main exists
    if not Path("app/main.py").exists():
        print("ERROR: app/main.py not found.")
        print("  Make sure you're running this from the project root directory.")
        sys.exit(1)
    
    # Check models (warn but don't fail)
    models_ok, missing = check_models_exist()
    if not models_ok:
        print("WARNING: Some model files are missing (server will use defaults):")
        for m in missing:
            print(f"  - {m}")
        print()
    
    # Start server
    try:
        process = start_server(port=port, host=host)
        
        # Wait for server to start
        print("Waiting for server to start...")
        max_wait = 10  # seconds
        for i in range(max_wait):
            time.sleep(1)
            if check_server_running(health_url):
                print()
                print("=" * 70)
                print("✓ EDON v2 server is running!")
                print("=" * 70)
                print(f"  URL: http://{host}:{port}")
                print(f"  Health: {health_url}")
                print(f"  API: http://{host}:{port}/v2/oem/cav/batch")
                print(f"  Docs: http://{host}:{port}/docs")
                print()
                print("Press Ctrl+C to stop the server")
                print("=" * 70)
                print()
                
                # Stream output
                try:
                    for line in process.stdout:
                        print(line.rstrip())
                except KeyboardInterrupt:
                    print("\nStopping server...")
                    process.terminate()
                    process.wait(timeout=5)
                    print("Server stopped.")
                return
            
            if process.poll() is not None:
                # Process died
                print("ERROR: Server process exited unexpectedly")
                print("Output:")
                for line in process.stdout:
                    print(line.rstrip())
                sys.exit(1)
        
        print(f"WARNING: Server did not respond after {max_wait} seconds")
        print("  Check the output above for errors")
        print("  Server process is still running - check logs")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\nInterrupted. Stopping server...")
        if 'process' in locals():
            process.terminate()
            process.wait(timeout=5)
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

