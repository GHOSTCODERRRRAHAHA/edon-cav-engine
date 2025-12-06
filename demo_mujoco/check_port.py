"""Utility to check if a port is available."""

import socket
import sys


def is_port_available(port: int, host: str = '0.0.0.0') -> bool:
    """Check if a port is available."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        sock.close()
        return True
    except OSError:
        sock.close()
        return False


def find_available_port(start_port: int = 8080, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    for i in range(max_attempts):
        port = start_port + i
        if is_port_available(port):
            return port
    raise RuntimeError(f"Could not find available port starting from {start_port}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
        if is_port_available(port):
            print(f"Port {port} is available")
        else:
            print(f"Port {port} is NOT available")
            available = find_available_port(port)
            print(f"Next available port: {available}")
    else:
        available = find_available_port()
        print(f"Available port: {available}")

