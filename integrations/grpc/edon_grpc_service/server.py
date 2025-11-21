#!/usr/bin/env python3
"""
EDON gRPC Server

Provides gRPC interface for real-time CAV computation.
Supports both single request/response and server-side streaming.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import time
import threading
from concurrent import futures
from typing import Iterator
import grpc

# Import EDON engine
from app.engine import CAVEngine, WINDOW_LEN

# Import generated protobuf code
try:
    grpc_dir = Path(__file__).parent
    if str(grpc_dir) not in sys.path:
        sys.path.insert(0, str(grpc_dir))
    import edon_pb2
    import edon_pb2_grpc
except ImportError as e:
    print(f"ERROR: Protobuf files not generated. Run:")
    print(f"  cd {Path(__file__).parent}")
    print(f"  python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. edon.proto")
    sys.exit(1)


class EdonServiceServicer(edon_pb2_grpc.EdonServiceServicer):
    """gRPC service implementation for EDON CAV computation."""
    
    def __init__(self):
        """Initialize the service with CAV engine."""
        self.engine = CAVEngine()
        self._lock = threading.Lock()
        print("EDON gRPC Service initialized")
    
    def GetState(self, request, context):
        """Single request/response for CAV computation."""
        try:
            # Build window dict from request
            window = {
                'EDA': list(request.eda),
                'TEMP': list(request.temp),
                'BVP': list(request.bvp),
                'ACC_x': list(request.acc_x),
                'ACC_y': list(request.acc_y),
                'ACC_z': list(request.acc_z),
            }
            
            # Validate window length
            for key, values in window.items():
                if len(values) != WINDOW_LEN:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(f'Invalid window length for {key}: expected {WINDOW_LEN}, got {len(values)}')
                    return edon_pb2.CavResponse()
            
            # Compute CAV
            with self._lock:
                cav_raw, cav_smooth, state, parts = self.engine.cav_from_window(
                    window=window,
                    temp_c=request.temp_c if request.temp_c > 0 else None,
                    humidity=request.humidity if request.humidity > 0 else None,
                    aqi=request.aqi if request.aqi > 0 else None,
                    local_hour=request.local_hour if request.local_hour >= 0 else 12,
                )
            
            # Compute control scales
            controls = self._compute_controls(state, parts)
            
            # Build response
            response = edon_pb2.CavResponse()
            response.cav_raw = cav_raw
            response.cav_smooth = cav_smooth
            response.state = state
            response.timestamp_ms = int(time.time() * 1000)
            
            # Set component scores
            response.parts.bio = parts.get('bio', 0.0)
            response.parts.env = parts.get('env', 0.0)
            response.parts.circadian = parts.get('circadian', 0.0)
            response.parts.p_stress = parts.get('p_stress', 0.0)
            
            # Set control scales
            response.controls.speed = controls['speed']
            response.controls.torque = controls['torque']
            response.controls.safety = controls['safety']
            
            return response
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'CAV computation failed: {str(e)}')
            return edon_pb2.CavResponse()
    
    def StreamState(self, request, context):
        """Server-side streaming: continuously push state updates."""
        if not request.stream_mode:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details('stream_mode must be True for streaming')
            return
        
        window = {
            'EDA': list(request.eda),
            'TEMP': list(request.temp),
            'BVP': list(request.bvp),
            'ACC_x': list(request.acc_x),
            'ACC_y': list(request.acc_y),
            'ACC_z': list(request.acc_z),
        }
        
        # Validate
        for key, values in window.items():
            if len(values) != WINDOW_LEN:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f'Invalid window length for {key}')
                return
        
        # Stream updates every 5 seconds
        update_interval = 5.0
        last_update = time.time()
        
        try:
            while context.is_active():
                current_time = time.time()
                
                if current_time - last_update >= update_interval:
                    # Compute CAV
                    with self._lock:
                        cav_raw, cav_smooth, state, parts = self.engine.cav_from_window(
                            window=window,
                            temp_c=request.temp_c if request.temp_c > 0 else None,
                            humidity=request.humidity if request.humidity > 0 else None,
                            aqi=request.aqi if request.aqi > 0 else None,
                            local_hour=request.local_hour if request.local_hour >= 0 else 12,
                        )
                    
                    controls = self._compute_controls(state, parts)
                    
                    # Build and yield response
                    response = edon_pb2.StateStreamResponse()
                    response.cav_raw = cav_raw
                    response.cav_smooth = cav_smooth
                    response.state = state
                    response.timestamp_ms = int(time.time() * 1000)
                    
                    response.parts.bio = parts.get('bio', 0.0)
                    response.parts.env = parts.get('env', 0.0)
                    response.parts.circadian = parts.get('circadian', 0.0)
                    response.parts.p_stress = parts.get('p_stress', 0.0)
                    
                    response.controls.speed = controls['speed']
                    response.controls.torque = controls['torque']
                    response.controls.safety = controls['safety']
                    
                    yield response
                    last_update = current_time
                
                time.sleep(0.1)
                
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Streaming failed: {str(e)}')
    
    def _compute_controls(self, state: str, parts: dict) -> dict:
        """Compute robot control scales based on state."""
        if state == 'overload':
            return {'speed': 0.3, 'torque': 0.3, 'safety': 1.0}
        elif state == 'focus':
            return {'speed': 0.7, 'torque': 1.0, 'safety': 0.8}
        elif state == 'balanced':
            return {'speed': 1.0, 'torque': 1.0, 'safety': 0.7}
        else:  # restorative
            return {'speed': 1.0, 'torque': 1.0, 'safety': 0.5}


def serve(port: int = 50051, max_workers: int = 10):
    """Start the gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    edon_pb2_grpc.add_EdonServiceServicer_to_server(EdonServiceServicer(), server)
    
    listen_addr = f'0.0.0.0:{port}'
    server.add_insecure_port(listen_addr)
    
    print(f'Starting EDON gRPC server on {listen_addr}...')
    server.start()
    print(f'EDON gRPC server started on port {port}')
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print('Shutting down gRPC server...')
        server.stop(0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='EDON gRPC Server')
    parser.add_argument('--port', type=int, default=50051, help='gRPC server port')
    parser.add_argument('--workers', type=int, default=10, help='Max worker threads')
    args = parser.parse_args()
    
    serve(port=args.port, max_workers=args.workers)

