#!/usr/bin/env python3
"""
EDON v2 gRPC Server

Provides gRPC interface for v2 multimodal CAV computation.
Supports batch computation and bidirectional streaming.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import time
import threading
from concurrent import futures
from typing import Iterator
import grpc
import logging

# Import v2 engine
from app.v2.engine_v2 import CAVEngineV2
from app.v2.schemas_v2 import V2CavWindow, InfluenceFields
from app.v2 import __version__ as v2_version
from app import __version__ as app_version

# License enforcement
try:
    from app.licensing import validate_license, LicenseError
    LICENSING_AVAILABLE = True
except ImportError:
    LICENSING_AVAILABLE = False

# Import generated protobuf code
try:
    grpc_dir = Path(__file__).parent
    if str(grpc_dir) not in sys.path:
        sys.path.insert(0, str(grpc_dir))
    import edon_v2_pb2
    import edon_v2_pb2_grpc
except ImportError as e:
    print(f"ERROR: Protobuf files not generated. Run:")
    print(f"  cd {Path(__file__).parent}")
    print(f"  python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. edon_v2.proto")
    sys.exit(1)

logger = logging.getLogger(__name__)

# Track startup time for uptime
_start_time = time.time()

# Global engine instance (will be set by main or created here)
_engine_v2 = None
_engine_lock = threading.Lock()


def get_engine() -> CAVEngineV2:
    """Get or create v2 engine instance."""
    global _engine_v2
    if _engine_v2 is None:
        # Try to get from main.py if available
        try:
            from app.main import ENGINE_V2
            if ENGINE_V2 is not None:
                _engine_v2 = ENGINE_V2
                logger.info("[EDON v2 gRPC] Using engine from main.py")
                return _engine_v2
        except Exception:
            pass
        
        # Create new engine
        device_profile = os.getenv("EDON_DEVICE_PROFILE", None)
        _engine_v2 = CAVEngineV2(device_profile=device_profile)
        logger.info("[EDON v2 gRPC] Created new engine instance")
    return _engine_v2


class EdonV2ServiceServicer(edon_v2_pb2_grpc.EdonV2ServiceServicer):
    """gRPC service implementation for EDON v2 CAV computation."""
    
    def __init__(self):
        """Initialize the service with v2 engine."""
        self.engine = get_engine()
        logger.info("[EDON v2 gRPC] Service initialized")
    
    def Health(self, request, context):
        """Health check endpoint."""
        try:
            uptime_s = time.time() - _start_time
            
            # Check if PCA and neural are loaded
            pca_loaded = getattr(self.engine, 'pca_fitted', False)
            neural_loaded = hasattr(self.engine, 'neural_head') and hasattr(self.engine.neural_head, 'model')
            
            return edon_v2_pb2.HealthResponse(
                ok=True,
                mode="v2",
                engine="v2",
                neural_loaded=neural_loaded,
                pca_loaded=pca_loaded,
                uptime_s=uptime_s,
                version=f"EDON CAV Engine v{app_version} (v2 API: {v2_version})"
            )
        except Exception as e:
            logger.exception(f"Health check failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Health check failed: {str(e)}')
            return edon_v2_pb2.HealthResponse(ok=False)
    
    def ComputeCavBatchV2(self, request, context):
        """Batch CAV computation for v2."""
        # License validation
        if LICENSING_AVAILABLE:
            try:
                validate_license(force_online=False)
            except LicenseError as e:
                context.set_code(grpc.StatusCode.PERMISSION_DENIED)
                context.set_details(f"License validation failed: {e}")
                return edon_v2_pb2.CavBatchV2Response()
        
        start_time = time.time()
        
        try:
            if not request.windows:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("windows must be non-empty")
                return edon_v2_pb2.CavBatchV2Response()
            
            if len(request.windows) > 10:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Maximum 10 windows per batch")
                return edon_v2_pb2.CavBatchV2Response()
            
            results = []
            
            with _engine_lock:
                for window_proto in request.windows:
                    try:
                        # Convert proto window to Pydantic model
                        window_dict = self._proto_window_to_dict(window_proto)
                        window = V2CavWindow(**window_dict)
                        
                        # Get device profile (from window or request)
                        device_profile = window_proto.device_profile or request.device_profile or None
                        
                        # Compute CAV v2
                        result = self.engine.compute_cav_v2(window, device_profile=device_profile)
                        
                        # Convert result to proto
                        result_proto = self._dict_result_to_proto(result)
                        results.append(result_proto)
                        
                    except Exception as e:
                        logger.exception(f"Error processing v2 window: {e}")
                        # Per-window error: return ok=false
                        error_result = edon_v2_pb2.CavResultV2(
                            ok=False,
                            error=str(e)
                        )
                        results.append(error_result)
            
            latency_ms = (time.time() - start_time) * 1000.0
            
            return edon_v2_pb2.CavBatchV2Response(
                results=results,
                latency_ms=latency_ms,
                server_version=f"EDON CAV Engine v{app_version} (v2 API: {v2_version})"
            )
            
        except Exception as e:
            logger.exception(f"Batch computation failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Batch computation failed: {str(e)}')
            return edon_v2_pb2.CavBatchV2Response()
    
    def StreamCavWindowsV2(self, request_iterator, context):
        """Bidirectional streaming: process windows as they arrive."""
        # License validation (once at start of stream)
        if LICENSING_AVAILABLE:
            try:
                validate_license(force_online=False)
            except LicenseError as e:
                context.set_code(grpc.StatusCode.PERMISSION_DENIED)
                context.set_details(f"License validation failed: {e}")
                return
        
        try:
            for window_proto in request_iterator:
                try:
                    # Convert proto window to Pydantic model
                    window_dict = self._proto_window_to_dict(window_proto)
                    window = V2CavWindow(**window_dict)
                    
                    # Get device profile
                    device_profile = window_proto.device_profile or None
                    
                    # Compute CAV v2
                    with _engine_lock:
                        result = self.engine.compute_cav_v2(window, device_profile=device_profile)
                    
                    # Convert result to proto
                    result_proto = self._dict_result_to_proto(result)
                    
                    # Yield stream response
                    yield edon_v2_pb2.CavStreamResponse(
                        ok=True,
                        error="",
                        result=result_proto
                    )
                    
                except Exception as e:
                    logger.exception(f"Error processing stream window: {e}")
                    # Per-window error: yield error response but continue
                    yield edon_v2_pb2.CavStreamResponse(
                        ok=False,
                        error=str(e),
                        result=None
                    )
                    
        except Exception as e:
            logger.exception(f"Streaming failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Streaming failed: {str(e)}')
    
    def _proto_window_to_dict(self, window_proto) -> dict:
        """Convert proto CavWindowV2 to dict for Pydantic."""
        window_dict = {}
        
        # Physio
        if window_proto.HasField('physio'):
            physio = {}
            if window_proto.physio.EDA:
                physio['EDA'] = list(window_proto.physio.EDA)
            if window_proto.physio.BVP:
                physio['BVP'] = list(window_proto.physio.BVP)
            if window_proto.physio.TEMP:
                physio['TEMP'] = list(window_proto.physio.TEMP)
            if physio:
                window_dict['physio'] = physio
        
        # Motion
        if window_proto.HasField('motion'):
            motion = {}
            if window_proto.motion.ACC_x:
                motion['ACC_x'] = list(window_proto.motion.ACC_x)
            if window_proto.motion.ACC_y:
                motion['ACC_y'] = list(window_proto.motion.ACC_y)
            if window_proto.motion.ACC_z:
                motion['ACC_z'] = list(window_proto.motion.ACC_z)
            if window_proto.motion.velocity:
                motion['velocity'] = list(window_proto.motion.velocity)
            if window_proto.motion.torque:
                motion['torque'] = list(window_proto.motion.torque)
            if motion:
                window_dict['motion'] = motion
        
        # Environment
        if window_proto.HasField('environment'):
            env = {}
            if window_proto.environment.temp_c > 0:
                env['temp_c'] = window_proto.environment.temp_c
            if window_proto.environment.humidity > 0:
                env['humidity'] = window_proto.environment.humidity
            if window_proto.environment.aqi > 0:
                env['aqi'] = int(window_proto.environment.aqi)
            if window_proto.environment.local_hour >= 0:
                env['local_hour'] = window_proto.environment.local_hour
            if env:
                window_dict['env'] = env
        
        # Vision
        if window_proto.HasField('vision'):
            vision = {}
            if window_proto.vision.embedding:
                vision['embedding'] = list(window_proto.vision.embedding)
            if window_proto.vision.objects:
                vision['objects'] = list(window_proto.vision.objects)
            if vision:
                window_dict['vision'] = vision
        
        # Audio
        if window_proto.HasField('audio'):
            audio = {}
            if window_proto.audio.embedding:
                audio['embedding'] = list(window_proto.audio.embedding)
            if window_proto.audio.keywords:
                audio['keywords'] = list(window_proto.audio.keywords)
            if audio:
                window_dict['audio'] = audio
        
        # Task
        if window_proto.HasField('task'):
            task = {}
            # TaskInput schema has 'goal' but examples use 'id'
            # Support both: if proto has 'id', use it as 'goal' for schema compatibility
            # Also support 'goal' directly if provided
            if window_proto.task.id:  # id is a string, check if non-empty
                # Map 'id' to 'goal' for schema compatibility (schema expects 'goal')
                task['goal'] = window_proto.task.id
            elif window_proto.task.goal:
                task['goal'] = window_proto.task.goal
            # Always include complexity if task field is present
            task['complexity'] = window_proto.task.complexity
            if window_proto.task.difficulty > 0:  # Optional field
                task['difficulty'] = window_proto.task.difficulty
            if task:
                window_dict['task'] = task
        
        # System
        if window_proto.HasField('system'):
            system = {}
            if window_proto.system.cpu_usage > 0:
                system['cpu_usage'] = window_proto.system.cpu_usage
            if window_proto.system.battery_level > 0:
                system['battery_level'] = window_proto.system.battery_level
            if window_proto.system.error_rate > 0:
                system['error_rate'] = window_proto.system.error_rate
            if system:
                window_dict['system'] = system
        
        # Device profile
        if window_proto.device_profile:
            window_dict['device_profile'] = window_proto.device_profile
        
        return window_dict
    
    def _dict_result_to_proto(self, result: dict) -> edon_v2_pb2.CavResultV2:
        """Convert engine result dict to proto CavResultV2."""
        # Build influences
        influences = edon_v2_pb2.Influences(
            speed_scale=result['influences']['speed_scale'],
            torque_scale=result['influences']['torque_scale'],
            safety_scale=result['influences']['safety_scale'],
            caution_flag=result['influences']['caution_flag'],
            emergency_flag=result['influences']['emergency_flag'],
            focus_boost=result['influences']['focus_boost'],
            recovery_recommended=result['influences']['recovery_recommended']
        )
        
        # Build metadata
        metadata = edon_v2_pb2.Metadata(
            modalities_present=result['metadata'].get('modalities_present', []),
            num_features=result['metadata'].get('num_features', 0),
            has_embeddings=result['metadata'].get('has_embeddings', False),
            scores=result['metadata'].get('scores', {}),
            device_profile=result['metadata'].get('device_profile', ''),
            pca_fitted=result['metadata'].get('pca_fitted', False),
            neural_confidence=result['metadata'].get('neural_confidence', 0.0),
            neural_state_probs=result['metadata'].get('neural_state_probs', {})
        )
        
        # Build result
        return edon_v2_pb2.CavResultV2(
            ok=True,
            error="",
            cav_vector=result['cav_vector'],
            state_class=result['state_class'],
            p_stress=result['p_stress'],
            p_chaos=result['p_chaos'],
            influences=influences,
            confidence=result['confidence'],
            metadata=metadata
        )


def serve(port: int = 50052, max_workers: int = 10):
    """Start the v2 gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    edon_v2_pb2_grpc.add_EdonV2ServiceServicer_to_server(EdonV2ServiceServicer(), server)
    
    listen_addr = f'0.0.0.0:{port}'
    server.add_insecure_port(listen_addr)
    
    logger.info(f'[EDON v2 gRPC] Starting server on {listen_addr}...')
    print(f'[EDON v2 gRPC] Starting server on {listen_addr}...')
    server.start()
    logger.info(f'[EDON v2 gRPC] Server started on port {port}')
    print(f'[EDON v2 gRPC] Server started on port {port}')
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info('[EDON v2 gRPC] Shutting down server...')
        print('[EDON v2 gRPC] Shutting down server...')
        server.stop(0)


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='EDON v2 gRPC Server')
    parser.add_argument('--port', type=int, default=50052, help='gRPC server port (default: 50052)')
    parser.add_argument('--workers', type=int, default=10, help='Max worker threads (default: 10)')
    args = parser.parse_args()
    
    serve(port=args.port, max_workers=args.workers)

