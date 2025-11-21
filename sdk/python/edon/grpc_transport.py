"""gRPC transport implementation for EDON SDK."""

from typing import Dict, Any, Iterator
from pathlib import Path
import sys

from .transport import Transport
from .exceptions import EdonError, EdonConnectionError


class GRPCTransport(Transport):
    """gRPC transport implementation (supports both v1 and v2)."""
    
    def __init__(self, host: str = "localhost", port: int = 50051, version: str = "v1"):
        """
        Initialize gRPC transport.
        
        Args:
            host: gRPC server host
            port: gRPC server port
            version: API version ("v1" or "v2")
        """
        try:
            import grpc
        except ImportError as e:
            raise ImportError(
                f"gRPC dependencies not installed: {e}. "
                f"Install with: pip install grpcio grpcio-tools"
            ) from e
        
        self.version = version
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent
        
        if version == "v2":
            # v2 protobuf files
            grpc_path = project_root / "integrations" / "grpc" / "edon_v2_service"
            
            if not (grpc_path / "edon_v2_pb2.py").exists():
                raise ImportError(
                    f"v2 Protobuf files not found at {grpc_path}. "
                    f"Run: cd {grpc_path} && python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. edon_v2.proto"
                )
            
            if str(grpc_path) not in sys.path:
                sys.path.insert(0, str(grpc_path))
            
            try:
                import edon_v2_pb2
                import edon_v2_pb2_grpc
            except ImportError as e:
                raise ImportError(
                    f"Failed to import v2 protobuf generated code: {e}. "
                    f"Ensure protobuf files are generated."
                ) from e
            
            self.edon_v2_pb2 = edon_v2_pb2
            self.edon_v2_pb2_grpc = edon_v2_pb2_grpc
            
            try:
                self.channel = grpc.insecure_channel(f"{host}:{port}")
                self.stub = self.edon_v2_pb2_grpc.EdonV2ServiceStub(self.channel)
                self.host = host
                self.port = port
            except Exception as e:
                raise EdonConnectionError(f"Failed to create v2 gRPC channel: {e}") from e
        else:
            # v1 protobuf files (default)
            grpc_path = project_root / "integrations" / "grpc" / "edon_grpc_service"
            
            if not (grpc_path / "edon_pb2.py").exists():
                raise ImportError(
                    f"Protobuf files not found at {grpc_path}. "
                    f"Run: cd integrations/grpc/edon_grpc_service && python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. edon.proto"
                )
            
            if str(grpc_path) not in sys.path:
                sys.path.insert(0, str(grpc_path))
            
            try:
                import edon_pb2
                import edon_pb2_grpc
            except ImportError as e:
                raise ImportError(
                    f"Failed to import protobuf generated code: {e}. "
                    f"Ensure protobuf files are generated."
                ) from e
            
            self.edon_pb2 = edon_pb2
            self.edon_pb2_grpc = edon_pb2_grpc
            
            try:
                self.channel = grpc.insecure_channel(f"{host}:{port}")
                self.stub = self.edon_pb2_grpc.EdonServiceStub(self.channel)
                self.host = host
                self.port = port
            except Exception as e:
                raise EdonConnectionError(f"Failed to create gRPC channel: {e}") from e
    
    def _window_to_request(self, window: Dict[str, Any]) -> Any:
        """Convert window dict to protobuf request."""
        req = self.edon_pb2.CavRequest()
        req.eda[:] = window.get("EDA", [])
        req.temp[:] = window.get("TEMP", [])
        req.bvp[:] = window.get("BVP", [])
        req.acc_x[:] = window.get("ACC_x", [])
        req.acc_y[:] = window.get("ACC_y", [])
        req.acc_z[:] = window.get("ACC_z", [])
        req.temp_c = window.get("temp_c", 0.0)
        req.humidity = window.get("humidity", 0.0)
        req.aqi = window.get("aqi", 0)
        req.local_hour = window.get("local_hour", 12)
        return req
    
    def _response_to_dict(self, resp: Any) -> Dict[str, Any]:
        """Convert protobuf response to dict."""
        return {
            "cav_raw": resp.cav_raw,
            "cav_smooth": resp.cav_smooth,
            "state": resp.state,
            "parts": {
                "bio": resp.parts.bio,
                "env": resp.parts.env,
                "circadian": resp.parts.circadian,
                "p_stress": resp.parts.p_stress,
            },
            "controls": {
                "speed": resp.controls.speed,
                "torque": resp.controls.torque,
                "safety": resp.controls.safety,
            },
            "timestamp_ms": resp.timestamp_ms,
        }
    
    def compute_cav(self, window: Dict[str, Any]) -> Dict[str, Any]:
        """Compute CAV via gRPC."""
        try:
            req = self._window_to_request(window)
            resp = self.stub.GetState(req)
            return self._response_to_dict(resp)
        except Exception as e:
            raise EdonError(f"gRPC CAV computation failed: {str(e)}") from e
    
    def stream(self, window: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Stream CAV updates via gRPC."""
        try:
            # Create StateStreamRequest for streaming
            req = self.edon_pb2.StateStreamRequest()
            req.eda[:] = window.get("EDA", [])
            req.temp[:] = window.get("TEMP", [])
            req.bvp[:] = window.get("BVP", [])
            req.acc_x[:] = window.get("ACC_x", [])
            req.acc_y[:] = window.get("ACC_y", [])
            req.acc_z[:] = window.get("ACC_z", [])
            req.temp_c = window.get("temp_c", 0.0)
            req.humidity = window.get("humidity", 0.0)
            req.aqi = window.get("aqi", 0)
            req.local_hour = window.get("local_hour", 12)
            req.stream_mode = True
            
            for resp in self.stub.StreamState(req):
                yield self._response_to_dict(resp)
        except Exception as e:
            raise EdonError(f"gRPC streaming failed: {str(e)}") from e
    
    def health(self) -> Dict[str, Any]:
        """Check health via gRPC (simple ping)."""
        try:
            # Create a minimal valid request
            req = self.edon_pb2.CavRequest()
            req.eda[:] = [0.0] * 240
            req.temp[:] = [0.0] * 240
            req.bvp[:] = [0.0] * 240
            req.acc_x[:] = [0.0] * 240
            req.acc_y[:] = [0.0] * 240
            req.acc_z[:] = [0.0] * 240
            req.temp_c = 22.0
            req.humidity = 50.0
            req.aqi = 50
            req.local_hour = 12
            
            self.stub.GetState(req, timeout=2.0)
            return {"ok": True, "transport": "grpc"}
        except Exception as e:
            return {"ok": False, "error": str(e), "transport": "grpc"}
    
    def cav_batch_v2_grpc(
        self,
        windows: list,
        device_profile: str = None,
        timeout: float = None
    ) -> dict:
        """Compute CAV v2 batch via gRPC."""
        if self.version != "v2":
            raise EdonError("cav_batch_v2_grpc requires v2 gRPC transport (version='v2')")
        
        try:
            # Build batch request
            batch_req = self.edon_v2_pb2.CavBatchV2Request()
            
            for window_dict in windows:
                window_proto = self._dict_window_to_v2_proto(window_dict)
                batch_req.windows.append(window_proto)
            
            if device_profile:
                batch_req.device_profile = device_profile
            
            # Call gRPC
            if timeout:
                import grpc
                resp = self.stub.ComputeCavBatchV2(batch_req, timeout=timeout)
            else:
                resp = self.stub.ComputeCavBatchV2(batch_req)
            
            # Convert response to dict
            return self._v2_proto_response_to_dict(resp)
            
        except Exception as e:
            raise EdonError(f"v2 gRPC batch computation failed: {str(e)}") from e
    
    def stream_v2_grpc(self, windows_iterator) -> Iterator[dict]:
        """Stream v2 CAV computation via bidirectional gRPC."""
        if self.version != "v2":
            raise EdonError("stream_v2_grpc requires v2 gRPC transport (version='v2')")
        
        try:
            def window_generator():
                for window_dict in windows_iterator:
                    yield self._dict_window_to_v2_proto(window_dict)
            
            # Call bidirectional streaming RPC
            for stream_resp in self.stub.StreamCavWindowsV2(window_generator()):
                if stream_resp.ok and stream_resp.HasField('result'):
                    yield self._v2_proto_result_to_dict(stream_resp.result)
                else:
                    # Error response
                    yield {
                        "ok": False,
                        "error": stream_resp.error,
                        "cav_vector": None,
                        "state_class": None,
                        "p_stress": None,
                        "p_chaos": None,
                        "influences": None,
                        "confidence": None,
                        "metadata": None
                    }
                    
        except Exception as e:
            raise EdonError(f"v2 gRPC streaming failed: {str(e)}") from e
    
    def _dict_window_to_v2_proto(self, window_dict: dict):
        """Convert window dict to v2 proto CavWindowV2."""
        window_proto = self.edon_v2_pb2.CavWindowV2()
        
        # Physio
        if 'physio' in window_dict and window_dict['physio']:
            physio = window_dict['physio']
            window_proto.physio.EDA[:] = physio.get('EDA', [])
            window_proto.physio.BVP[:] = physio.get('BVP', [])
            window_proto.physio.TEMP[:] = physio.get('TEMP', [])
        
        # Motion
        if 'motion' in window_dict and window_dict['motion']:
            motion = window_dict['motion']
            window_proto.motion.ACC_x[:] = motion.get('ACC_x', [])
            window_proto.motion.ACC_y[:] = motion.get('ACC_y', [])
            window_proto.motion.ACC_z[:] = motion.get('ACC_z', [])
            if motion.get('velocity'):
                window_proto.motion.velocity[:] = motion['velocity']
            if motion.get('torque'):
                window_proto.motion.torque[:] = motion['torque']
        
        # Environment
        if 'env' in window_dict and window_dict['env']:
            env = window_dict['env']
            window_proto.environment.temp_c = env.get('temp_c', 0.0)
            window_proto.environment.humidity = env.get('humidity', 0.0)
            window_proto.environment.aqi = env.get('aqi', 0)
            window_proto.environment.local_hour = env.get('local_hour', 12)
        
        # Vision
        if 'vision' in window_dict and window_dict['vision']:
            vision = window_dict['vision']
            if vision.get('embedding'):
                window_proto.vision.embedding[:] = vision['embedding']
            if vision.get('objects'):
                window_proto.vision.objects[:] = vision['objects']
        
        # Audio
        if 'audio' in window_dict and window_dict['audio']:
            audio = window_dict['audio']
            if audio.get('embedding'):
                window_proto.audio.embedding[:] = audio['embedding']
            if audio.get('keywords'):
                window_proto.audio.keywords[:] = audio['keywords']
        
        # Task
        if 'task' in window_dict and window_dict['task']:
            task = window_dict['task']
            if task.get('id'):
                window_proto.task.id = task['id']
            window_proto.task.complexity = task.get('complexity', 0.0)
            window_proto.task.difficulty = task.get('difficulty', 0.0)
        
        # System
        if 'system' in window_dict and window_dict['system']:
            system = window_dict['system']
            window_proto.system.cpu_usage = system.get('cpu_usage', 0.0)
            window_proto.system.battery_level = system.get('battery_level', 0.0)
            window_proto.system.error_rate = system.get('error_rate', 0.0)
        
        # Device profile
        if 'device_profile' in window_dict and window_dict['device_profile']:
            window_proto.device_profile = window_dict['device_profile']
        
        return window_proto
    
    def _v2_proto_response_to_dict(self, resp) -> dict:
        """Convert v2 proto CavBatchV2Response to dict."""
        return {
            "results": [self._v2_proto_result_to_dict(r) for r in resp.results],
            "latency_ms": resp.latency_ms,
            "server_version": resp.server_version
        }
    
    def _v2_proto_result_to_dict(self, result_proto) -> dict:
        """Convert v2 proto CavResultV2 to dict."""
        if not result_proto.ok:
            return {
                "ok": False,
                "error": result_proto.error,
                "cav_vector": None,
                "state_class": None,
                "p_stress": None,
                "p_chaos": None,
                "influences": None,
                "confidence": None,
                "metadata": None
            }
        
        # Convert influences
        influences = {}
        if result_proto.HasField('influences'):
            influences = {
                "speed_scale": result_proto.influences.speed_scale,
                "torque_scale": result_proto.influences.torque_scale,
                "safety_scale": result_proto.influences.safety_scale,
                "caution_flag": result_proto.influences.caution_flag,
                "emergency_flag": result_proto.influences.emergency_flag,
                "focus_boost": result_proto.influences.focus_boost,
                "recovery_recommended": result_proto.influences.recovery_recommended
            }
        
        # Convert metadata
        metadata = {}
        if result_proto.HasField('metadata'):
            metadata = {
                "modalities_present": list(result_proto.metadata.modalities_present),
                "num_features": result_proto.metadata.num_features,
                "has_embeddings": result_proto.metadata.has_embeddings,
                "scores": dict(result_proto.metadata.scores),
                "device_profile": result_proto.metadata.device_profile,
                "pca_fitted": result_proto.metadata.pca_fitted,
                "neural_confidence": result_proto.metadata.neural_confidence,
                "neural_state_probs": dict(result_proto.metadata.neural_state_probs)
            }
        
        return {
            "ok": True,
            "error": None,
            "cav_vector": list(result_proto.cav_vector),
            "state_class": result_proto.state_class,
            "p_stress": result_proto.p_stress,
            "p_chaos": result_proto.p_chaos,
            "influences": influences,
            "confidence": result_proto.confidence,
            "metadata": metadata
        }
    
    def health_v2(self) -> dict:
        """Check v2 health via gRPC."""
        if self.version != "v2":
            raise EdonError("health_v2 requires v2 gRPC transport (version='v2')")
        
        try:
            if not hasattr(self, 'edon_v2_pb2'):
                raise EdonError("v2 protobuf not loaded")
            req = self.edon_v2_pb2.HealthRequest()
            resp = self.stub.Health(req, timeout=2.0)
            return {
                "ok": resp.ok,
                "mode": resp.mode,
                "engine": resp.engine,
                "neural_loaded": resp.neural_loaded,
                "pca_loaded": resp.pca_loaded,
                "uptime_s": resp.uptime_s,
                "version": resp.version
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "transport": "grpc_v2"}
    
    def close(self):
        """Close gRPC channel."""
        if hasattr(self, 'channel'):
            self.channel.close()

