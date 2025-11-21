"""Transport layer abstraction for EDON SDK."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Iterator
from enum import Enum


class TransportType(Enum):
    """Transport layer types."""
    REST = "rest"
    GRPC = "grpc"


class Transport(ABC):
    """Abstract base class for transport implementations."""
    
    @abstractmethod
    def compute_cav(self, window: Dict[str, Any]) -> Dict[str, Any]:
        """Compute CAV from a sensor window."""
        pass
    
    @abstractmethod
    def stream(self, window: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Stream CAV updates (server push)."""
        pass
    
    @abstractmethod
    def health(self) -> Dict[str, Any]:
        """Check service health."""
        pass


class RESTTransport(Transport):
    """REST HTTP transport implementation."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 5.0, max_retries: int = 2):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def compute_cav(self, window: Dict[str, Any]) -> Dict[str, Any]:
        """Compute CAV via REST API."""
        url = f"{self.base_url}/oem/cav/batch"
        response = self.session.post(
            url,
            json={"windows": [window]},
            headers=self._get_headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if not results:
            raise ValueError("No results returned")
        result = results[0]
        if not result.get("ok", False):
            raise ValueError(f"CAV computation failed: {result.get('error')}")
        return {
            "cav_raw": result.get("cav_raw"),
            "cav_smooth": result.get("cav_smooth"),
            "state": result.get("state"),
            "parts": result.get("parts", {}),
        }
    
    def stream(self, window: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """REST doesn't support streaming - yield single result."""
        yield self.compute_cav(window)
    
    def health(self) -> Dict[str, Any]:
        """Check health via REST."""
        url = f"{self.base_url}/health"
        response = self.session.get(url, headers=self._get_headers(), timeout=self.timeout)
        response.raise_for_status()
        return response.json()


class GRPCTransport(Transport):
    """gRPC transport implementation."""
    
    def __init__(self, host: str = "localhost", port: int = 50051):
        try:
            import grpc
            # Import generated protobuf code
            import sys
            from pathlib import Path
            
            # Try to find generated protobuf files
            grpc_path = Path(__file__).parent.parent.parent.parent / "integrations" / "grpc" / "edon_grpc_service"
            if (grpc_path / "edon_pb2.py").exists():
                sys.path.insert(0, str(grpc_path))
                import edon_pb2
                import edon_pb2_grpc
                self.edon_pb2 = edon_pb2
                self.edon_pb2_grpc = edon_pb2_grpc
            else:
                raise ImportError("Protobuf files not generated. Run generate_proto.sh")
            
            self.channel = grpc.insecure_channel(f"{host}:{port}")
            self.stub = self.edon_pb2_grpc.EdonServiceStub(self.channel)
            self.host = host
            self.port = port
        except ImportError as e:
            raise ImportError(f"gRPC dependencies not installed: {e}. Install with: pip install grpcio") from e
    
    def _window_to_request(self, window: Dict[str, Any]) -> Any:
        """Convert window dict to protobuf request."""
        req = self.edon_pb2.StreamDataRequest()
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
        req.stream_mode = False
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
        req = self._window_to_request(window)
        resp = self.stub.GetState(req)
        return self._response_to_dict(resp)
    
    def stream(self, window: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Stream CAV updates via gRPC."""
        req = self._window_to_request(window)
        req.stream_mode = True
        for resp in self.stub.StreamState(req):
            yield self._response_to_dict(resp)
    
    def health(self) -> Dict[str, Any]:
        """Check health via gRPC (simple ping)."""
        # gRPC doesn't have a standard health endpoint, so we'll try a simple request
        try:
            # Create a minimal valid request
            req = self.edon_pb2.StreamDataRequest()
            # Fill with zeros as placeholder
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
            req.stream_mode = False
            
            self.stub.GetState(req, timeout=2.0)
            return {"ok": True, "transport": "grpc"}
        except Exception as e:
            return {"ok": False, "error": str(e), "transport": "grpc"}
    
    def close(self):
        """Close gRPC channel."""
        if hasattr(self, 'channel'):
            self.channel.close()

