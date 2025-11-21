"""EDON CAV Engine Python SDK Client."""

import os
from typing import Dict, List, Optional, Any, Iterator

from .transport import TransportType
from .rest_transport import RESTTransport
from .grpc_transport import GRPCTransport
from .exceptions import EdonError, EdonHTTPError


class EdonClient:
    """
    Client for interacting with the EDON CAV Engine API.
    
    Supports both REST and gRPC transports. Uses environment variables
    for configuration:
    - EDON_BASE_URL: Base URL for REST API (default: http://127.0.0.1:8000)
    - EDON_API_TOKEN: API token for authentication (optional)
    
    Example:
        >>> from edon import EdonClient, TransportType
        >>> 
        >>> # REST transport (default)
        >>> client = EdonClient()
        >>> 
        >>> # Or with gRPC
        >>> client = EdonClient(transport=TransportType.GRPC)
        >>> 
        >>> window = {
        ...     "EDA": [0.1] * 240,
        ...     "TEMP": [36.5] * 240,
        ...     "BVP": [0.5] * 240,
        ...     "ACC_x": [0.0] * 240,
        ...     "ACC_y": [0.0] * 240,
        ...     "ACC_z": [1.0] * 240,
        ...     "temp_c": 22.0,
        ...     "humidity": 50.0,
        ...     "aqi": 35,
        ...     "local_hour": 14,
        ... }
        >>> 
        >>> result = client.cav(window)
        >>> print(result["state"])
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 5.0,
        max_retries: int = 2,
        verbose: bool = False,
        transport: TransportType = TransportType.REST,
        grpc_host: str = "localhost",
        grpc_port: int = 50051,
        grpc_version: str = "v1",
    ):
        """
        Initialize the EDON client.
        
        Args:
            base_url: Base URL of the EDON API (for REST). If not provided, reads from
                     EDON_BASE_URL env var, or defaults to http://127.0.0.1:8000
            api_key: API token for authentication (REST only). If not provided, reads from
                    EDON_API_TOKEN env var
            timeout: Request timeout in seconds (default: 5.0)
            max_retries: Maximum number of retries on 5xx or connection errors (default: 2)
            verbose: If True, log requests to stdout (default: False)
            transport: Transport layer type - REST or GRPC (default: REST)
            grpc_host: gRPC server host (default: localhost)
            grpc_port: gRPC server port (default: 50051 for v1, 50052 for v2)
            grpc_version: gRPC API version - "v1" or "v2" (default: "v1")
        """
        self.transport_type = transport
        self.verbose = verbose
        self.grpc_version = grpc_version
        
        # Initialize transport layer
        if transport == TransportType.REST:
            base_url = base_url or os.getenv("EDON_BASE_URL", "http://127.0.0.1:8000")
            api_key = api_key or os.getenv("EDON_API_TOKEN")
            self.transport = RESTTransport(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
            )
        elif transport == TransportType.GRPC:
            self.transport = GRPCTransport(host=grpc_host, port=grpc_port, version=grpc_version)
        else:
            raise ValueError(f"Unsupported transport type: {transport}")
    
    def cav(self, window: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute CAV from a single sensor window.
        
        Args:
            window: Sensor window dict with:
                   - EDA, TEMP, BVP, ACC_x, ACC_y, ACC_z (each 240 floats)
                   - temp_c, humidity, aqi, local_hour (environmental data)
        
        Returns:
            CAV response with 'cav_raw', 'cav_smooth', 'state', 'parts', and optional 'controls' (gRPC)
            
        Raises:
            EdonHTTPError: If the request fails (REST)
            EdonError: If the request fails (gRPC)
        """
        try:
            return self.transport.compute_cav(window)
        except Exception as e:
            if self.transport_type == TransportType.REST:
                if isinstance(e, EdonHTTPError):
                    raise
                raise EdonHTTPError(f"CAV computation failed: {str(e)}", status_code=500) from e
            else:
                if isinstance(e, EdonError):
                    raise
                raise EdonError(f"CAV computation failed: {str(e)}") from e
    
    def cav_batch(self, windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compute CAV for multiple sensor windows in batch (REST only, v1 API).
        
        Args:
            windows: List of sensor window dicts (same format as cav())
        
        Returns:
            List of CAV results, each with 'ok', 'cav_raw', 'cav_smooth', 'state', 'parts'
            or 'error' if processing failed
            
        Raises:
            EdonError: If gRPC transport is used (batch not supported)
        """
        if self.transport_type != TransportType.REST:
            raise EdonError("cav_batch() is only available for REST transport")
        
        # Use REST transport's session directly
        import requests
        url = f"{self.transport.base_url}/oem/cav/batch"
        headers = self.transport._get_headers()
        
        try:
            response = requests.post(
                url,
                json={"windows": windows},
                headers=headers,
                timeout=self.transport.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except Exception as e:
            raise EdonHTTPError(f"Batch CAV computation failed: {str(e)}", status_code=500) from e
    
    def cav_batch_v2(
        self,
        windows: list | None = None,
        device_profile: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
    ) -> dict:
        """
        Call the v2 multimodal CAV batch endpoint.

        Usage patterns (all should work):

        1) Fully explicit:

            res = client.cav_batch_v2(
                windows=[window],
                device_profile="humanoid_full",
            )

        2) Raw payload (backwards compatible):

            res = client.cav_batch_v2({
                "windows": [window],
                "device_profile": "humanoid_full",
            })

        Args:
            windows: List of v2 window dicts.
            device_profile: Optional device profile hint (e.g. "humanoid_full").
            payload: Optional raw dict payload; if provided it takes precedence.
            timeout: Optional request timeout in seconds.

        Returns:
            Response dict with 'results' key containing list of v2 CAV results, each with:
            - ok: bool
            - cav_vector: List[float] (128-dim)
            - state_class: str (restorative|focus|balanced|alert|overload|emergency)
            - p_stress: float
            - p_chaos: float
            - influences: Dict (speed_scale, torque_scale, safety_scale, flags)
            - confidence: float
            - metadata: Dict
            - error: str (if ok=false)
            
        Raises:
            EdonError: If gRPC transport is used (batch not supported)
            ValueError: If neither windows nor payload with 'windows' key is provided
        """
        if self.transport_type != TransportType.REST:
            raise EdonError("cav_batch_v2() is only available for REST transport")
        
        # Backwards-compatible: if the first argument is a dict, treat it as payload
        if isinstance(windows, dict) and payload is None:
            effective_payload = dict(windows)  # shallow copy
        else:
            effective_payload = payload.copy() if isinstance(payload, dict) else {}

            # If caller passed explicit windows, inject them
            if windows is not None:
                effective_payload["windows"] = windows

        # Inject device_profile if provided and not already present
        if device_profile is not None and "device_profile" not in effective_payload:
            effective_payload["device_profile"] = device_profile

        # Basic sanity check
        if "windows" not in effective_payload:
            raise ValueError("cav_batch_v2 requires either `windows` or a payload with a 'windows' key")

        # Use same pattern as cav_batch
        import requests
        url = f"{self.transport.base_url}/v2/oem/cav/batch"
        headers = self.transport._get_headers()
        request_timeout = timeout if timeout is not None else self.transport.timeout
        
        try:
            response = requests.post(
                url,
                json=effective_payload,
                headers=headers,
                timeout=request_timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            try:
                error_detail = e.response.json().get("detail", e.response.text)
            except Exception:
                error_detail = e.response.text
            raise EdonHTTPError(
                f"API error: {e.response.status_code} {e.response.reason} - {error_detail}",
                status_code=e.response.status_code,
                response_body=e.response.text,
            ) from e
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            from .exceptions import EdonConnectionError
            raise EdonConnectionError(f"Connection error: {str(e)}") from e
        except Exception as e:
            raise EdonHTTPError(f"v2 Batch CAV computation failed: {str(e)}", status_code=500) from e
    
    def cav_batch_v2_grpc(
        self,
        windows: list | None = None,
        device_profile: str | None = None,
        timeout: float | None = None,
    ) -> dict:
        """
        Call the v2 multimodal CAV batch endpoint via gRPC.
        
        Args:
            windows: List of v2 window dicts.
            device_profile: Optional device profile hint (e.g. "humanoid_full").
            timeout: Optional request timeout in seconds.
        
        Returns:
            Response dict with 'results' key containing list of v2 CAV results.
            
        Raises:
            EdonError: If not using gRPC transport or if transport is not v2
        """
        if self.transport_type != TransportType.GRPC:
            raise EdonError("cav_batch_v2_grpc() requires gRPC transport")
        
        if self.grpc_version != "v2":
            raise EdonError("cav_batch_v2_grpc() requires v2 gRPC transport (grpc_version='v2')")
        
        if windows is None:
            raise ValueError("windows parameter is required")
        
        return self.transport.cav_batch_v2_grpc(
            windows=windows,
            device_profile=device_profile,
            timeout=timeout
        )
    
    def stream_v2_grpc(self, windows: list) -> Iterator[dict]:
        """
        Stream v2 CAV computation via bidirectional gRPC.
        
        Args:
            windows: List of v2 window dicts to process.
        
        Yields:
            Result dicts as they are computed.
            
        Raises:
            EdonError: If not using gRPC transport or if transport is not v2
        """
        if self.transport_type != TransportType.GRPC:
            raise EdonError("stream_v2_grpc() requires gRPC transport")
        
        if self.grpc_version != "v2":
            raise EdonError("stream_v2_grpc() requires v2 gRPC transport (grpc_version='v2')")
        
        yield from self.transport.stream_v2_grpc(windows)
    
    def stream(self, window: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Stream CAV updates (server push).
        
        Args:
            window: Sensor window dict (same format as cav())
        
        Yields:
            CAV response dicts as they arrive from the server
        """
        try:
            yield from self.transport.stream(window)
        except Exception as e:
            if self.transport_type == TransportType.REST:
                if isinstance(e, EdonHTTPError):
                    raise
                raise EdonHTTPError(f"Streaming failed: {str(e)}", status_code=500) from e
            else:
                if isinstance(e, EdonError):
                    raise
                raise EdonError(f"Streaming failed: {str(e)}") from e
    
    def classify(self, window: Dict[str, Any]) -> str:
        """
        Classify state from sensor window (convenience method).
        
        Args:
            window: Sensor window dict (same format as cav())
        
        Returns:
            State string: 'overload', 'balanced', 'focus', or 'restorative'
        """
        result = self.cav(window)
        return result.get("state", "unknown")
    
    def health(self) -> Dict[str, Any]:
        """
        Check API health status.
        
        Returns:
            Health response with 'ok' and transport-specific fields
        """
        if self.transport_type == TransportType.GRPC:
            if hasattr(self, 'grpc_version') and self.grpc_version == "v2":
                return self.transport.health_v2()
        return self.transport.health()
    
    def close(self):
        """Close transport connections (gRPC only)."""
        if hasattr(self.transport, 'close'):
            self.transport.close()

