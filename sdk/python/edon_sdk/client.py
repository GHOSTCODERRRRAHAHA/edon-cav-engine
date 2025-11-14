"""EDON CAV Engine Python SDK Client."""

import os
import time
from typing import Dict, List, Optional, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .exceptions import EdonError, EdonHTTPError, EdonAuthError, EdonConnectionError


class EdonClient:
    """
    Client for interacting with the EDON CAV Engine API.
    
    Example:
        >>> client = EdonClient(base_url="http://127.0.0.1:8000")
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
    ):
        """
        Initialize the EDON client.
        
        Args:
            base_url: Base URL of the EDON API. If not provided, reads from
                     EDON_BASE_URL env var, or defaults to http://127.0.0.1:8000
            api_key: API token for authentication. If not provided, reads from
                    EDON_API_TOKEN env var. If set, sends as Authorization: Bearer <token>
            timeout: Request timeout in seconds (default: 5.0)
            max_retries: Maximum number of retries on 5xx or connection errors (default: 2)
            verbose: If True, log requests to stdout (default: False)
        """
        self.base_url = base_url or os.getenv("EDON_BASE_URL", "http://127.0.0.1:8000")
        # Remove trailing slash
        self.base_url = self.base_url.rstrip("/")
        
        self.api_key = api_key or os.getenv("EDON_API_TOKEN")
        self.timeout = timeout
        self.max_retries = max_retries
        self.verbose = verbose
        
        # Create session with retry strategy
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
        """Get request headers with optional authentication."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _request(
        self,
        method: str,
        path: str,
        json: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., "/health")
            json: Optional JSON payload for POST requests
            
        Returns:
            Parsed JSON response as dict
            
        Raises:
            EdonAuthError: On 401/403 responses
            EdonHTTPError: On other 4xx/5xx responses
            EdonConnectionError: On connection errors/timeouts
        """
        url = f"{self.base_url}{path}"
        headers = self._get_headers()
        
        if self.verbose:
            print(f"[EDON SDK] {method} {url}")
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=json,
                headers=headers,
                timeout=self.timeout,
            )
            
            if self.verbose:
                print(f"[EDON SDK] {response.status_code} {response.reason}")
            
            # Handle authentication errors
            if response.status_code in (401, 403):
                raise EdonAuthError(
                    f"Authentication failed: {response.status_code} {response.reason}",
                    status_code=response.status_code,
                    response_body=response.text,
                )
            
            # Handle other HTTP errors
            if not response.ok:
                try:
                    error_detail = response.json().get("detail", response.text)
                except Exception:
                    error_detail = response.text
                
                raise EdonHTTPError(
                    f"API error: {response.status_code} {response.reason} - {error_detail}",
                    status_code=response.status_code,
                    response_body=response.text,
                )
            
            return response.json()
            
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            raise EdonConnectionError(
                f"Connection error: {str(e)}"
            ) from e
        except EdonAuthError:
            raise
        except EdonHTTPError:
            raise
        except Exception as e:
            raise EdonError(f"Unexpected error: {str(e)}") from e
    
    def health(self) -> Dict[str, Any]:
        """
        Check API health status.
        
        Returns:
            Health response with 'ok', 'model', and 'uptime_s' fields
        """
        return self._request("GET", "/health")
    
    def cav(self, window: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute CAV from a single sensor window.
        
        Uses the batch endpoint internally (POST /oem/cav/batch) with a single window.
        
        Args:
            window: Sensor window dict with:
                   - EDA, TEMP, BVP, ACC_x, ACC_y, ACC_z (each 240 floats)
                   - temp_c, humidity, aqi, local_hour (environmental data)
        
        Returns:
            CAV response with 'cav_raw', 'cav_smooth', 'state', 'parts', and optional 'adaptive'
            
        Raises:
            EdonHTTPError: If the batch request fails or returns an error for the window
        """
        # Use batch endpoint with single window
        response = self._request("POST", "/oem/cav/batch", json={"windows": [window]})
        results = response.get("results", [])
        
        if not results:
            raise EdonHTTPError(
                "No results returned from batch endpoint",
                status_code=500,
                response_body="",
            )
        
        result = results[0]
        
        # Check if the result has an error
        if not result.get("ok", False):
            error_msg = result.get("error", "Unknown error")
            raise EdonHTTPError(
                f"CAV computation failed: {error_msg}",
                status_code=400,
                response_body=str(result),
            )
        
        # Return the result in the same format as the old /cav endpoint
        return {
            "cav_raw": result.get("cav_raw"),
            "cav_smooth": result.get("cav_smooth"),
            "state": result.get("state"),
            "parts": result.get("parts", {}),
            # Note: adaptive info is not available from batch endpoint
            # but the SDK interface remains compatible
        }
    
    def cav_batch(self, windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compute CAV for multiple sensor windows in batch.
        
        Args:
            windows: List of sensor window dicts (same format as cav())
        
        Returns:
            List of CAV results, each with 'ok', 'cav_raw', 'cav_smooth', 'state', 'parts'
            or 'error' if processing failed
        """
        response = self._request("POST", "/oem/cav/batch", json={"windows": windows})
        return response.get("results", [])
    
    def ingest(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingest sensor frames for state tracking.
        
        Args:
            payload: Ingest payload with 'frames' list containing sensor data
        
        Returns:
            Ingest response with 'ok', 'frames', and state information
        """
        return self._request("POST", "/v1/ingest", json=payload)
    
    def debug_state(self) -> Optional[Dict[str, Any]]:
        """
        Get current debug state (if available).
        
        Returns:
            Debug state dict, or None if endpoint not available (404)
        """
        try:
            return self._request("GET", "/_debug/state")
        except EdonHTTPError as e:
            if e.status_code == 404:
                return None
            raise

