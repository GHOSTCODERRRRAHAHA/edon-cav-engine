"""REST HTTP transport implementation for EDON SDK."""

from typing import Dict, Any, Optional, Iterator
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .transport import Transport
from .exceptions import EdonHTTPError, EdonAuthError, EdonConnectionError


class RESTTransport(Transport):
    """REST HTTP transport implementation."""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 5.0,
        max_retries: int = 2
    ):
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
        try:
            response = self.session.post(
                url,
                json={"windows": [window]},
                headers=self._get_headers(),
                timeout=self.timeout,
            )
            
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
            
            data = response.json()
            results = data.get("results", [])
            if not results:
                raise EdonHTTPError(
                    "No results returned from batch endpoint",
                    status_code=500,
                    response_body=response.text,
                )
            
            result = results[0]
            if not result.get("ok", False):
                error_msg = result.get("error", "Unknown error")
                raise EdonHTTPError(
                    f"CAV computation failed: {error_msg}",
                    status_code=400,
                    response_body=str(result),
                )
            
            return {
                "cav_raw": result.get("cav_raw"),
                "cav_smooth": result.get("cav_smooth"),
                "state": result.get("state"),
                "parts": result.get("parts", {}),
            }
            
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            raise EdonConnectionError(f"Connection error: {str(e)}") from e
        except (EdonAuthError, EdonHTTPError):
            raise
        except Exception as e:
            raise EdonHTTPError(f"Unexpected error: {str(e)}", status_code=500) from e
    
    def stream(self, window: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """REST doesn't support streaming - yield single result."""
        yield self.compute_cav(window)
    
    def health(self) -> Dict[str, Any]:
        """Check health via REST."""
        url = f"{self.base_url}/health"
        try:
            response = self.session.get(
                url,
                headers=self._get_headers(),
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"ok": False, "error": str(e), "transport": "rest"}

