"""Exception classes for EDON SDK."""


class EdonError(Exception):
    """Base exception for all EDON SDK errors."""
    pass


class EdonHTTPError(EdonError):
    """Exception raised for HTTP-related errors."""
    
    def __init__(self, message: str, status_code: int = 0, response_body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class EdonAuthError(EdonHTTPError):
    """Exception raised for authentication errors (401/403)."""
    pass


class EdonConnectionError(EdonError):
    """Exception raised for connection errors (timeouts, network issues)."""
    pass

