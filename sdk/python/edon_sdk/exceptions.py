"""Custom exceptions for EDON SDK."""


class EdonError(Exception):
    """Base exception for all EDON SDK errors."""
    pass


class EdonHTTPError(EdonError):
    """Raised when the API returns an HTTP error (4xx/5xx)."""
    
    def __init__(self, message: str, status_code: int, response_body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class EdonAuthError(EdonHTTPError):
    """Raised when authentication fails (401/403)."""
    pass


class EdonConnectionError(EdonError):
    """Raised when connection to the API fails or times out."""
    pass

