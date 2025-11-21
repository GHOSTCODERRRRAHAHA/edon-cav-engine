"""Telemetry routes."""

import time
from fastapi import APIRouter
from app.models import TelemetryResponse

router = APIRouter(tags=["System"])

# Telemetry state (in-memory, resets on restart)
_start_time = time.time()
_request_count = 0
_latency_sum = 0.0


def get_uptime_seconds() -> float:
    """Get server uptime in seconds."""
    return time.time() - _start_time


def record_request(latency_ms: float):
    """Record a request for telemetry."""
    global _request_count, _latency_sum
    _request_count += 1
    _latency_sum += latency_ms


@router.get("/telemetry", response_model=TelemetryResponse)
async def telemetry() -> TelemetryResponse:
    """
    Telemetry endpoint with request statistics.
    
    Returns:
        TelemetryResponse with request count, average latency, and uptime
    """
    uptime_seconds = time.time() - _start_time
    avg_latency_ms = _latency_sum / _request_count if _request_count > 0 else 0.0
    
    return TelemetryResponse(
        request_count=_request_count,
        avg_latency_ms=avg_latency_ms,
        uptime_seconds=uptime_seconds
    )




