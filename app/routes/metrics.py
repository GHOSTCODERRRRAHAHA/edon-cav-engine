"""Prometheus metrics endpoint."""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from app.routes.telemetry import _request_count, _latency_sum, get_uptime_seconds

router = APIRouter(tags=["System"])


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns:
        Prometheus-formatted metrics
    """
    uptime_s = get_uptime_seconds()
    avg_latency_ms = _latency_sum / _request_count if _request_count > 0 else 0.0
    
    metrics_text = f"""# HELP edon_requests_total Total number of requests
# TYPE edon_requests_total counter
edon_requests_total {_request_count}

# HELP edon_latency_ms Average request latency in milliseconds
# TYPE edon_latency_ms gauge
edon_latency_ms {avg_latency_ms:.2f}

# HELP edon_uptime_seconds Server uptime in seconds
# TYPE edon_uptime_seconds gauge
edon_uptime_seconds {uptime_s:.2f}
"""
    
    return metrics_text

