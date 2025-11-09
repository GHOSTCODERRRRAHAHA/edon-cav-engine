"""Batch CAV computation routes."""

import time
from fastapi import APIRouter, HTTPException
from app.models import BatchRequest, BatchResponse, BatchResponseItem, CAVRequest
from app.engine import CAVEngine, STRESS_LABEL
from app import __version__

router = APIRouter(prefix="/oem/cav/batch", tags=["Batch"])

# Note: For batch processing, we use a shared engine instance
# Each window computation maintains its own EMA state within the engine
# For true isolation, consider creating per-request engine instances
engine = CAVEngine(stress_label=STRESS_LABEL)


@router.post("", response_model=BatchResponse)
async def compute_batch(request: BatchRequest) -> BatchResponse:
    """
    Compute CAV scores for multiple windows in batch.
    
    Args:
        request: BatchRequest with list of CAVRequest windows
        
    Returns:
        BatchResponse with results, latency, and server version
    """
    start_time = time.time()
    
    results = []
    
    try:
        for window_req in request.windows:
            # Convert to window dictionary
            window = {
                'EDA': window_req.EDA,
                'TEMP': window_req.TEMP,
                'BVP': window_req.BVP,
                'ACC_x': window_req.ACC_x,
                'ACC_y': window_req.ACC_y,
                'ACC_z': window_req.ACC_z,
            }
            
            # Compute CAV
            cav_raw, cav_smooth, state, parts = engine.cav_from_window(
                window,
                temp_c=window_req.temp_c,
                humidity=window_req.humidity,
                aqi=window_req.aqi,
                local_hour=window_req.local_hour
            )
            
            results.append(BatchResponseItem(
                cav_raw=cav_raw,
                cav_smooth=cav_smooth,
                state=state,
                parts=parts
            ))
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000.0
        
        return BatchResponse(
            results=results,
            latency_ms=latency_ms,
            server_version=f"EDON CAV Engine v{__version__}"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")

