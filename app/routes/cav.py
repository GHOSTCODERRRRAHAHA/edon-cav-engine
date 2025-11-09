"""CAV computation routes."""

from fastapi import APIRouter, HTTPException
from app.models import CAVRequest, CAVResponse, AdaptiveInfo
from app.engine import CAVEngine, STRESS_LABEL
from app.adaptive_memory import AdaptiveMemoryEngine
from typing import Dict

router = APIRouter(prefix="/cav", tags=["CAV"])

# Initialize engine (shared instance)
engine = CAVEngine(stress_label=STRESS_LABEL)

# Initialize adaptive memory engine (shared instance)
memory_engine = AdaptiveMemoryEngine()


@router.post("", response_model=CAVResponse)
async def compute_cav(request: CAVRequest) -> CAVResponse:
    """
    Compute CAV score from physiological window and environmental data.
    
    Args:
        request: CAVRequest with window data and environmental parameters
        
    Returns:
        CAVResponse with CAV score and component parts
    """
    # Convert to window dictionary
    window = {
        'EDA': request.EDA,
        'TEMP': request.TEMP,
        'BVP': request.BVP,
        'ACC_x': request.ACC_x,
        'ACC_y': request.ACC_y,
        'ACC_z': request.ACC_z,
    }
    
    # Compute CAV
    try:
        cav_raw, cav_smooth, state, parts = engine.cav_from_window(
            window,
            temp_c=request.temp_c,
            humidity=request.humidity,
            aqi=request.aqi,
            local_hour=request.local_hour
        )
        
        # Record in memory engine
        memory_engine.record(
            cav_raw=cav_raw,
            cav_smooth=cav_smooth,
            state=state,
            parts=parts,
            temp_c=request.temp_c,
            humidity=request.humidity,
            aqi=request.aqi,
            local_hour=request.local_hour
        )
        
        # Compute adaptive adjustments
        adaptive_dict = memory_engine.compute_adaptive(
            cav_smooth=cav_smooth,
            state=state,
            aqi=request.aqi,
            local_hour=request.local_hour
        )
        
        adaptive_info = AdaptiveInfo(**adaptive_dict)
        
        # Update dashboard data
        try:
            from app.routes.dashboard import update_cav_from_response
            update_cav_from_response({
                'cav_smooth': cav_smooth,
                'state': state,
                'adaptive': adaptive_dict
            })
        except Exception:
            pass  # Dashboard update is optional
        
        return CAVResponse(
            cav_raw=cav_raw,
            cav_smooth=cav_smooth,
            state=state,
            parts=parts,
            adaptive=adaptive_info
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing CAV: {str(e)}")

