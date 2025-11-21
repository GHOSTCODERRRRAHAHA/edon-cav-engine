"""v2 Batch CAV computation routes."""

import time
import threading
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException
from app.v2.schemas_v2 import (
    V2CavBatchRequest, V2CavBatchResponse, V2CavResult, V2CavWindow,
    InfluenceFields,
    # Backward compatibility aliases
    BatchRequestV2, BatchResponseV2, BatchResponseItemV2
)
from app.v2.engine_v2 import CAVEngineV2
from app.v2 import __version__ as v2_version
from app import __version__ as app_version
import logging

# License enforcement
try:
    from app.licensing import validate_license, LicenseError
    LICENSING_AVAILABLE = True
except ImportError:
    LICENSING_AVAILABLE = False

router = APIRouter(prefix="/v2/oem/cav/batch", tags=["v2", "Batch"])

# Engine will be set by main.py during startup
ENGINE_V2 = None

# Thread lock for engine access
_engine_lock = threading.Lock()

LOGGER = logging.getLogger(__name__)


@router.post("", response_model=V2CavBatchResponse)
async def cav_batch_v2(req: V2CavBatchRequest):
    """
    v2 Multimodal CAV batch endpoint.
    
    Accepts multimodal inputs:
    - physio: Physiological signals (EDA, TEMP, BVP, ACC_x/y/z)
    - motion: Motion and torque data
    - env: Environmental context
    - vision: Vision embeddings and object detection
    - audio: Audio embeddings and keywords
    - task: Task metadata
    - system: System/robotics signals
    """
    # License validation
    if LICENSING_AVAILABLE:
        try:
            validate_license(force_online=False)
        except LicenseError as e:
            raise HTTPException(status_code=403, detail=f"License validation failed: {e}")
    
    start_time = time.time()
    
    if not req.windows:
        raise HTTPException(status_code=422, detail="windows must be a non-empty list")
    
    if len(req.windows) > 10:
        raise HTTPException(status_code=422, detail="Maximum 10 windows per batch")
    
    results = []
    
    # Get engine (should be set by main.py)
    engine = ENGINE_V2
    if engine is None:
        # Fallback: create engine if not set
        engine = CAVEngineV2()
    
    with _engine_lock:  # Thread-safe access to shared engine
        for window in req.windows:
            try:
                # Compute CAV v2 (with optional device profile)
                device_profile = getattr(window, 'device_profile', None)
                result = engine.compute_cav_v2(window, device_profile=device_profile)
                
                # Build influence fields
                influences = InfluenceFields(
                    speed_scale=result['influences']['speed_scale'],
                    torque_scale=result['influences']['torque_scale'],
                    safety_scale=result['influences']['safety_scale'],
                    caution_flag=result['influences']['caution_flag'],
                    emergency_flag=result['influences']['emergency_flag'],
                    focus_boost=result['influences']['focus_boost'],
                    recovery_recommended=result['influences']['recovery_recommended']
                )
                
                # Build result with ok=true (all fields required)
                results.append(V2CavResult(
                    ok=True,
                    error=None,
                    cav_vector=result['cav_vector'],
                    state_class=result['state_class'],
                    p_stress=result['p_stress'],
                    p_chaos=result['p_chaos'],
                    influences=influences,
                    confidence=result['confidence'],
                    metadata=result['metadata']
                ))
            except Exception as e:
                # Per-window error handling: return ok=false with error message
                LOGGER.exception(f"Error processing v2 window: {e}")
                results.append(V2CavResult(
                    ok=False,
                    error=str(e),
                    cav_vector=None,
                    state_class=None,
                    p_stress=None,
                    p_chaos=None,
                    influences=None,
                    confidence=None,
                    metadata=None
                ))
    
    latency_ms = (time.time() - start_time) * 1000.0
    
    return V2CavBatchResponse(
        results=results,
        latency_ms=latency_ms,
        server_version=f"EDON CAV Engine v{app_version} (v2 API: {v2_version})"
    )

