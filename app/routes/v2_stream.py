"""v2 Streaming CAV computation routes (WebSocket)."""

import time
import json
import threading
from typing import Any, Dict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from app.v2.schemas_v2 import V2CavWindow, V2CavResult, InfluenceFields
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

router = APIRouter(prefix="/v2/stream", tags=["v2", "Streaming"])

# Engine will be set by main.py during startup
ENGINE_V2 = None

# Thread lock for engine access
_engine_lock = threading.Lock()

LOGGER = logging.getLogger(__name__)


@router.websocket("/cav")
async def stream_cav_v2(websocket: WebSocket):
    """
    WebSocket endpoint for streaming v2 CAV computation.
    
    Client sends JSON messages with v2 window format:
    {
        "physio": {"EDA": [...], "BVP": [...]},
        "motion": {"ACC_x": [...], "ACC_y": [...], "ACC_z": [...]},
        "env": {"temp_c": 22.0, "humidity": 45.0, "aqi": 20},
        "task": {"id": "work", "complexity": 0.5},
        "device_profile": "humanoid_full"  // optional
    }
    
    Server responds with JSON messages:
    {
        "ok": true,
        "error": null,
        "cav_vector": [...],
        "state_class": "focus",
        "p_stress": 0.325,
        "p_chaos": 0.162,
        "influences": {...},
        "confidence": 0.85,
        "metadata": {...}
    }
    """
    # License validation
    if LICENSING_AVAILABLE:
        try:
            validate_license(force_online=False)
        except LicenseError as e:
            await websocket.accept()
            await websocket.send_json({"ok": False, "error": f"License validation failed: {e}"})
            await websocket.close()
            return
    
    await websocket.accept()
    LOGGER.info("[v2 stream] WebSocket connection established")
    
    # Get engine (should be set by main.py)
    engine = ENGINE_V2
    if engine is None:
        # Fallback: create engine if not set
        engine = CAVEngineV2()
        LOGGER.warning("[v2 stream] Engine not set, created new instance")
    
    try:
        while True:
            # Receive message from client
            try:
                data = await websocket.receive_text()
            except WebSocketDisconnect:
                LOGGER.info("[v2 stream] Client disconnected")
                break
            
            try:
                # Parse JSON
                window_dict = json.loads(data)
                
                # Validate and convert to Pydantic model
                try:
                    window = V2CavWindow(**window_dict)
                except Exception as e:
                    # Send error but keep connection alive
                    error_response = {
                        "ok": False,
                        "error": f"Invalid window format: {str(e)}",
                        "cav_vector": None,
                        "state_class": None,
                        "p_stress": None,
                        "p_chaos": None,
                        "influences": None,
                        "confidence": None,
                        "metadata": None
                    }
                    await websocket.send_text(json.dumps(error_response))
                    continue
                
                # Compute CAV v2
                try:
                    with _engine_lock:
                        device_profile = getattr(window, 'device_profile', None)
                        result = engine.compute_cav_v2(window, device_profile=device_profile)
                    
                    # Build response
                    response = {
                        "ok": True,
                        "error": None,
                        "cav_vector": result['cav_vector'],
                        "state_class": result['state_class'],
                        "p_stress": result['p_stress'],
                        "p_chaos": result['p_chaos'],
                        "influences": result['influences'],
                        "confidence": result['confidence'],
                        "metadata": result['metadata']
                    }
                    
                    await websocket.send_text(json.dumps(response))
                    
                except Exception as e:
                    LOGGER.exception(f"[v2 stream] Error processing window: {e}")
                    # Send error but keep connection alive
                    error_response = {
                        "ok": False,
                        "error": str(e),
                        "cav_vector": None,
                        "state_class": None,
                        "p_stress": None,
                        "p_chaos": None,
                        "influences": None,
                        "confidence": None,
                        "metadata": None
                    }
                    await websocket.send_text(json.dumps(error_response))
                    
            except json.JSONDecodeError as e:
                # Invalid JSON - send error but keep connection alive
                error_response = {
                    "ok": False,
                    "error": f"Invalid JSON: {str(e)}",
                    "cav_vector": None,
                    "state_class": None,
                    "p_stress": None,
                    "p_chaos": None,
                    "influences": None,
                    "confidence": None,
                    "metadata": None
                }
                await websocket.send_text(json.dumps(error_response))
                
    except Exception as e:
        LOGGER.exception(f"[v2 stream] WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        LOGGER.info("[v2 stream] WebSocket connection closed")

