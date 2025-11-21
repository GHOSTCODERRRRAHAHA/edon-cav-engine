"""
License management endpoints for EDON v2.
"""

from fastapi import APIRouter, HTTPException
import logging
from typing import Dict, Any

try:
    from app.licensing import get_validator, get_license_info, LicenseError
    LICENSING_AVAILABLE = True
except ImportError:
    LICENSING_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/license", tags=["license"])


@router.get("/info")
async def license_info() -> Dict[str, Any]:
    """Get license information."""
    if not LICENSING_AVAILABLE:
        return {"status": "not_available", "message": "License enforcement not available"}
    
    try:
        return get_license_info()
    except Exception as e:
        logger.error(f"[LICENSE] Error getting license info: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/activate")
async def activate_license(
    activation_code: str = None,
    org_id: str = None,
    project_id: str = None
) -> Dict[str, Any]:
    """Activate evaluation license."""
    if not LICENSING_AVAILABLE:
        raise HTTPException(status_code=503, detail="License enforcement not available")
    
    try:
        validator = get_validator()
        success = validator.activate(
            activation_code=activation_code,
            org_id=org_id,
            project_id=project_id
        )
        if success:
            return {
                "ok": True,
                "message": "License activated successfully",
                "license_info": validator.get_license_info()
            }
        else:
            raise HTTPException(status_code=400, detail="Activation failed")
    except LicenseError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[LICENSE] Activation error: {e}")
        raise HTTPException(status_code=500, detail=f"Activation error: {e}")


@router.post("/validate")
async def validate_license_endpoint() -> Dict[str, Any]:
    """Manually trigger license validation."""
    if not LICENSING_AVAILABLE:
        raise HTTPException(status_code=503, detail="License enforcement not available")
    
    try:
        validator = get_validator()
        validator.validate(force_online=True)
        return {
            "ok": True,
            "message": "License is valid",
            "license_info": validator.get_license_info()
        }
    except LicenseError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"[LICENSE] Validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Validation error: {e}")

