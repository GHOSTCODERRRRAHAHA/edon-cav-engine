"""EDON CAV Engine - Main FastAPI application."""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import time
import os
import logging
from pathlib import Path
from app import __version__
from app.routes import batch, telemetry, memory, dashboard, metrics
from app.routes.streaming import router as streaming_router
from app.routes.ingest import router as ingest_router
from app.routes.models import router as models_router

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, skip

# Initialize logger first
logger = logging.getLogger(__name__)

# License enforcement
try:
    from app.licensing import validate_license, get_license_info, LicenseError, get_validator
    LICENSING_AVAILABLE = True
except ImportError:
    LICENSING_AVAILABLE = False
    logger.warning("[EDON] License enforcement module not available")

# Determine EDON mode from environment variable
EDON_MODE = os.getenv("EDON_MODE", "v1").lower()
if EDON_MODE not in ["v1", "v2"]:
    EDON_MODE = "v1"

logger.info(f"[EDON] Boot mode: {EDON_MODE}")

# Initialize engines based on mode
ENGINE_V2 = None
NEURAL_LOADED = False
PCA_LOADED = False

if EDON_MODE == "v2":
    from app.v2.engine_v2 import CAVEngineV2
    from app.v2.device_profiles import get_profile
    
    # Get device profile from env if specified (must be non-empty)
    device_profile_env = os.getenv("EDON_DEVICE_PROFILE", None)
    device_profile = device_profile_env if device_profile_env and device_profile_env.strip() else None
    ENGINE_V2 = CAVEngineV2(device_profile=device_profile)
    if device_profile:
        logger.info(f"[EDON] Engine initialized with device profile: {device_profile}")
    else:
        logger.info(f"[EDON] Engine initialized without device profile (default weights)")
    
    # Load PCA from environment
    pca_path = os.getenv("EDON_PCA_PATH", "models/pca.pkl")
    if os.path.exists(pca_path):
        try:
            import joblib
            from sklearn.decomposition import PCA as SklearnPCA
            pca_data = joblib.load(pca_path)
            
            # Handle different PCA file formats
            if isinstance(pca_data, SklearnPCA):
                # Direct PCA object
                ENGINE_V2.pca_fusion.pca = pca_data
                ENGINE_V2.pca_fusion.is_fitted = True
                ENGINE_V2.pca_fitted = True
                PCA_LOADED = True
                logger.info(f"[EDON] Loaded PCA (direct object): {pca_path}")
            elif isinstance(pca_data, dict):
                # Dictionary format - try multiple key names
                pca_obj = None
                if 'pca' in pca_data:
                    pca_obj = pca_data['pca']
                elif 'pca_model' in pca_data:
                    pca_obj = pca_data['pca_model']
                elif 'model' in pca_data:
                    pca_obj = pca_data['model']
                elif 'components' in pca_data and 'mean' in pca_data:
                    # Reconstruct PCA from components and mean (sklearn format)
                    try:
                        import numpy as np
                        pca_obj = SklearnPCA()
                        # Set required attributes
                        comps = np.array(pca_data['components'])
                        mean = np.array(pca_data['mean'])
                        pca_obj.components_ = comps
                        pca_obj.mean_ = mean
                        # Determine dimensions
                        if comps.ndim == 2:
                            pca_obj.n_components_ = comps.shape[0]
                            pca_obj.n_features_in_ = pca_data.get('n_features_in_', comps.shape[1])
                        elif comps.ndim == 1:
                            pca_obj.n_components_ = 1
                            pca_obj.n_features_in_ = pca_data.get('n_features_in_', len(comps))
                        else:
                            pca_obj.n_components_ = pca_data.get('n_components', 1)
                            pca_obj.n_features_in_ = pca_data.get('n_features_in_', len(mean))
                        # Set optional attributes if available
                        if 'explained_variance_' in pca_data:
                            pca_obj.explained_variance_ = np.array(pca_data['explained_variance_'])
                        if 'explained_variance_ratio_' in pca_data:
                            pca_obj.explained_variance_ratio_ = np.array(pca_data['explained_variance_ratio_'])
                        if 'singular_values_' in pca_data:
                            pca_obj.singular_values_ = np.array(pca_data['singular_values_'])
                        logger.info(f"[EDON] Reconstructed PCA from components/mean (shape: {comps.shape})")
                    except Exception as e:
                        logger.warning(f"[EDON] Failed to reconstruct PCA: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                        pca_obj = None
                
                if pca_obj is not None:
                    ENGINE_V2.pca_fusion.pca = pca_obj
                    ENGINE_V2.pca_fusion.scaler = pca_data.get('scaler', pca_data.get('standard_scaler', ENGINE_V2.pca_fusion.scaler))
                    ENGINE_V2.pca_fusion.feature_order = pca_data.get('feature_order', pca_data.get('features', ENGINE_V2.pca_fusion.feature_order))
                    ENGINE_V2.pca_fusion.is_fitted = True
                    ENGINE_V2.pca_fitted = True
                    PCA_LOADED = True
                    logger.info(f"[EDON] Loaded PCA: {pca_path}")
                else:
                    logger.warning(f"[EDON] PCA file format not recognized. Keys: {list(pca_data.keys())}")
            else:
                logger.warning(f"[EDON] PCA file has unexpected type: {type(pca_data)}")
        except Exception as e:
            logger.warning(f"[EDON] Failed to load PCA from {pca_path}: {e}")
    else:
        logger.info(f"[EDON] PCA path not found (using default): {pca_path}")
    
    # Load neural head from environment
    neural_weights_path = os.getenv("EDON_NEURAL_WEIGHTS", None)
    if neural_weights_path and os.path.exists(neural_weights_path):
        try:
            import torch
            if hasattr(ENGINE_V2.neural_head, 'model'):
                saved_state = torch.load(neural_weights_path, map_location='cpu')
                # Handle both direct state_dict and wrapped formats
                if isinstance(saved_state, dict) and 'state_dict' in saved_state:
                    saved_state = saved_state['state_dict']
                elif isinstance(saved_state, dict) and 'model' in saved_state:
                    saved_state = saved_state['model']
                
                # Try strict loading first
                try:
                    ENGINE_V2.neural_head.model.load_state_dict(saved_state, strict=True)
                except RuntimeError:
                    # If strict fails, try partial loading (ignore missing/unexpected keys)
                    model_state = ENGINE_V2.neural_head.model.state_dict()
                    # Filter to only load matching keys
                    filtered_state = {k: v for k, v in saved_state.items() if k in model_state and model_state[k].shape == v.shape}
                    if filtered_state:
                        model_state.update(filtered_state)
                        ENGINE_V2.neural_head.model.load_state_dict(model_state, strict=False)
                        logger.info(f"[EDON] Loaded neural head (partial): {len(filtered_state)}/{len(model_state)} layers")
                    else:
                        raise ValueError("No matching layers found in saved state")
                
                ENGINE_V2.neural_head.model.eval()
                NEURAL_LOADED = True
                logger.info(f"[EDON] Loaded neural head: {neural_weights_path}")
        except Exception as e:
            logger.warning(f"[EDON] Failed to load neural head from {neural_weights_path}: {e}")
    elif neural_weights_path:
        logger.info(f"[EDON] Neural weights path not found: {neural_weights_path}")
    
    # Import v2 routes (will be included below)
    from app.routes import v2_batch
    # Update v2_batch to use the global engine
    v2_batch.ENGINE_V2 = ENGINE_V2
    logger.info(f"[EDON] v2 engine initialized and assigned to routes")  


app = FastAPI(
    title="EDON CAV Engine",
    description="Context-Aware Vector scoring API for OEM partners",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for OEM integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
# v1 routes (backwards compatible - always available)
app.include_router(batch.router)
app.include_router(telemetry.router)
app.include_router(metrics.router)
app.include_router(memory.router)
app.include_router(streaming_router)
app.include_router(ingest_router)
from app.routes import debug_state
app.include_router(debug_state.router)
app.include_router(models_router, prefix="/models", tags=["models"])

# v2 routes (multimodal - only if v2 mode)
if EDON_MODE == "v2":
    from app.routes import v2_batch, v2_stream
    app.include_router(v2_batch.router)
    app.include_router(v2_stream.router)
    # Update v2_stream to use the global engine
    v2_stream.ENGINE_V2 = ENGINE_V2
    
    # License management endpoints (v2 only)
    if LICENSING_AVAILABLE:
        from app.routes import license
        app.include_router(license.router)
    
    # License management endpoints (v2 only)
    if LICENSING_AVAILABLE:
        from app.routes import license
        app.include_router(license.router)


# Mount dashboard
# Note: Dash integration requires WSGI-to-ASGI adapter
# For now, we'll serve it on a separate port or use a simpler approach
# The dashboard route is defined in dashboard.py
try:
    from app.routes.dashboard import get_dash_app
    dash_app = get_dash_app()
    
    # Mount Dash app using ASGI adapter
    from starlette.middleware.wsgi import WSGIMiddleware
    app.mount("/dashboard", WSGIMiddleware(dash_app.server))
except Exception as e:
    # Dashboard is optional - log error but don't fail
    import logging
    logging.warning(f"Dashboard not available: {e}")


@app.middleware("http")
async def track_latency(request: Request, call_next):
    """Middleware to track request latency for telemetry."""
    from app.routes import telemetry
    
    start_time = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start_time) * 1000.0
    
    # Record latency for telemetry (only for CAV endpoints)
    if request.url.path.startswith("/oem") or request.url.path.startswith("/v2"):
        telemetry.record_request(latency_ms)
    
    return response


@app.get("/")
async def root():
    """Root endpoint."""
    endpoints = {
        "v1": {
            "batch": "POST /oem/cav/batch"
        },
        "common": {
            "health": "GET /health",
            "telemetry": "GET /telemetry",
            "memory_summary": "GET /memory/summary",
            "memory_clear": "POST /memory/clear",
            "dashboard": "GET /dashboard",
            "models_info": "GET /models/info",
            "docs": "/docs"
        }
    }
    
    if EDON_MODE == "v2":
        endpoints["v2"] = {
            "batch": "POST /v2/oem/cav/batch",
            "stream": "WS /v2/stream/cav"
        }
    
    return {
        "service": "EDON CAV Engine",
        "version": __version__,
        "mode": EDON_MODE,
        "endpoints": endpoints
    }


# Track startup time for uptime
_start_time = time.time()


@app.get("/health")
async def health():
    """
    Unified health check endpoint.
    
    Returns:
        Health status with mode, engine type, and component status
    """
    from fastapi import HTTPException
    
    uptime_s = time.time() - _start_time
    
    # License validation (periodic check for v2)
    license_valid = True
    license_info = {}
    if EDON_MODE == "v2" and LICENSING_AVAILABLE:
        try:
            validate_license(force_online=False)  # Use cached validation
            license_info = get_license_info()
        except LicenseError as e:
            license_valid = False
            license_info = {"error": str(e)}
    
    if EDON_MODE == "v2":
        health_data = {
            "ok": True and license_valid,
            "mode": "v2",
            "engine": "v2",
            "neural_loaded": NEURAL_LOADED,
            "pca_loaded": PCA_LOADED,
            "uptime_s": uptime_s,
            "license": license_info
        }
        if not license_valid:
            raise HTTPException(status_code=403, detail=f"License validation failed: {license_info.get('error')}")
        return health_data
    else:
        # v1 mode - return model info
        from app.routes.models import _discover_model
        model_data = _discover_model()
        model_info = f"{model_data['name']} sha256={model_data['sha256'][:16]}... features={model_data['features']} window={model_data['window']}Hz*{model_data['sample_rate_hz']} pca={model_data['pca_dim']}"
        
        return {
            "ok": True,
            "mode": "v1",
            "engine": "v1",
            "model": model_info,
            "neural_loaded": False,
            "pca_loaded": False,
            "uptime_s": uptime_s
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

