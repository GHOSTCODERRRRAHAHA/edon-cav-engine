"""Robot stability control routes (v8 integration)."""

import os
import time
import logging
import hashlib
import asyncio
import cProfile
import pstats
import io
from typing import Optional, Dict, Any
from functools import lru_cache
from collections import OrderedDict
from fastapi import APIRouter, HTTPException
import numpy as np
import torch

from app.models import RobotStabilityRequest, RobotStabilityResponse, RobotState, Modulations, RecordOutcomeRequest
from app import __version__
from app.robot_stability_memory import get_robot_stability_memory

logger = logging.getLogger(__name__)

# Profiling configuration
ENABLE_PROFILING = os.getenv("EDON_ENABLE_PROFILING", "0") == "1"
PROFILE_SAMPLE_RATE = int(os.getenv("EDON_PROFILE_SAMPLE_RATE", "100"))  # Profile every N requests
_profile_counter = 0

router = APIRouter(prefix="/oem/robot/stability", tags=["Robot Stability"])

# Cache for strategy selections and modulations
# Key: hash of robot state (rounded to avoid cache misses from tiny differences)
# Value: (strategy_id, modulations, timestamp)
# TTL: 0.1 seconds (100ms) - cache for very recent similar states
_strategy_cache: OrderedDict = OrderedDict()
_cache_max_size = 1000  # Max cache entries
_cache_ttl = 0.1  # 100ms TTL

# Global v8 models (loaded on startup)
V8_POLICY: Optional[Any] = None
V8_FAIL_RISK_MODEL: Optional[Any] = None
V8_POLICY_INPUT_SIZE: Optional[int] = None


def set_v8_models(policy: Any, fail_risk_model: Any, input_size: int):
    """Set v8 models (called from main.py on startup)."""
    global V8_POLICY, V8_FAIL_RISK_MODEL, V8_POLICY_INPUT_SIZE
    V8_POLICY = policy
    V8_FAIL_RISK_MODEL = fail_risk_model
    V8_POLICY_INPUT_SIZE = input_size
    logger.info(f"[ROBOT-STABILITY] v8 models loaded: policy_input_size={input_size}")


def _check_models_loaded():
    """Check if v8 models are loaded."""
    if V8_POLICY is None or V8_FAIL_RISK_MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="v8 models not loaded. Please ensure models are available and server is properly initialized."
        )


def _robot_state_to_dict(state: RobotState) -> Dict[str, Any]:
    """Convert Pydantic RobotState to dict."""
    return {
        "roll": state.roll,
        "pitch": state.pitch,
        "roll_velocity": state.roll_velocity,
        "pitch_velocity": state.pitch_velocity,
        "com_x": state.com_x,
        "com_y": state.com_y,
        "com_z": state.com_z if state.com_z is not None else 0.0,
        "yaw": state.yaw if state.yaw is not None else 0.0,
        "yaw_velocity": state.yaw_velocity if state.yaw_velocity is not None else 0.0,
    }


def _hash_robot_state(obs: Dict[str, Any], fail_risk: float) -> str:
    """Create cache key from robot state (rounded to avoid cache misses from tiny differences)."""
    # Round values to reduce cache misses from tiny differences
    rounded = (
        round(obs.get("roll", 0.0), 3),
        round(obs.get("pitch", 0.0), 3),
        round(obs.get("roll_velocity", 0.0), 2),
        round(obs.get("pitch_velocity", 0.0), 2),
        round(obs.get("com_x", 0.0), 2),
        round(obs.get("com_y", 0.0), 2),
        round(fail_risk, 2)
    )
    key_str = str(rounded)
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_cached_strategy(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get cached strategy and modulations if available and not expired."""
    global _strategy_cache
    current_time = time.time()
    
    if cache_key in _strategy_cache:
        cached_data = _strategy_cache[cache_key]
        cached_time = cached_data.get("timestamp", 0)
        
        # Check if cache is still valid (within TTL)
        if current_time - cached_time < _cache_ttl:
            # Move to end (LRU)
            _strategy_cache.move_to_end(cache_key)
            return cached_data.get("result")
        else:
            # Expired, remove
            del _strategy_cache[cache_key]
    
    return None


def _set_cached_strategy(cache_key: str, result: Dict[str, Any]):
    """Cache strategy and modulations."""
    global _strategy_cache
    
    # Remove oldest entries if cache is full
    while len(_strategy_cache) >= _cache_max_size:
        _strategy_cache.popitem(last=False)  # Remove oldest
    
    _strategy_cache[cache_key] = {
        "result": result,
        "timestamp": time.time()
    }


def _compute_baseline_action(obs: Dict[str, Any]) -> np.ndarray:
    """Compute baseline action from observation."""
    try:
        from run_eval import baseline_controller
        baseline = baseline_controller(obs, edon_state=None)
        return np.array(baseline)
    except Exception as e:
        logger.warning(f"[ROBOT-STABILITY] Could not compute baseline action: {e}, using zeros")
        return np.zeros(10)  # Default action size


def _compute_fail_risk(obs: Dict[str, Any], fail_risk_model: Any) -> float:
    """Compute fail risk using fail-risk model."""
    try:
        # Extract features for fail-risk model
        # Fail-risk model expects: roll, pitch, roll_vel, pitch_vel, com_x, com_y, and derived features
        roll = obs.get("roll", 0.0)
        pitch = obs.get("pitch", 0.0)
        roll_vel = obs.get("roll_velocity", 0.0)
        pitch_vel = obs.get("pitch_velocity", 0.0)
        com_x = obs.get("com_x", 0.0)
        com_y = obs.get("com_y", 0.0)
        
        # Build feature vector (matching fail-risk model input)
        features = np.array([
            roll, pitch, roll_vel, pitch_vel, com_x, com_y,
            np.sqrt(roll**2 + pitch**2),  # tilt magnitude
            np.sqrt(roll_vel**2 + pitch_vel**2),  # velocity magnitude
            roll * roll_vel,  # roll energy
            pitch * pitch_vel,  # pitch energy
            com_x * roll,  # COM-roll coupling
            com_y * pitch,  # COM-pitch coupling
            abs(roll), abs(pitch),  # absolute tilts
        ])
        
        # Pad to expected input size (fail-risk model expects 15 features)
        if len(features) < 15:
            features = np.pad(features, (0, 15 - len(features)), mode='constant')
        elif len(features) > 15:
            features = features[:15]
        
        # Compute fail risk
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            fail_risk = fail_risk_model(features_tensor).item()
            return float(np.clip(fail_risk, 0.0, 1.0))
    except Exception as e:
        logger.warning(f"[ROBOT-STABILITY] Could not compute fail risk: {e}, using 0.0")
        return 0.0


def _compute_instability_score(obs: Dict[str, Any]) -> float:
    """Compute instability score from robot state."""
    roll = obs.get("roll", 0.0)
    pitch = obs.get("pitch", 0.0)
    roll_vel = obs.get("roll_velocity", 0.0)
    pitch_vel = obs.get("pitch_velocity", 0.0)
    
    tilt_mag = np.sqrt(roll**2 + pitch**2)
    vel_mag = np.sqrt(roll_vel**2 + pitch_vel**2)
    
    # Simple instability score
    instability = tilt_mag * 2.0 + vel_mag * 1.0
    return float(instability)


def _determine_phase(instability_score: float, fail_risk: float) -> str:
    """Determine phase from instability and fail risk."""
    if fail_risk > 0.7 or instability_score > 0.5:
        return "recovery"
    elif fail_risk > 0.4 or instability_score > 0.3:
        return "warning"
    else:
        return "stable"


@router.post("", response_model=RobotStabilityResponse)
async def robot_stability(req: RobotStabilityRequest):
    """
    Compute robot stability control from robot state.
    
    This endpoint uses EDON v8 strategy policy to compute control modulations
    that prevent robot interventions and maintain stability.
    
    **Input:**
    - Current robot state (roll, pitch, velocities, COM position)
    - Optional: History of previous states (for temporal memory)
    - Optional: Pre-computed fail risk
    
    **Output:**
    - Strategy ID and name
    - Control modulations (gain_scale, compliance, bias)
    - Intervention risk prediction
    
    **Example:**
    ```json
    {
      "robot_state": {
        "roll": 0.05,
        "pitch": 0.02,
        "roll_velocity": 0.1,
        "pitch_velocity": 0.05,
        "com_x": 0.0,
        "com_y": 0.0
      }
    }
    ```
    """
    global _profile_counter
    
    # Timing breakdown
    timings = {
        "total": 0.0,
        "cache_check": 0.0,
        "fail_risk": 0.0,
        "baseline_action": 0.0,
        "instability_score": 0.0,
        "pack_observation": 0.0,
        "policy_inference": 0.0,
        "adaptive_memory": 0.0,
        "cache_store": 0.0
    }
    
    start_time = time.time()
    
    # Profiling setup
    profiler = None
    if ENABLE_PROFILING:
        _profile_counter += 1
        if _profile_counter % PROFILE_SAMPLE_RATE == 0:
            profiler = cProfile.Profile()
            profiler.enable()
    
    # Check models are loaded
    _check_models_loaded()
    
    try:
        # Convert robot state to dict
        obs = _robot_state_to_dict(req.robot_state)
        
        # Compute fail risk if not provided (needed for cache key)
        t0 = time.time()
        if req.fail_risk is None:
            fail_risk = _compute_fail_risk(obs, V8_FAIL_RISK_MODEL)
        else:
            fail_risk = req.fail_risk
        timings["fail_risk"] = (time.time() - t0) * 1000.0
        
        # Check cache first (fast path)
        t0 = time.time()
        cache_key = _hash_robot_state(obs, fail_risk)
        cached_result = _get_cached_strategy(cache_key)
        timings["cache_check"] = (time.time() - t0) * 1000.0
        
        if cached_result is not None:
            # Cache hit! Return immediately
            timings["total"] = (time.time() - start_time) * 1000.0
            logger.debug(f"[ROBOT-STABILITY] Cache hit! Latency: {timings['total']:.2f}ms (cache_check: {timings['cache_check']:.2f}ms)")
            return RobotStabilityResponse(
                strategy_id=cached_result["strategy_id"],
                strategy_name=cached_result["strategy_name"],
                modulations=Modulations(
                    gain_scale=cached_result["gain_scale"],
                    compliance=cached_result["compliance"],
                    bias=cached_result["bias"]
                ),
                intervention_risk=fail_risk,
                latency_ms=timings["total"]
            )
        
        # Cache miss - compute strategy and modulations
        # Compute baseline action if not provided
        t0 = time.time()
        if req.baseline_action is None:
            baseline_action = _compute_baseline_action(obs)
        else:
            baseline_action = np.array(req.baseline_action)
        timings["baseline_action"] = (time.time() - t0) * 1000.0
        
        # Compute instability score
        t0 = time.time()
        instability_score = _compute_instability_score(obs)
        timings["instability_score"] = (time.time() - t0) * 1000.0
        
        # Determine phase
        phase = _determine_phase(instability_score, fail_risk)
        
        # Build history for temporal memory
        obs_history = []
        near_fail_history = []
        obs_vec_history = []
        
        if req.history:
            # Convert history to dict format
            for hist_state in req.history[-8:]:  # Max 8 frames
                hist_obs = _robot_state_to_dict(hist_state)
                obs_history.append(hist_obs)
                near_fail_history.append(1.0 if _compute_fail_risk(hist_obs, V8_FAIL_RISK_MODEL) > 0.5 else 0.0)
        
        # Pack observation for v8 policy
        t0 = time.time()
        from training.edon_v8_policy import pack_stacked_observation_v8
        
        obs_vec = pack_stacked_observation_v8(
            obs=obs,
            baseline_action=baseline_action,
            fail_risk=fail_risk,
            instability_score=instability_score,
            phase=phase,
            obs_history=obs_history,
            near_fail_history=near_fail_history,
            obs_vec_history=obs_vec_history,  # Will be empty for single request
            stack_size=8
        )
        timings["pack_observation"] = (time.time() - t0) * 1000.0
        
        # Convert to tensor
        obs_tensor = torch.FloatTensor(obs_vec).unsqueeze(0)
        
        # Get strategy and modulations from policy
        t0 = time.time()
        V8_POLICY.eval()
        with torch.no_grad():
            strategy_logits, modulations_dict = V8_POLICY(obs_tensor)
            strategy_id = int(torch.argmax(strategy_logits, dim=1).item())
            
            # Extract modulations from dictionary
            # modulations_dict contains: gain_scale, lateral_compliance, step_height_bias
            gain_scale = float(modulations_dict["gain_scale"][0, 0].item())
            compliance = float(modulations_dict["lateral_compliance"][0, 0].item())  # Already sigmoid'd
            step_height_bias = float(modulations_dict["step_height_bias"][0, 0].item())  # Already tanh'd
            
            # Create bias vector (repeat step_height_bias for action space size)
            bias = [step_height_bias] * len(baseline_action)
        timings["policy_inference"] = (time.time() - t0) * 1000.0
        
        # Strategy names
        strategy_names = ["NORMAL", "HIGH_DAMPING", "RECOVERY_BALANCE", "COMPLIANT_TERRAIN"]
        strategy_name = strategy_names[strategy_id] if strategy_id < len(strategy_names) else "UNKNOWN"
        
        # Base modulations from v8 policy
        base_modulations = {
            'gain_scale': gain_scale,
            'lateral_compliance': compliance,
            'step_height_bias': step_height_bias
        }
        
        # Apply adaptive memory adjustments (learns patterns over time)
        # Can be disabled via environment variable: EDON_DISABLE_ADAPTIVE_MEMORY=1
        t0 = time.time()
        use_adaptive_memory = os.getenv("EDON_DISABLE_ADAPTIVE_MEMORY", "0") != "1"
        
        if not use_adaptive_memory:
            logger.debug("[ROBOT-STABILITY] Adaptive memory DISABLED (EDON_DISABLE_ADAPTIVE_MEMORY=1) - using base v8 policy only")
        
        if use_adaptive_memory:
            try:
                memory = get_robot_stability_memory()
                adaptive_modulations = memory.compute_adaptive_modulations(
                    strategy_id=strategy_id,
                    base_modulations=base_modulations,
                    fail_risk=fail_risk,
                    robot_state=obs
                )
                
                # Use adaptive modulations
                gain_scale = adaptive_modulations['gain_scale']
                compliance = adaptive_modulations['lateral_compliance']
                # step_height_bias stays the same (not adjusted by memory yet)
                
                logger.debug(
                    f"[ROBOT-STABILITY] Adaptive memory applied: "
                    f"strategy={strategy_name}, "
                    f"base_gain={base_modulations['gain_scale']:.3f} -> {gain_scale:.3f}, "
                    f"base_compliance={base_modulations['lateral_compliance']:.3f} -> {compliance:.3f}"
                )
            except Exception as e:
                logger.warning(f"[ROBOT-STABILITY] Adaptive memory not available: {e}, using base modulations")
                # Fallback to base modulations if memory fails
        timings["adaptive_memory"] = (time.time() - t0) * 1000.0
        
        # Cache the result for future similar requests
        t0 = time.time()
        _set_cached_strategy(cache_key, {
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "gain_scale": float(np.clip(gain_scale, 0.5, 2.0)),
            "compliance": float(np.clip(compliance, 0.0, 1.0)),
            "bias": bias
        })
        timings["cache_store"] = (time.time() - t0) * 1000.0
        
        # Compute total latency
        timings["total"] = (time.time() - start_time) * 1000.0
        
        # Log timing breakdown (only for slow requests or periodically)
        if timings["total"] > 100.0 or _profile_counter % 50 == 0:
            logger.info(
                f"[ROBOT-STABILITY] Timing breakdown (total={timings['total']:.2f}ms): "
                f"cache_check={timings['cache_check']:.2f}ms, "
                f"fail_risk={timings['fail_risk']:.2f}ms, "
                f"baseline_action={timings['baseline_action']:.2f}ms, "
                f"instability_score={timings['instability_score']:.2f}ms, "
                f"pack_observation={timings['pack_observation']:.2f}ms, "
                f"policy_inference={timings['policy_inference']:.2f}ms, "
                f"adaptive_memory={timings['adaptive_memory']:.2f}ms, "
                f"cache_store={timings['cache_store']:.2f}ms"
            )
        
        # Prepare response
        response = RobotStabilityResponse(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            modulations=Modulations(
                gain_scale=float(np.clip(gain_scale, 0.5, 2.0)),
                compliance=float(np.clip(compliance, 0.0, 1.0)),
                bias=bias
            ),
            intervention_risk=fail_risk,
            latency_ms=timings["total"]
        )
        
        # Stop profiling and log results
        if profiler is not None:
            profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            logger.info(f"[ROBOT-STABILITY] Profiling results (request #{_profile_counter}):\n{s.getvalue()}")
        
        return response
        
    except Exception as e:
        logger.error(f"[ROBOT-STABILITY] Error processing request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing robot stability request: {str(e)}"
        )


@router.post("/record-outcome")
async def record_intervention_outcome(req: RecordOutcomeRequest):
    """
    Record intervention outcome for adaptive learning.
    
    This endpoint is called after each control step to record whether
    an intervention occurred, allowing the adaptive memory to learn
    which strategies and modulations work best.
    
    Database writes are done asynchronously to avoid blocking the response.
    """
    try:
        memory = get_robot_stability_memory()
        
        modulations = {
            'gain_scale': req.gain_scale,
            'lateral_compliance': req.lateral_compliance,
            'step_height_bias': req.step_height_bias
        }
        
        robot_state_dict = None
        if req.robot_state:
            robot_state_dict = _robot_state_to_dict(req.robot_state)
        
        # Run database write in thread pool to avoid blocking
        # This allows the response to return immediately while DB write happens in background
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,  # Use default thread pool
            memory.record_intervention,
            req.strategy_id,
            modulations,
            req.intervention_occurred,
            req.fail_risk,
            robot_state_dict
        )
        
        return {"status": "recorded"}
    except Exception as e:
        logger.error(f"[ROBOT-STABILITY] Error recording outcome: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error recording intervention outcome: {str(e)}"
        )


@router.get("/memory-summary")
async def get_memory_summary():
    """Get summary of learned patterns from adaptive memory."""
    try:
        memory = get_robot_stability_memory()
        summary = memory.get_summary()
        return summary
    except Exception as e:
        logger.error(f"[ROBOT-STABILITY] Error getting memory summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting memory summary: {str(e)}"
        )


@router.post("/memory/clear")
async def clear_memory():
    """Clear adaptive memory database (for fresh start)."""
    try:
        memory = get_robot_stability_memory()
        memory.clear()
        logger.info("[ROBOT-STABILITY] Adaptive memory cleared")
        return {"status": "cleared", "message": "Adaptive memory database cleared. Will start fresh with base v8 policy."}
    except Exception as e:
        logger.error(f"[ROBOT-STABILITY] Error clearing memory: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing memory: {str(e)}"
        )

