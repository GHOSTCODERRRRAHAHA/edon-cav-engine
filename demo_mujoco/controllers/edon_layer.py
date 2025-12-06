"""EDON stabilization layer for humanoid control."""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import sys
import os
import torch

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Add parent directory to path to import EDON client
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from sdk.python.edon.client import EdonClient
    EDON_AVAILABLE = True
except ImportError:
    EDON_AVAILABLE = False
    print("Warning: EDON client not available. EDON layer will be disabled.")


class EdonLayer:
    """
    EDON stabilization layer that modulates baseline control.
    
    Reads robot state, calls EDON API, and applies modulations
    to the baseline controller output.
    """
    
    def __init__(
        self,
        edon_base_url: str = "http://localhost:8000",
        enabled: bool = True,
        call_frequency: int = 1,  # Call EDON every N steps (1 = every step, 2 = every other step, etc.)
        mode: str = "zero-shot",  # "zero-shot" or "trained"
        trained_model_path: Optional[str] = None  # Path to trained model for "trained" mode
    ):
        """
        Initialize EDON layer.
        
        Args:
            edon_base_url: Base URL for EDON API server
            enabled: Whether EDON is enabled (can be toggled)
            call_frequency: Call EDON API every N steps (default: 1 = every step)
                           Higher values reduce API calls but may reduce responsiveness
            mode: "zero-shot" (uses base v8 policy via API) or "trained" (uses trained model)
            trained_model_path: Path to trained model file (required for "trained" mode)
        """
        self.enabled = enabled
        self.edon_base_url = edon_base_url
        self.mode = mode
        self.trained_model_path = trained_model_path
        self.client = None
        self.history: List[Dict[str, Any]] = []
        self.max_history = 8
        self.call_frequency = call_frequency
        self._step_count = 0
        self._last_modulations = {"gain_scale": 1.0, "lateral_compliance": 1.0, "step_height_bias": 0.0}
        
        # Trained model (for "trained" mode)
        self.trained_policy = None
        self.obs_history = deque(maxlen=8)
        
        # Track last step for recording intervention outcomes
        self._last_strategy_id: Optional[int] = None
        self._last_modulations: Optional[Dict[str, float]] = None
        self._last_fail_risk: Optional[float] = None
        self._last_robot_state: Optional[Dict[str, Any]] = None
        
        # Modulation smoothing (prevent rapid changes that cause oscillation)
        self._modulation_history = deque(maxlen=5)  # Last 5 modulations for smoothing
        self._smoothing_alpha = 0.7  # Exponential smoothing factor (0.7 = 70% new, 30% old)
        
        # Modulation smoothing (prevent rapid changes that cause oscillation)
        self._modulation_history = deque(maxlen=5)  # Last 5 modulations for smoothing
        self._smoothing_alpha = 0.7  # Exponential smoothing factor (0.7 = 70% new, 30% old)
        
        # Load trained model if in trained mode
        if mode == "trained" and trained_model_path:
            self._load_trained_model(trained_model_path)
        
        if enabled:
            if mode == "trained":
                if trained_model_path:
                    print(f"✅ EDON will use trained model: {trained_model_path}")
                else:
                    print(f"⚠️  Warning: Trained mode requires trained_model_path")
                    self.enabled = False
            elif mode == "zero-shot" and EDON_AVAILABLE:
                try:
                    # Use shorter timeout for faster failure if server is slow
                    # But still allow enough time for normal API calls (~0.1-0.2s)
                    self.client = EdonClient(base_url=edon_base_url, timeout=2.0)
                    # Test connection
                    health = self.client.health()
                    v8_available = health.get("v8_robot_stability", {}).get("available", False)
                    if not v8_available:
                        print(f"⚠️  Warning: EDON robot stability API not available")
                        print(f"   Health response: {health}")
                        self.enabled = False
                    else:
                        print(f"✅ EDON robot stability API connected at {edon_base_url}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not connect to EDON server at {edon_base_url}: {e}")
                    print(f"   Make sure EDON server is running: python -m app.main")
                    self.enabled = False
            elif not EDON_AVAILABLE and mode == "zero-shot":
                print("⚠️  EDON client not available (sdk not found)")
                self.enabled = False
    
    def _load_trained_model(self, model_path: str):
        """Load trained policy model for trained mode."""
        try:
            # Import policy network
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            from training.edon_v8_policy import EdonV8StrategyPolicy, pack_observation_v8
            
            # Infer input size (will be set when we get first observation)
            # For now, use a default that matches the training setup
            # The actual size will be inferred from the first observation
            self.pack_observation_v8 = pack_observation_v8
            
            # Create policy network (input_size will be set on first observation)
            # We'll create it lazily in step() when we have the first observation
            self.trained_model_path = model_path
            self._policy_ready = False
            print(f"  [EDON] Trained model path set: {model_path}")
            print(f"  [EDON] Policy will be loaded on first observation")
        except Exception as e:
            print(f"⚠️  Warning: Could not load trained model: {e}")
            print(f"   Falling back to zero-shot mode")
            self.mode = "zero-shot"
            self.trained_policy = None
    
    def _ensure_policy_loaded(self, first_obs: Dict[str, Any]):
        """Ensure trained policy is loaded (lazy loading on first observation)."""
        if self._policy_ready:
            return
        
        try:
            from training.edon_v8_policy import EdonV8StrategyPolicy, pack_observation_v8
            
            # Build observation history for stacking
            for _ in range(8):
                self.obs_history.append(first_obs)
            
            # Pack observation to get input size
            stacked_obs = pack_observation_v8(
                obs=first_obs,
                baseline_action=np.zeros(12),  # Dummy baseline
                fail_risk=0.5,
                instability_score=0.0,
                phase="stable",
                obs_history=list(self.obs_history),
                near_fail_history=[],
                obs_vec_history=[],
                stack_size=8
            )
            input_size = len(stacked_obs)
            
            # Create and load policy
            self.trained_policy = EdonV8StrategyPolicy(input_size=input_size)
            self.trained_policy.load_state_dict(torch.load(self.trained_model_path, map_location='cpu'))
            self.trained_policy.eval()
            self._policy_ready = True
            print(f"✅ Trained policy loaded: {input_size} input features")
        except Exception as e:
            print(f"⚠️  Warning: Could not load trained policy: {e}")
            self.mode = "zero-shot"
            self.trained_policy = None
    
    def step(
        self,
        state: Dict[str, Any],
        baseline_action: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply EDON stabilization to baseline action.
        
        Args:
            state: Current robot state observation
            baseline_action: Baseline controller output (MuJoCo torques in [-20, 20] range)
        
        Returns:
            (corrected_action, edon_info)
        """
        if not self.enabled or self.client is None:
            return baseline_action, {"enabled": False}
        
        # Track step count for call frequency
        self._step_count += 1
        
        # Only call EDON API every N steps to reduce latency
        # Use cached modulations for steps between API calls
        should_call_api = (self._step_count % self.call_frequency == 0)
        
        if should_call_api:
            try:
                # Build robot state for EDON API
                robot_state = {
                    "roll": state["roll"],
                    "pitch": state["pitch"],
                    "roll_velocity": state["roll_velocity"],
                    "pitch_velocity": state["pitch_velocity"],
                    "com_x": state.get("com_x", 0.0),
                    "com_y": state.get("com_y", 0.0),
                }
                
                # Get EDON stability control
                stability = self.client.robot_stability(
                    robot_state,
                    history=self.history[-self.max_history:] if self.history else None
                )
                
                # Update history
                self.history.append(robot_state)
                if len(self.history) > self.max_history:
                    self.history.pop(0)
                
                # Extract modulations - match original successful v8 implementation
                modulations = stability.get("modulations", {})
                gain_scale = modulations.get("gain_scale", 1.0)
                
                # Original v8 uses "lateral_compliance" not "compliance"
                lateral_compliance = modulations.get("lateral_compliance", modulations.get("compliance", 1.0))
                
                # Original v8 uses "step_height_bias" (scalar) not "bias" (vector)
                # If bias is provided as vector, extract first value as step_height_bias
                bias_raw = modulations.get("bias", [0.0])
                if isinstance(bias_raw, list) and len(bias_raw) > 0:
                    step_height_bias = float(bias_raw[0]) if isinstance(bias_raw[0], (int, float)) else 0.0
                else:
                    step_height_bias = modulations.get("step_height_bias", 0.0)
                
                # Apply state-aware bounds and smoothing to prevent wrong modulations
                gain_scale, lateral_compliance, step_height_bias = self._apply_modulation_fixes(
                    gain_scale, lateral_compliance, step_height_bias, robot_state
                )
                
                # Cache modulations for use in steps between API calls
                self._last_modulations = {
                    "gain_scale": gain_scale,
                    "lateral_compliance": lateral_compliance,
                    "step_height_bias": step_height_bias
                }
                
                # Store for recording intervention outcomes
                self._last_strategy_id = stability.get("strategy_id", -1)
                self._last_fail_risk = stability.get("intervention_risk", 0.0)
                self._last_robot_state = robot_state.copy()
            except Exception as e:
                # If API call fails, use cached modulations
                print(f"  [EDON] API call failed at step {self._step_count}: {e}, using cached modulations")
                gain_scale = self._last_modulations["gain_scale"]
                lateral_compliance = self._last_modulations["lateral_compliance"]
                step_height_bias = self._last_modulations["step_height_bias"]
        else:
            # Use cached modulations (no API call this step)
            gain_scale = self._last_modulations["gain_scale"]
            lateral_compliance = self._last_modulations["lateral_compliance"]
            step_height_bias = self._last_modulations["step_height_bias"]
        
        # CRITICAL: Normalize MuJoCo actions to [-1, 1] range before applying EDON
        # EDON was trained on normalized actions, so we must match that scale
        # The baseline controller now outputs [-20, 20] (scaled from original [-1, 1])
        # So we normalize back to [-1, 1] for EDON
        action_scale = 20.0  # MuJoCo action range is [-20, 20]
        normalized_action = baseline_action / action_scale
        
        # Apply modulations exactly like training script (matches train_edon_mujoco.py)
        # This ensures zero-shot and trained modes use the same logic
        corrected_normalized = normalized_action.copy()
        
        # Apply state-aware modulation fixes (if not already applied from API call)
        if not should_call_api:
            gain_scale, lateral_compliance, step_height_bias = self._apply_modulation_fixes(
                gain_scale, lateral_compliance, step_height_bias, state
            )
        
        # 1. Apply gain_scale to entire action
        corrected_normalized = corrected_normalized * gain_scale
        
        # 2. Apply lateral_compliance to root rotation (indices 3-5: roll/pitch/yaw)
        # This matches the training script and MuJoCo's action space structure
        # Root rotation controls lateral stability (roll/pitch)
        if len(corrected_normalized) >= 6:
            corrected_normalized[3:6] = corrected_normalized[3:6] * lateral_compliance
        
        # 3. Apply step_height_bias to leg joints (indices 6-11: legs)
        # This matches the training script and MuJoCo's action space structure
        # Leg joints control step height (knees, hips)
        
        if len(corrected_normalized) >= 12:
            corrected_normalized[6:12] = corrected_normalized[6:12] + step_height_bias * 0.1
        elif len(corrected_normalized) >= 8:
            # Fallback if we have fewer than 12 actions
            corrected_normalized[6:8] = corrected_normalized[6:8] + step_height_bias * 0.1
        
        # Clamp to [-1, 1] range (as in original v8)
        corrected_normalized = np.clip(corrected_normalized, -1.0, 1.0)
        
        # Scale back to MuJoCo action range [-20, 20]
        corrected_action = corrected_normalized * action_scale
        
        # Build info dict (use cached values if API wasn't called this step)
        if should_call_api and 'stability' in locals():
            # We have fresh stability data from API call
            edon_info = {
                "enabled": True,
                "strategy_id": stability.get("strategy_id", -1),
                "strategy_name": stability.get("strategy_name", "unknown"),
                "intervention_risk": stability.get("intervention_risk", 0.0),
                "latency_ms": stability.get("latency_ms", 0.0),
                "gain_scale": gain_scale,
                "lateral_compliance": lateral_compliance,
                "step_height_bias": step_height_bias,
            }
        else:
            # Using cached modulations, no fresh API data
            edon_info = {
                "enabled": True,
                "strategy_id": getattr(self, '_last_strategy_id', -1),
                "strategy_name": "cached",
                "intervention_risk": getattr(self, '_last_fail_risk', 0.0),
                "latency_ms": 0.0,
                "gain_scale": gain_scale,
                "lateral_compliance": lateral_compliance,
                "step_height_bias": step_height_bias,
            }
        
        return corrected_action, edon_info
    
    def _step_trained(self, state: Dict[str, Any], baseline_action: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Step using trained policy model (no API calls)."""
        # Ensure policy is loaded (lazy loading)
        if not self._policy_ready:
            self._ensure_policy_loaded(state)
            if not self._policy_ready:
                # Fallback to baseline if policy loading failed
                return baseline_action, {"enabled": False, "mode": "trained", "error": "policy_not_loaded"}
        
        # Update observation history
        self.obs_history.append(state)
        
        # Pack stacked observation
        try:
            from training.edon_v8_policy import pack_observation_v8
            stacked_obs = pack_observation_v8(
                obs=state,
                baseline_action=baseline_action,
                fail_risk=0.5,  # Could compute from state if needed
                instability_score=0.0,
                phase="stable",
                obs_history=list(self.obs_history),
                near_fail_history=[],
                obs_vec_history=[],
                stack_size=8
            )
        except Exception as e:
            print(f"  [EDON] Error packing observation: {e}")
            return baseline_action, {"enabled": False, "mode": "trained", "error": str(e)}
        
        # Get strategy and modulations from trained policy
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(stacked_obs).unsqueeze(0)
            strategy_logits, gain_scale_mean, lateral_compliance_mean = self.trained_policy(obs_tensor)
            
            # Sample strategy
            strategy_dist = torch.distributions.Categorical(logits=strategy_logits)
            strategy_id = strategy_dist.sample().item()
            
            # Get modulations
            gain_scale = float(torch.sigmoid(gain_scale_mean).item() * 1.0 + 0.5)  # [0.5, 1.5]
            lateral_compliance = float(torch.sigmoid(lateral_compliance_mean).item())  # [0, 1]
            step_height_bias = 0.0  # Could be added to policy if needed
        
        modulations = {
            "gain_scale": gain_scale,
            "lateral_compliance": lateral_compliance,
            "step_height_bias": step_height_bias
        }
        
        # Apply modulations (same as zero-shot mode)
        action_scale = 20.0
        normalized_action = baseline_action / action_scale
        corrected_normalized = normalized_action.copy()
        
        # Apply gain_scale
        gain_scale = np.clip(gain_scale, 0.5, 1.5)
        corrected_normalized = corrected_normalized * gain_scale
        
        # Apply lateral_compliance
        if len(corrected_normalized) >= 4:
            corrected_normalized[:4] = corrected_normalized[:4] * lateral_compliance
        
        # Apply step_height_bias
        if len(corrected_normalized) >= 8:
            corrected_normalized[4:8] = corrected_normalized[4:8] + step_height_bias * 0.1
        
        # Clamp and scale back
        corrected_normalized = np.clip(corrected_normalized, -1.0, 1.0)
        corrected_action = corrected_normalized * action_scale
        
        # Build info
        strategy_names = ["NORMAL", "HIGH_DAMPING", "RECOVERY_BALANCE", "COMPLIANT_TERRAIN"]
        edon_info = {
            "enabled": True,
            "mode": "trained",
            "strategy_id": strategy_id,
            "strategy_name": strategy_names[strategy_id] if strategy_id < len(strategy_names) else "unknown",
            "intervention_risk": 0.5,  # Could compute from state if needed
            "latency_ms": 0.0,  # No API call
            "gain_scale": gain_scale,
            "lateral_compliance": lateral_compliance,
            "step_height_bias": step_height_bias,
        }
        
        return corrected_action, edon_info
    
    def set_enabled(self, enabled: bool):
        """Enable or disable EDON layer."""
        self.enabled = enabled
    
    def record_intervention_outcome(self, intervention_occurred: bool):
        """
        Record intervention outcome for adaptive learning.
        
        Args:
            intervention_occurred: True if intervention happened, False if stable
        """
        if not self.enabled or self.client is None:
            return
        
        if self._last_strategy_id is None or self._last_modulations is None:
            return  # No data to record
        
        if not REQUESTS_AVAILABLE:
            return  # Can't record without requests library
        
        try:
            # Call EDON API to record outcome
            record_url = f"{self.edon_base_url}/oem/robot/stability/record-outcome"
            
            robot_state_data = None
            if self._last_robot_state:
                robot_state_data = {
                    "roll": self._last_robot_state.get("roll", 0.0),
                    "pitch": self._last_robot_state.get("pitch", 0.0),
                    "roll_velocity": self._last_robot_state.get("roll_velocity", 0.0),
                    "pitch_velocity": self._last_robot_state.get("pitch_velocity", 0.0),
                    "com_x": self._last_robot_state.get("com_x", 0.0),
                    "com_y": self._last_robot_state.get("com_y", 0.0),
                }
            
            payload = {
                "strategy_id": self._last_strategy_id,
                "gain_scale": self._last_modulations["gain_scale"],
                "lateral_compliance": self._last_modulations["lateral_compliance"],
                "step_height_bias": self._last_modulations["step_height_bias"],
                "intervention_occurred": intervention_occurred,
                "fail_risk": self._last_fail_risk or 0.0,
                "robot_state": robot_state_data
            }
            
            # Use very short timeout and make it truly non-blocking
            # This prevents blocking the simulation thread
            # Use timeout=0.05s (50ms) - if server is slow, just skip recording
            response = requests.post(record_url, json=payload, timeout=0.05)
            # Don't raise on error - recording is best-effort
            if response.status_code != 200:
                pass  # Silently fail - recording is optional
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            # Timeout or connection error - silently skip (non-blocking)
            # This is expected if server is busy - don't block simulation
            pass
        except Exception:
            # Silently fail - recording is optional and shouldn't break control loop
            pass
    
    def _apply_modulation_fixes(
        self,
        gain_scale: float,
        lateral_compliance: float,
        step_height_bias: float,
        robot_state: Dict[str, Any]
    ) -> Tuple[float, float, float]:
        """
        Apply fixes to modulations to prevent wrong values from destabilizing robot.
        
        Fixes:
        1. State-aware bounds (adjust based on robot stability)
        2. Modulation smoothing (prevent rapid changes)
        3. Fail-risk based adjustments (if fail-risk seems wrong, adjust modulations)
        """
        roll = abs(robot_state.get("roll", 0.0))
        pitch = abs(robot_state.get("pitch", 0.0))
        max_tilt = max(roll, pitch)
        fail_risk = self._last_fail_risk or 0.0
        
        # Fix 1: State-aware bounds for gain_scale
        # If robot is unstable, allow more aggressive corrections
        if max_tilt > 0.2:  # ~11.5 degrees - unstable
            gain_scale_min, gain_scale_max = 0.7, 1.4
        elif max_tilt > 0.1:  # ~5.7 degrees - moderate
            gain_scale_min, gain_scale_max = 0.6, 1.3
        else:
            gain_scale_min, gain_scale_max = 0.5, 1.2  # Conservative for stable
        
        # Fix 2: Adjust gain_scale based on fail-risk
        # If fail-risk is high but robot is stable, reduce gain_scale (fail-risk might be wrong)
        if fail_risk > 0.7 and max_tilt < 0.1:
            gain_scale = gain_scale * 0.9  # Reduce by 10% - fail-risk might be overestimating
        # If fail-risk is low but robot is unstable, increase gain_scale (fail-risk might be wrong)
        elif fail_risk < 0.3 and max_tilt > 0.15:
            gain_scale = gain_scale * 1.1  # Increase by 10% - fail-risk might be underestimating
        
        gain_scale = np.clip(gain_scale, gain_scale_min, gain_scale_max)
        
        # Fix 3: State-aware bounds for lateral_compliance
        # If robot is unstable, need stronger root rotation
        if max_tilt > 0.2:  # Unstable
            lateral_compliance_min, lateral_compliance_max = 0.6, 1.0
        elif max_tilt > 0.1:  # Moderate
            lateral_compliance_min, lateral_compliance_max = 0.5, 1.0
        else:
            lateral_compliance_min, lateral_compliance_max = 0.4, 0.9  # Conservative
        
        lateral_compliance = np.clip(lateral_compliance, lateral_compliance_min, lateral_compliance_max)
        
        # Fix 4: More conservative bounds for step_height_bias
        step_height_bias = np.clip(step_height_bias, -0.5, 0.5)  # Limit to ±0.5
        
        # Fix 5: Modulation smoothing (prevent rapid changes that cause oscillation)
        if len(self._modulation_history) > 0:
            last_mods = self._modulation_history[-1]
            # Exponential smoothing: 70% new value, 30% old value
            gain_scale = self._smoothing_alpha * gain_scale + (1 - self._smoothing_alpha) * last_mods["gain_scale"]
            lateral_compliance = self._smoothing_alpha * lateral_compliance + (1 - self._smoothing_alpha) * last_mods["lateral_compliance"]
            step_height_bias = self._smoothing_alpha * step_height_bias + (1 - self._smoothing_alpha) * last_mods["step_height_bias"]
        
        # Store in history for next smoothing
        self._modulation_history.append({
            "gain_scale": gain_scale,
            "lateral_compliance": lateral_compliance,
            "step_height_bias": step_height_bias
        })
        
        return gain_scale, lateral_compliance, step_height_bias
    
    def reset(self):
        """Reset EDON layer state (clear history)."""
        self.history = []
        self._last_strategy_id = None
        self._last_modulations = None
        self._last_fail_risk = None
        self._last_robot_state = None
        self._modulation_history.clear()

