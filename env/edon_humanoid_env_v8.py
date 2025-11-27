"""
EDON v8 Environment Wrapper

Wraps the base humanoid environment and applies v8 layered control:
- Strategy layer (learned policy) outputs strategy + modulations
- Reflex layer applies deterministic adjustments
- Final action is applied to base environment
"""

from typing import Dict, Any, Optional, Tuple
from collections import deque
import numpy as np
import torch

from env.edon_humanoid_env import EdonHumanoidEnv
# REFLEX CONTROLLER DISABLED - proven to destroy stability
# from controllers.edon_v8_reflex import EdonReflexController
from training.edon_v8_policy import EdonV8StrategyPolicy, pack_stacked_observation_v8
from run_eval import baseline_controller


class EdonHumanoidEnvV8(EdonHumanoidEnv):
    """
    v8 environment wrapper that uses layered control.
    
    At each step:
    1. Get baseline action
    2. Get strategy + modulations from learned policy
    3. Apply reflex controller adjustments
    4. Step base environment
    """
    
    def __init__(
        self,
        strategy_policy: EdonV8StrategyPolicy,
        fail_risk_model: Optional[Any] = None,
        base_env: Optional[Any] = None,
        seed: Optional[int] = None,
        profile: Optional[str] = None,
        device: str = "cpu",
        w_intervention: float = 20.0,
        w_stability: float = 1.0,
        w_torque: float = 0.1,
        gain_multiplier: float = 1.0,
        inverted: bool = False
    ):
        """
        Initialize v8 environment.
        
        Args:
            strategy_policy: Trained v8 strategy policy
            fail_risk_model: Optional fail-risk model for computing fail_risk
            base_env: Base environment to wrap
            seed: Random seed
            profile: Stress profile
            device: Device for model inference
            w_intervention: Weight for intervention penalty (default: 20.0)
            w_stability: Weight for stability penalty (default: 1.0)
            w_torque: Weight for torque/action penalty (default: 0.1)
        """
        super().__init__(
            base_env=base_env,
            seed=seed,
            profile=profile,
            w_intervention=w_intervention,
            w_stability=w_stability,
            w_torque=w_torque
        )
        
        self.strategy_policy = strategy_policy
        self.fail_risk_model = fail_risk_model
        self.device = torch.device(device)
        # REFLEX CONTROLLER DISABLED - proven to destroy stability
        # self.reflex_controller = EdonReflexController()
        
        # Track v8-specific info
        self.current_fail_risk = 0.0
        self.current_strategy_id = 0
        self.current_modulations = {}
        self.gain_multiplier = gain_multiplier
        self.inverted = inverted
        
        # Temporal memory buffers for early-warning features and stacked observations
        self.obs_history = deque(maxlen=20)  # For rolling variance (last 20 steps)
        self.obs_vec_history = deque(maxlen=8)  # For stacked observations (last 8 frames)
        self.near_fail_history = deque(maxlen=20)  # For near-fail density
    
    def compute_fail_risk(self, obs: Dict[str, Any], features: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute fail-risk using fail-risk model.
        
        Args:
            obs: Current observation
            features: Optional pre-computed features
        
        Returns:
            fail_risk ∈ [0, 1]
        """
        if self.fail_risk_model is None:
            return 0.0
        
        try:
            # Extract features (same as fail_risk_model.extract_features_from_step)
            roll = float(obs.get("roll", 0.0))
            pitch = float(obs.get("pitch", 0.0))
            roll_velocity = float(obs.get("roll_velocity", 0.0))
            pitch_velocity = float(obs.get("pitch_velocity", 0.0))
            com_x = float(obs.get("com_x", 0.0))
            com_y = float(obs.get("com_y", 0.0))
            com_velocity_x = float(obs.get("com_velocity_x", 0.0))
            com_velocity_y = float(obs.get("com_velocity_y", 0.0))
            
            tilt_mag = np.sqrt(roll**2 + pitch**2)
            vel_norm = np.sqrt(roll_velocity**2 + pitch_velocity**2)
            com_norm = np.sqrt(com_x**2 + com_y**2)
            com_vel_norm = np.sqrt(com_velocity_x**2 + com_velocity_y**2)
            
            # Extract from features dict if available
            if features:
                instability_score = float(features.get("instability_score", 0.0))
                risk_ema = float(features.get("risk_ema", 0.0))
                phase = features.get("phase", "stable")
            else:
                instability_score = 0.0
                risk_ema = 0.0
                phase = "stable"
            
            phase_map = {"stable": 0.0, "warning": 1.0, "recovery": 2.0, "prefall": 3.0, "fail": 4.0}
            phase_encoded = phase_map.get(phase, 0.0)
            
            # Pack feature vector
            feature_vec = np.array([
                roll, pitch, roll_velocity, pitch_velocity,
                com_x, com_y, com_velocity_x, com_velocity_y,
                tilt_mag, vel_norm, com_norm, com_vel_norm,
                instability_score, risk_ema, phase_encoded
            ], dtype=np.float32)
            
            # Run inference
            with torch.no_grad():
                feature_tensor = torch.FloatTensor(feature_vec).unsqueeze(0).to(self.device)
                fail_risk = self.fail_risk_model(feature_tensor).item()
            
            return float(fail_risk)
        except Exception as e:
            # Fallback to 0.0 on error
            return 0.0
    
    def step(
        self,
        action: Optional[np.ndarray] = None,
        edon_core_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Step environment with v8 layered control.
        
        Args:
            action: Ignored (v8 computes action internally)
            edon_core_state: Optional EDON core state for fail-risk computation
        
        Returns:
            (next_obs, reward, done, info)
        """
        # Get current observation
        if self.prev_obs is None:
            obs = self.env.reset()
        else:
            obs = self.prev_obs
        
        # Get baseline action
        baseline_action = baseline_controller(obs, edon_state=None)
        baseline_action = np.array(baseline_action)
        
        # Compute fail-risk
        features = None
        if edon_core_state:
            features = {
                "instability_score": edon_core_state.get("instability_score", 0.0),
                "risk_ema": edon_core_state.get("risk_ema", 0.0),
                "phase": edon_core_state.get("phase", "stable")
            }
        
        fail_risk = self.compute_fail_risk(obs, features)
        self.current_fail_risk = fail_risk
        
        # Pack observation for strategy policy with early-warning features and stacking
        instability_score = features.get("instability_score", 0.0) if features else 0.0
        phase = features.get("phase", "stable") if features else "stable"
        
        # Check if near-fail (fail_risk > threshold)
        near_fail = fail_risk > 0.6
        self.near_fail_history.append(near_fail)
        
        # Pack base frame first (for history storage)
        from training.edon_v8_policy import pack_observation_v8
        base_frame = pack_observation_v8(
            obs=obs,
            baseline_action=baseline_action,
            fail_risk=fail_risk,
            instability_score=instability_score,
            phase=phase,
            obs_history=list(self.obs_history),
            near_fail_history=list(self.near_fail_history)
        )
        
        # Pack with early-warning features and stacking
        obs_vec = pack_stacked_observation_v8(
            obs=obs,
            baseline_action=baseline_action,
            fail_risk=fail_risk,
            instability_score=instability_score,
            phase=phase,
            obs_history=list(self.obs_history),
            near_fail_history=list(self.near_fail_history),
            obs_vec_history=list(self.obs_vec_history),
            stack_size=8
        )
        
        # Update history buffers (before policy call, so history is available)
        # Note: We'll update obs_vec_history again after we use it, but this ensures
        # the history is populated for early-warning feature computation
        if len(self.obs_vec_history) >= 8:
            self.obs_vec_history.popleft()  # Keep only last 8
        if len(self.obs_history) >= 20:
            self.obs_history.popleft()  # Keep only last 20
        if len(self.near_fail_history) >= 20:
            self.near_fail_history.popleft()  # Keep only last 20
        
        # Get strategy + modulations from policy
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_vec).unsqueeze(0).to(self.device)
            strategy_id, modulations, _ = self.strategy_policy.sample_action(obs_tensor, deterministic=False)
        
        self.current_strategy_id = strategy_id
        self.current_modulations = modulations
        
        # Update history buffers AFTER packing (so obs_vec is available for next step)
        # Note: obs_vec_history is updated before this, so we need to update it after packing
        # Actually, we already have obs_vec, so we can update it here
        # But we need to be careful - we want to store the packed vector for next step's stacking
        # So we'll update it after we use it for the current step
        
        # REFLEX CONTROLLER DISABLED - proven to destroy stability
        # Apply strategy modulations directly to baseline action
        final_action = baseline_action.copy()
        
        if modulations:
            gain_scale = modulations.get("gain_scale", 1.0)
            # Apply gain multiplier and inversion for leverage testing
            gain_scale = gain_scale * self.gain_multiplier
            if self.inverted:
                gain_scale = -gain_scale
            final_action = final_action * gain_scale
            
            lateral_compliance = modulations.get("lateral_compliance", 1.0)
            if len(final_action) >= 4:
                final_action[:4] = final_action[:4] * lateral_compliance
            
            step_height_bias = modulations.get("step_height_bias", 0.0)
            if len(final_action) >= 8:
                final_action[4:8] = final_action[4:8] + step_height_bias * 0.1
        
        # Just clip to valid range (minimal safety)
        final_action = np.clip(final_action, -1.0, 1.0)
        
        # Step base environment
        next_obs, reward, done, info = self.env.step(final_action)
        
        # Add v8-specific info
        info["fail_risk"] = fail_risk
        info["strategy_id"] = strategy_id
        info["strategy_name"] = EdonV8StrategyPolicy.STRATEGIES[strategy_id]
        info["gain_scale"] = modulations.get("gain_scale", 1.0)
        info["lateral_compliance"] = modulations.get("lateral_compliance", 1.0)
        info["step_height_bias"] = modulations.get("step_height_bias", 0.0)
        
        # Update state tracking
        self.prev_obs = next_obs
        self.step_count += 1
        
        # Update history buffers for next step (after we've used current obs_vec)
        # Store the BASE frame (not stacked) for stacking in next step
        # base_frame was already computed above, so we can use it
        if len(self.obs_vec_history) >= 8:
            self.obs_vec_history.popleft()
        self.obs_vec_history.append(base_frame.copy() if isinstance(base_frame, np.ndarray) else base_frame)
        
        # Update observation history (for early-warning features)
        if len(self.obs_history) >= 20:
            self.obs_history.popleft()
        self.obs_history.append(next_obs)
        
        return next_obs, reward, done, info
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset environment (inherits from EdonHumanoidEnv which sets prev_obs).
        """
        obs = super().reset()
        self.current_fail_risk = 0.0
        self.current_strategy_id = 0
        self.current_modulations = {}
        
        # Reset temporal buffers
        self.obs_history.clear()
        self.obs_vec_history.clear()
        self.near_fail_history.clear()
        
        # Initialize with current observation
        self.obs_history.append(obs)
        
        return obs

