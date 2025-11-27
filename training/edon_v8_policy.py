"""
EDON v8 Strategy Policy

Learned policy that outputs strategy selection and modulation signals
(not raw action deltas). Works with reflex layer for layered control.

NOW WITH:
- Early-warning features (rolling variance, oscillation energy, near-fail density)
- Stacked observations (last K=8 frames for temporal memory)
"""

from typing import Dict, Any, List, Tuple, Optional, TYPE_CHECKING
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from collections import deque

if TYPE_CHECKING:
    from env.edon_humanoid_env_v8 import EdonHumanoidEnvV8


class EdonV8StrategyPolicy(nn.Module):
    """
    EDON v8 Strategy Policy Network.
    
    Outputs:
    - strategy_id: Discrete strategy selection (NORMAL, HIGH_DAMPING, RECOVERY_BALANCE, etc.)
    - modulations: Continuous modulation signals (gain_scale, lateral_compliance, step_height_bias)
    """
    
    STRATEGIES = ["NORMAL", "HIGH_DAMPING", "RECOVERY_BALANCE", "COMPLIANT_TERRAIN"]
    N_STRATEGIES = len(STRATEGIES)
    
    def __init__(
        self,
        input_size: Optional[int] = None,
        env: Optional[Any] = None,
        hidden_sizes: List[int] = [128, 128, 64],
        max_gain_scale: float = 1.5,
        min_gain_scale: float = 0.5
    ):
        """
        Initialize v8 strategy policy.
        
        Args:
            input_size: Size of input feature vector (should be base_features_size * 8 for stacked obs).
                       If None, will be inferred from env.
            env: Environment to infer input_size from (required if input_size is None)
            hidden_sizes: Hidden layer sizes
            max_gain_scale: Maximum gain scale (default: 1.5)
            min_gain_scale: Minimum gain scale (default: 0.5)
        """
        super().__init__()
        
        # Infer input_size from env if not provided
        if input_size is None:
            if env is None:
                raise ValueError("Either input_size or env must be provided")
            input_size = self._infer_obs_dim_from_env(env)
        
        self.input_size = input_size
        
        # Shared feature extraction network
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.feature_net = nn.Sequential(*layers)
        
        # Strategy head (discrete selection)
        self.strategy_head = nn.Linear(prev_size, self.N_STRATEGIES)
        
        # Modulation heads (continuous outputs)
        self.gain_scale_head = nn.Linear(prev_size, 1)  # Output ∈ [min_gain_scale, max_gain_scale]
        self.lateral_compliance_head = nn.Linear(prev_size, 1)  # Output ∈ [0, 1]
        self.step_height_bias_head = nn.Linear(prev_size, 1)  # Output ∈ [-1, 1]
        
        self.max_gain_scale = max_gain_scale
        self.min_gain_scale = min_gain_scale
    
    @staticmethod
    def _infer_obs_dim_from_env(env: Any) -> int:
        """
        Infer observation dimension from environment by creating a test observation.
        
        Args:
            env: Environment instance (must have reset() method and support pack_stacked_observation_v8)
        
        Returns:
            Observation dimension (input_size)
        """
        # Import here to avoid circular dependencies
        from run_eval import baseline_controller
        
        # Reset environment to get a sample observation
        test_obs = env.reset()
        
        # Get baseline action
        test_baseline = baseline_controller(test_obs, edon_state=None)
        test_baseline = np.array(test_baseline)
        
        # Actually pack the observation using the real packing function
        # This ensures we get the exact size that will be used during training
        # Import the function - it's defined later in this same module but will be available
        # when this method is called (module is fully loaded)
        try:
            # Try to get from current module namespace first
            import sys
            current_module = sys.modules[__name__]
            if hasattr(current_module, 'pack_stacked_observation_v8'):
                pack_stacked_observation_v8 = getattr(current_module, 'pack_stacked_observation_v8')
            else:
                # Import from module (will work since module is loaded)
                from training.edon_v8_policy import pack_stacked_observation_v8
            
            # Pack with stacked observations (8 frames) - this gives us the actual size
            test_input = pack_stacked_observation_v8(
                obs=test_obs,
                baseline_action=test_baseline,
                fail_risk=0.0,
                instability_score=0.0,
                phase="stable",
                obs_history=None,
                near_fail_history=None,
                obs_vec_history=None,
                stack_size=8
            )
            obs_dim = len(test_input)
            return obs_dim
        except (ImportError, AttributeError) as e:
            # If import fails, we need to compute manually
            print(f"[INFER] Warning: Could not import pack_stacked_observation_v8: {e}")
            print(f"[INFER] Computing obs_dim manually...")
            # Manual computation: base features per frame
            # 4 (roll/pitch/vels) + 4 (COM) + 4 (derived) + 3 (risk/phase) + len(baseline) + 6 (early-warning)
            base_size = 4 + 4 + 4 + 3 + len(test_baseline) + 6
            # Stacked: base_size * 8
            manual_dim = base_size * 8
            print(f"[INFER] Manual computation: base_size={base_size}, stacked={manual_dim}")
            return manual_dim
        except Exception as e:
            print(f"[INFER] Error packing observation: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to manual computation
            base_size = 4 + 4 + 4 + 3 + len(test_baseline) + 6
            fallback_dim = base_size * 8
            print(f"[INFER] Using fallback: {fallback_dim}")
            return fallback_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input feature tensor (batch_size, input_size)
                For stacked observations: (batch_size, base_features_size * 8)
        
        Returns:
            (strategy_logits, modulations_dict)
            - strategy_logits: Logits over strategies (batch_size, N_STRATEGIES)
            - modulations_dict: Dict with:
                - gain_scale: (batch_size, 1) ∈ [min_gain_scale, max_gain_scale]
                - lateral_compliance: (batch_size, 1) ∈ [0, 1]
                - step_height_bias: (batch_size, 1) ∈ [-1, 1]
        """
        features = self.feature_net(x)
        
        # Strategy logits
        strategy_logits = self.strategy_head(features)
        
        # Modulations (with appropriate activations)
        gain_scale_raw = self.gain_scale_head(features)
        gain_scale = torch.sigmoid(gain_scale_raw) * (self.max_gain_scale - self.min_gain_scale) + self.min_gain_scale
        
        lateral_compliance = torch.sigmoid(self.lateral_compliance_head(features))
        
        step_height_bias = torch.tanh(self.step_height_bias_head(features))
        
        modulations = {
            "gain_scale": gain_scale,
            "lateral_compliance": lateral_compliance,
            "step_height_bias": step_height_bias
        }
        
        return strategy_logits, modulations
    
    def sample_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[int, Dict[str, float], torch.Tensor]:
        """
        Sample strategy and modulations from policy.
        
        Args:
            obs: Observation tensor (batch_size, input_size) or (input_size,)
                For stacked observations: (batch_size, base_features_size * 8) or (base_features_size * 8,)
            deterministic: If True, use argmax for strategy selection
        
        Returns:
            (strategy_id, modulations_dict, log_prob)
            - strategy_id: Selected strategy index (int or tensor)
            - modulations_dict: Dict with gain_scale, lateral_compliance, step_height_bias (as floats or tensors)
            - log_prob: Log probability of the sampled action
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        strategy_logits, modulations = self.forward(obs)
        
        # Sample strategy
        if deterministic:
            strategy_id = torch.argmax(strategy_logits, dim=-1)
            log_prob_strategy = torch.zeros_like(strategy_id, dtype=torch.float32)
        else:
            strategy_dist = Categorical(logits=strategy_logits)
            strategy_id = strategy_dist.sample()
            log_prob_strategy = strategy_dist.log_prob(strategy_id)
        
        # Extract modulations (already in correct ranges from forward)
        modulations_dict = {
            "gain_scale": modulations["gain_scale"].squeeze(-1).item() if modulations["gain_scale"].numel() == 1 else modulations["gain_scale"].squeeze(-1),
            "lateral_compliance": modulations["lateral_compliance"].squeeze(-1).item() if modulations["lateral_compliance"].numel() == 1 else modulations["lateral_compliance"].squeeze(-1),
            "step_height_bias": modulations["step_height_bias"].squeeze(-1).item() if modulations["step_height_bias"].numel() == 1 else modulations["step_height_bias"].squeeze(-1)
        }
        
        # Total log prob (strategy only, modulations are deterministic given features)
        log_prob = log_prob_strategy
        
        strategy_id_val = strategy_id.item() if strategy_id.numel() == 1 else strategy_id
        
        return strategy_id_val, modulations_dict, log_prob


def pack_observation_v8(
    obs: Dict[str, Any],
    baseline_action: np.ndarray,
    fail_risk: float = 0.0,
    instability_score: float = 0.0,
    phase: str = "stable",
    obs_history: Optional[List[Dict[str, Any]]] = None,
    near_fail_history: Optional[List[bool]] = None
) -> np.ndarray:
    """
    Pack observation into input vector for v8 strategy policy.
    
    Includes:
    - roll, pitch, velocities
    - COM position and velocity
    - tilt_mag, vel_norm
    - fail_risk
    - instability_score
    - phase (encoded)
    - baseline_action (normalized)
    - EARLY-WARNING FEATURES:
      - Rolling variance of roll, pitch, roll_velocity, pitch_velocity
      - High-frequency oscillation energy
      - Near-fail density
    
    Args:
        obs: Observation dict
        baseline_action: Baseline action array
        fail_risk: Predicted failure risk
        instability_score: Current instability score
        phase: Current phase (stable/warning/recovery)
        obs_history: List of last N observation dicts (for rolling variance)
        near_fail_history: List of last N near-fail flags (for near-fail density)
    
    Returns:
        Packed feature vector (single frame)
    """
    # Extract basic state
    roll = float(obs.get("roll", 0.0))
    pitch = float(obs.get("pitch", 0.0))
    roll_velocity = float(obs.get("roll_velocity", 0.0))
    pitch_velocity = float(obs.get("pitch_velocity", 0.0))
    com_x = float(obs.get("com_x", 0.0))
    com_y = float(obs.get("com_y", 0.0))
    com_velocity_x = float(obs.get("com_velocity_x", 0.0))
    com_velocity_y = float(obs.get("com_velocity_y", 0.0))
    
    # Derived features
    tilt_mag = np.sqrt(roll**2 + pitch**2)
    vel_norm = np.sqrt(roll_velocity**2 + pitch_velocity**2)
    com_norm = np.sqrt(com_x**2 + com_y**2)
    com_vel_norm = np.sqrt(com_velocity_x**2 + com_velocity_y**2)
    
    # Phase encoding
    phase_map = {"stable": 0.0, "warning": 1.0, "recovery": 2.0}
    phase_encoded = phase_map.get(phase, 0.0)
    
    # EARLY-WARNING FEATURES (from rolling history)
    early_warning_features = []
    
    if obs_history and len(obs_history) >= 2:
        # Rolling variance of tilt and angular velocities (last N steps)
        roll_values = [float(h.get("roll", 0.0)) for h in obs_history]
        pitch_values = [float(h.get("pitch", 0.0)) for h in obs_history]
        roll_vel_values = [float(h.get("roll_velocity", 0.0)) for h in obs_history]
        pitch_vel_values = [float(h.get("pitch_velocity", 0.0)) for h in obs_history]
        
        roll_var = float(np.var(roll_values)) if len(roll_values) > 1 else 0.0
        pitch_var = float(np.var(pitch_values)) if len(pitch_values) > 1 else 0.0
        roll_vel_var = float(np.var(roll_vel_values)) if len(roll_vel_values) > 1 else 0.0
        pitch_vel_var = float(np.var(pitch_vel_values)) if len(pitch_vel_values) > 1 else 0.0
        
        # High-frequency oscillation energy (mean of squared angular velocities)
        ang_vel_squared = [rv**2 + pv**2 for rv, pv in zip(roll_vel_values, pitch_vel_values)]
        osc_energy = float(np.mean(ang_vel_squared)) if ang_vel_squared else 0.0
        
        early_warning_features = [roll_var, pitch_var, roll_vel_var, pitch_vel_var, osc_energy]
    else:
        # Not enough history yet, use zeros
        early_warning_features = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Near-fail density (rolling count of near-fail events in last N steps)
    if near_fail_history:
        near_fail_count = sum(1 for nf in near_fail_history if nf)
        near_fail_density = float(near_fail_count) / len(near_fail_history) if near_fail_history else 0.0
    else:
        near_fail_density = 0.0
    
    early_warning_features.append(near_fail_density)
    
    # Pack base feature vector (current frame)
    base_features = np.concatenate([
        [roll, pitch, roll_velocity, pitch_velocity],
        [com_x, com_y, com_velocity_x, com_velocity_y],
        [tilt_mag, vel_norm, com_norm, com_vel_norm],
        [fail_risk, instability_score, phase_encoded],
        baseline_action.astype(np.float32),
        early_warning_features  # Add early-warning features
    ])
    
    return base_features


def pack_stacked_observation_v8(
    obs: Dict[str, Any],
    baseline_action: np.ndarray,
    fail_risk: float = 0.0,
    instability_score: float = 0.0,
    phase: str = "stable",
    obs_history: Optional[List[Dict[str, Any]]] = None,
    near_fail_history: Optional[List[bool]] = None,
    obs_vec_history: Optional[List[np.ndarray]] = None,
    stack_size: int = 8
) -> np.ndarray:
    """
    Pack observation with stacked frames for temporal memory.
    
    Concatenates the last K=8 observation vectors to give the policy
    temporal context.
    
    Args:
        obs: Current observation dict
        baseline_action: Current baseline action
        fail_risk: Current fail risk
        instability_score: Current instability score
        phase: Current phase
        obs_history: List of last N observation dicts (for early-warning features)
        near_fail_history: List of last N near-fail flags
        obs_vec_history: List of last K packed observation vectors (for stacking)
        stack_size: Number of frames to stack (default: 8)
    
    Returns:
        Stacked feature vector (base_features_size * stack_size)
    """
    # Pack current frame
    current_frame = pack_observation_v8(
        obs=obs,
        baseline_action=baseline_action,
        fail_risk=fail_risk,
        instability_score=instability_score,
        phase=phase,
        obs_history=obs_history,
        near_fail_history=near_fail_history
    )
    
    # Stack with history
    if obs_vec_history and len(obs_vec_history) > 0:
        # Use last (stack_size - 1) frames from history
        history_frames = list(obs_vec_history)[-(stack_size - 1):]
        # Pad with zeros if not enough history
        while len(history_frames) < (stack_size - 1):
            history_frames.insert(0, np.zeros_like(current_frame))
        # Concatenate: [oldest, ..., newest, current]
        stacked = np.concatenate(history_frames + [current_frame])
    else:
        # Not enough history, pad with zeros
        zero_frame = np.zeros_like(current_frame)
        stacked = np.concatenate([zero_frame] * (stack_size - 1) + [current_frame])
    
    return stacked
