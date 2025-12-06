"""
EDON Humanoid Environment Wrapper

Wraps the existing humanoid evaluation environment to expose a gym-like
interface with EDON reward functions for reinforcement learning.

This wrapper:
- Exposes reset() -> obs
- Exposes step(action) -> (next_obs, reward, done, info)
- Uses EDON reward function from training.edon_score
- Maintains compatibility with existing evaluation infrastructure
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.mock_env import MockHumanoidEnv
from training.edon_score import step_reward

# Import baseline_controller
try:
    from run_eval import baseline_controller
except ImportError:
    # Fallback: define a simple baseline controller
    def baseline_controller(obs: dict, edon_state=None) -> np.ndarray:
        """Simple baseline controller that returns zeros."""
        return np.zeros(10, dtype=np.float32)


class EdonHumanoidEnv:
    """
    Wrapper around humanoid environment with EDON rewards.
    
    This class wraps the existing humanoid evaluation environment
    (e.g., MockHumanoidEnv) and adds EDON-specific reward computation.
    """
    
    def __init__(
        self,
        base_env: Optional[Any] = None,
        seed: Optional[int] = None,
        profile: Optional[str] = None,
        w_intervention: float = 20.0,
        w_stability: float = 1.0,
        w_torque: float = 0.1
    ):
        """
        Initialize EDON humanoid environment.
        
        Args:
            base_env: Base environment to wrap (if None, creates MockHumanoidEnv)
            seed: Random seed for environment
            profile: Stress profile (e.g., "high_stress")
            w_intervention: Weight for intervention penalty (default: 20.0)
            w_stability: Weight for stability penalty (default: 1.0)
            w_torque: Weight for torque/action penalty (default: 0.1)
        """
        if base_env is None:
            # Import here to avoid circular dependencies
            try:
                from run_eval import make_humanoid_env
                self.env = make_humanoid_env(seed=seed, profile=profile)
            except ImportError:
                # Fallback: create MockHumanoidEnv directly
                self.env = MockHumanoidEnv(seed=seed)
                if profile:
                    self.env.stress_profile = profile
        else:
            self.env = base_env
        
        self.seed = seed
        self.profile = profile
        self.step_count = 0
        self.episode_count = 0
        self.prev_obs: Optional[Dict[str, Any]] = None
        
        # Reward weights
        self.w_intervention = w_intervention
        self.w_stability = w_stability
        self.w_torque = w_torque
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset environment and return initial observation.
        
        Returns:
            Initial observation dict
        """
        obs = self.env.reset()
        self.step_count = 0
        self.episode_count += 1
        self.prev_obs = obs
        self.last_baseline_action = None
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Step environment with action and return EDON reward.
        
        Args:
            action: Control action array
        
        Returns:
            (next_obs, reward, done, info)
            - next_obs: Next observation dict
            - reward: EDON step reward (from training.edon_score.step_reward)
            - done: Whether episode is done
            - info: Step info dict (may include intervention flags)
        """
        # Step the base environment
        next_obs, env_reward, done, info = self.env.step(action)
        
        # Compute EDON delta (action - baseline) for reward penalty
        # Use stored baseline_action from collect_trajectory
        if self.last_baseline_action is not None:
            baseline_action = self.last_baseline_action
        else:
            # Fallback: compute baseline from current obs
            baseline_action = baseline_controller(next_obs, edon_state=None)
            baseline_action = np.array(baseline_action)
        
        edon_delta = action - np.array(baseline_action)
        info["edon_delta"] = edon_delta
        
        # Compute EDON reward (includes action magnitude penalty)
        edon_reward = step_reward(
            prev_state=self.prev_obs,
            next_state=next_obs,
            info=info,
            w_intervention=self.w_intervention,
            w_stability=self.w_stability,
            w_torque=self.w_torque,
            return_components=False
        )
        
        # Store reward components for diagnostics (if needed)
        if not hasattr(self, '_reward_components'):
            self._reward_components = []
        
        # Update tracking
        self.step_count += 1
        self.prev_obs = next_obs
        
        # Add EDON-specific info
        info["edon_reward"] = edon_reward
        info["step_count"] = self.step_count
        info["episode_count"] = self.episode_count
        
        return next_obs, edon_reward, done, info
    
    def render(self) -> None:
        """Render environment (delegates to base env)."""
        if hasattr(self.env, "render"):
            self.env.render()
    
    def close(self) -> None:
        """Close environment (delegates to base env)."""
        if hasattr(self.env, "close"):
            self.env.close()
    
    @property
    def observation_space(self) -> Dict[str, Any]:
        """Get observation space (delegates to base env if available)."""
        if hasattr(self.env, "observation_space"):
            return self.env.observation_space
        # Default: return dict with common keys
        return {
            "roll": float,
            "pitch": float,
            "com_x": float,
            "com_y": float,
        }
    
    @property
    def action_space(self) -> Dict[str, Any]:
        """Get action space (delegates to base env if available)."""
        if hasattr(self.env, "action_space"):
            return self.env.action_space
        # Default: return array shape
        return {"shape": (10,), "dtype": float}

