"""
Improved Fail-Risk Model v2

Uses temporal and energy-based features with XGBoost/LightGBM for better discrimination.
Target: fail-risk separation >= 0.25
"""

import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import deque

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("[WARNING] LightGBM not installed. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARNING] XGBoost not installed. Install with: pip install xgboost")


def extract_temporal_features(
    step_data: Dict[str, Any],
    history: deque,
    window_size: int = 10
) -> np.ndarray:
    """
    Extract temporal features from step data and history.
    
    Features:
    - Rolling variance of state (roll, pitch, velocities)
    - Derivative of error (rate of change)
    - Oscillation frequency (zero-crossing rate)
    """
    features = []
    
    # Current state
    roll = step_data.get("obs", {}).get("roll", 0.0) if isinstance(step_data.get("obs"), dict) else 0.0
    pitch = step_data.get("obs", {}).get("pitch", 0.0) if isinstance(step_data.get("obs"), dict) else 0.0
    roll_vel = step_data.get("obs", {}).get("roll_velocity", 0.0) if isinstance(step_data.get("obs"), dict) else 0.0
    pitch_vel = step_data.get("obs", {}).get("pitch_velocity", 0.0) if isinstance(step_data.get("obs"), dict) else 0.0
    
    # Rolling variance (if we have history)
    if len(history) >= window_size:
        roll_history = [h.get("roll", 0.0) for h in history]
        pitch_history = [h.get("pitch", 0.0) for h in history]
        roll_vel_history = [h.get("roll_velocity", 0.0) for h in history]
        pitch_vel_history = [h.get("pitch_velocity", 0.0) for h in history]
        
        roll_var = np.var(roll_history[-window_size:])
        pitch_var = np.var(pitch_history[-window_size:])
        roll_vel_var = np.var(roll_vel_history[-window_size:])
        pitch_vel_var = np.var(pitch_vel_history[-window_size:])
        
        # Derivative of error (rate of change)
        if len(roll_history) >= 2:
            roll_derivative = abs(roll_history[-1] - roll_history[-2])
            pitch_derivative = abs(pitch_history[-1] - pitch_history[-2])
        else:
            roll_derivative = 0.0
            pitch_derivative = 0.0
        
        # Oscillation frequency (zero-crossing rate)
        roll_zero_crossings = sum(1 for i in range(1, len(roll_history)) if roll_history[i-1] * roll_history[i] < 0)
        pitch_zero_crossings = sum(1 for i in range(1, len(pitch_history)) if pitch_history[i-1] * pitch_history[i] < 0)
        roll_freq = roll_zero_crossings / max(1, len(roll_history) - 1)
        pitch_freq = pitch_zero_crossings / max(1, len(pitch_history) - 1)
    else:
        roll_var = 0.0
        pitch_var = 0.0
        roll_vel_var = 0.0
        pitch_vel_var = 0.0
        roll_derivative = 0.0
        pitch_derivative = 0.0
        roll_freq = 0.0
        pitch_freq = 0.0
    
    features.extend([
        roll_var,
        pitch_var,
        roll_vel_var,
        pitch_vel_var,
        roll_derivative,
        pitch_derivative,
        roll_freq,
        pitch_freq
    ])
    
    return np.array(features, dtype=np.float32)


def extract_energy_features(
    step_data: Dict[str, Any],
    prev_action: Optional[np.ndarray] = None,
    current_action: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Extract energy-based features.
    
    Features:
    - Actuator saturation (how close to limits)
    - Recovery overshoot (action magnitude after disturbance)
    - Energy dissipation rate
    """
    features = []
    
    # Actuator saturation
    if current_action is not None:
        action_array = np.array(current_action) if not isinstance(current_action, np.ndarray) else current_action
        saturation = np.mean(np.abs(action_array))  # Average absolute action magnitude
        max_saturation = np.max(np.abs(action_array))  # Max saturation
    else:
        saturation = 0.0
        max_saturation = 0.0
    
    # Recovery overshoot (action change magnitude)
    if prev_action is not None and current_action is not None:
        prev_array = np.array(prev_action) if not isinstance(prev_action, np.ndarray) else prev_action
        curr_array = np.array(current_action) if not isinstance(current_action, np.ndarray) else current_action
        action_delta = np.linalg.norm(curr_array - prev_array)
        overshoot = action_delta
    else:
        overshoot = 0.0
    
    # Energy dissipation (from velocities)
    roll_vel = step_data.get("obs", {}).get("roll_velocity", 0.0) if isinstance(step_data.get("obs"), dict) else 0.0
    pitch_vel = step_data.get("obs", {}).get("pitch_velocity", 0.0) if isinstance(step_data.get("obs"), dict) else 0.0
    energy = roll_vel**2 + pitch_vel**2
    
    features.extend([
        saturation,
        max_saturation,
        overshoot,
        energy
    ])
    
    return np.array(features, dtype=np.float32)


def extract_features_v2(
    step_data: Dict[str, Any],
    history: deque,
    prev_action: Optional[np.ndarray] = None,
    current_action: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Extract comprehensive features for fail-risk model v2.
    
    Combines:
    - Basic state features (roll, pitch, velocities, COM)
    - Temporal features (variance, derivatives, frequency)
    - Energy features (saturation, overshoot, energy)
    """
    # Basic state features
    obs = step_data.get("obs", {})
    if not isinstance(obs, dict):
        obs = {}
    
    basic_features = np.array([
        obs.get("roll", 0.0),
        obs.get("pitch", 0.0),
        obs.get("roll_velocity", 0.0),
        obs.get("pitch_velocity", 0.0),
        obs.get("com_velocity_x", 0.0),
        obs.get("com_velocity_y", 0.0),
        obs.get("com_velocity_z", 0.0),
    ], dtype=np.float32)
    
    # Temporal features
    temporal_features = extract_temporal_features(step_data, history)
    
    # Energy features
    energy_features = extract_energy_features(step_data, prev_action, current_action)
    
    # Combine all features
    all_features = np.concatenate([
        basic_features,
        temporal_features,
        energy_features
    ])
    
    return all_features


def compute_fail_label_v2(
    episode_trace: List[Dict[str, Any]],
    horizon_steps: int = 50
) -> np.ndarray:
    """
    Compute binary failure labels for v2 model.
    More conservative - only mark actual failures, not prefall.
    """
    labels = np.zeros(len(episode_trace), dtype=np.float32)
    
    for i in range(len(episode_trace)):
        # Look ahead within horizon
        for j in range(i + 1, min(i + 1 + horizon_steps, len(episode_trace))):
            step_data = episode_trace[j]
            
            # Check for actual failure (not prefall)
            interventions_so_far = step_data.get("interventions_so_far", 0)
            new_intervention = (interventions_so_far > 0)
            
            features = step_data.get("features", {})
            if isinstance(features, dict):
                tilt_zone = features.get("tilt_zone", "")
                is_fail_zone = (tilt_zone == "fail")  # Only actual failures
            else:
                is_fail_zone = False
            
            core_state = step_data.get("core_state", {})
            if isinstance(core_state, dict):
                phase = core_state.get("phase", "")
                is_fail_phase = (phase == "fail")  # Only actual failures
            else:
                is_fail_phase = False
            
            done = step_data.get("done", False)
            early_done = (done and j < len(episode_trace) - 1)
            
            # Only mark as failure if actual failure occurred
            if new_intervention or is_fail_zone or is_fail_phase or early_done:
                labels[i] = 1.0
                break
    
    return labels


def train_fail_risk_model_v2(
    dataset_paths: List[str],
    output_path: str,
    horizon_steps: int = 50,
    model_type: str = "lightgbm"  # or "xgboost"
) -> Dict[str, Any]:
    """
    Train fail-risk model v2 with improved features and XGBoost/LightGBM.
    
    Target: fail-risk separation >= 0.25
    """
    if model_type == "lightgbm" and not HAS_LIGHTGBM:
        print("[ERROR] LightGBM not available. Install with: pip install lightgbm")
        return {}
    if model_type == "xgboost" and not HAS_XGBOOST:
        print("[ERROR] XGBoost not available. Install with: pip install xgboost")
        return {}
    
    print("="*80)
    print("Training Fail-Risk Model v2")
    print(f"Model type: {model_type}")
    print("="*80)
    
    # Load episodes using proper loading function
    from training.fail_risk_model import load_episode_from_jsonl
    
    print(f"\n[1/4] Loading episodes from {len(dataset_paths)} file(s)...")
    all_episode_traces = []
    for dataset_path in dataset_paths:
        path = Path(dataset_path)
        if not path.exists():
            print(f"  Warning: {dataset_path} not found, skipping")
            continue
        
        episodes = load_episode_from_jsonl(path)
        all_episode_traces.extend(episodes)
        print(f"  Loaded {len(episodes)} episodes from {path.name}")
    
    if len(all_episode_traces) == 0:
        print("[ERROR] No episodes found")
        return {}
    
    print(f"  Total episodes: {len(all_episode_traces)}")
    
    # Extract features and labels
    print(f"\n[2/4] Extracting features and labels (horizon={horizon_steps} steps)...")
    X_list = []
    y_list = []
    
    for episode_trace in all_episode_traces:
        if len(episode_trace) < 2:
            continue
        
        # Compute labels
        labels = compute_fail_label_v2(episode_trace, horizon_steps)
        
        # Extract features with history tracking
        history = deque(maxlen=10)
        prev_action = None
        
        for i, step_data in enumerate(episode_trace):
            try:
                # Get current action (if available)
                current_action = step_data.get("action")
                
                # Extract features
                features = extract_features_v2(step_data, history, prev_action, current_action)
                X_list.append(features)
                y_list.append(labels[i])
                
                # Update history
                obs = step_data.get("obs", {})
                if isinstance(obs, dict):
                    history.append({
                        "roll": obs.get("roll", 0.0),
                        "pitch": obs.get("pitch", 0.0),
                        "roll_velocity": obs.get("roll_velocity", 0.0),
                        "pitch_velocity": obs.get("pitch_velocity", 0.0)
                    })
                else:
                    # Try to extract from step_data directly
                    history.append({
                        "roll": step_data.get("roll", 0.0),
                        "pitch": step_data.get("pitch", 0.0),
                        "roll_velocity": step_data.get("roll_velocity", 0.0),
                        "pitch_velocity": step_data.get("pitch_velocity", 0.0)
                    })
                
                prev_action = current_action
            except Exception as e:
                # Skip steps with errors
                continue
    
    if len(X_list) == 0:
        print("[ERROR] No features extracted")
        return {}
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"  Total samples: {len(X)}")
    print(f"  Feature shape: {X.shape}")
    print(f"  Positive labels: {np.sum(y):.0f} ({100*np.sum(y)/len(y):.1f}%)")
    
    if len(X) == 0:
        print("[ERROR] No features extracted")
        return {}
    
    # Train/test split
    print(f"\n[3/4] Splitting data (80/20)...")
    split_idx = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Train model
    print(f"\n[4/4] Training {model_type} model...")
    if model_type == "lightgbm":
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=63,
            objective='binary',
            metric='binary_logloss',
            verbose=-1,
            class_weight='balanced',  # Handle class imbalance
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.log_evaluation(0)])
    else:  # xgboost
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            objective='binary:logistic',
            eval_metric='logloss',
            verbosity=0
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    # Evaluate
    y_pred = model.predict_proba(X_test)[:, 1]
    y_pred_binary = model.predict(X_test)
    
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    accuracy = accuracy_score(y_test, y_pred_binary)
    auc = roc_auc_score(y_test, y_pred)
    
    # Compute separation (mean fail-risk for safe vs failure)
    safe_mask = y_test == 0
    failure_mask = y_test == 1
    
    if np.sum(safe_mask) > 0 and np.sum(failure_mask) > 0:
        mean_safe = np.mean(y_pred[safe_mask])
        mean_failure = np.mean(y_pred[failure_mask])
        separation = mean_failure - mean_safe
    else:
        separation = 0.0
    
    print()
    print("="*80)
    print("Training Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Separation: {separation:.4f}")
    print("="*80)
    
    # Save model
    import pickle
    with open(output_path, 'wb') as f:
        pickle.dump({
            "model": model,
            "model_type": model_type,
            "feature_size": X.shape[1],
            "accuracy": accuracy,
            "auc": auc,
            "separation": separation
        }, f)
    
    print(f"\nModel saved to: {output_path}")
    
    return {
        "accuracy": accuracy,
        "auc": auc,
        "separation": separation
    }

