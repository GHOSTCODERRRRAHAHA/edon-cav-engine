"""
Predictive Failure Risk Model for EDON v8

Computes fail_risk: probability of failure (intervention/fall) in the next 0.5-1.0 seconds.
Used as a first-class signal in v8 layered control architecture.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json


def compute_fail_label(episode_trace: List[Dict[str, Any]], horizon_steps: int = 50) -> np.ndarray:
    """
    Compute binary failure labels for each timestep in an episode.
    
    For each timestep t, label is:
    - 1 if a fall/intervention occurs within the next horizon_steps
    - 0 otherwise
    
    Args:
        episode_trace: List of step records, each containing:
            - obs or state dict (with roll, pitch, velocities, etc.)
            - info dict (with intervention, fallen flags)
            - done flag
        horizon_steps: Number of steps to look ahead (default 50 ≈ 0.5-1.0s at 50-100Hz)
    
    Returns:
        Binary labels array of shape (len(episode_trace),) with dtype float32
    """
    labels = np.zeros(len(episode_trace), dtype=np.float32)
    
    # Track interventions_so_far to detect when new intervention occurs
    prev_interventions = {}
    
    for i in range(len(episode_trace)):
        # Look ahead within horizon
        for j in range(i + 1, min(i + 1 + horizon_steps, len(episode_trace))):
            step_data = episode_trace[j]
            
            # Method 1: Check interventions_so_far (increases = new intervention)
            interventions_so_far = step_data.get("interventions_so_far", 0)
            prev_interventions_for_step = prev_interventions.get(j - 1, 0)
            new_intervention = (interventions_so_far > prev_interventions_for_step)
            prev_interventions[j] = interventions_so_far
            
            # Method 2: Check tilt_zone in features (ONLY mark "fail", not "prefall")
            features = step_data.get("features", {})
            if isinstance(features, dict):
                tilt_zone = features.get("tilt_zone", "")
                is_fail_zone = (tilt_zone == "fail")  # Only actual failures, not prefall
            else:
                is_fail_zone = False
            
            # Method 3: Check phase in core_state (ONLY mark "fail", not "prefall")
            core_state = step_data.get("core_state", {})
            if isinstance(core_state, dict):
                phase = core_state.get("phase", "")
                is_fail_phase = (phase == "fail")  # Only actual failures, not prefall
            else:
                is_fail_phase = False
            
            # Method 4: Check done flag (episode ended early = likely failure)
            done = step_data.get("done", False)
            early_done = (done and j < len(episode_trace) - 1)
            
            # Method 5: Check for extreme tilt from features (only very high tilt, not prefall threshold)
            if isinstance(features, dict):
                tilt_mag = features.get("tilt_mag", 0.0)
                extreme_tilt = (tilt_mag > 0.40)  # Higher threshold - only mark actual failures, not warnings
            else:
                extreme_tilt = False
            
            # Method 6: Check info dict if present (legacy format)
            info = step_data.get("info", {})
            if isinstance(info, dict):
                intervention_from_info = info.get("intervention", False)
                fallen_from_info = info.get("fallen", False)
            else:
                intervention_from_info = False
                fallen_from_info = False
            
            # Method 7: Check direct fields (legacy format)
            intervention_direct = step_data.get("intervention", False)
            fallen_direct = step_data.get("fallen", False)
            
            # Label as failure if any failure signal detected
            if (new_intervention or is_fail_zone or is_fail_phase or early_done or 
                extreme_tilt or intervention_from_info or fallen_from_info or 
                intervention_direct or fallen_direct):
                labels[i] = 1.0
                break  # Found failure, no need to check further ahead
    
    return labels


def extract_features_from_step(step_data: Dict[str, Any]) -> np.ndarray:
    """
    Extract feature vector from a single step for fail-risk prediction.
    
    Features:
    - roll, pitch (tilt angles)
    - roll_velocity, pitch_velocity
    - com_x, com_y (center of mass)
    - com_velocity_x, com_velocity_y
    - tilt_mag (magnitude of tilt)
    - vel_norm (velocity magnitude)
    - instability_score (if available)
    - phase/zone (encoded as one-hot or numeric)
    - risk_ema (if available)
    
    Args:
        step_data: Step record dict with obs/state and optional features
    
    Returns:
        Feature vector as numpy array
    """
    obs = step_data.get("obs") or step_data.get("state") or step_data
    
    # Extract basic state
    roll = float(obs.get("roll", 0.0)) if isinstance(obs, dict) else 0.0
    pitch = float(obs.get("pitch", 0.0)) if isinstance(obs, dict) else 0.0
    roll_velocity = float(obs.get("roll_velocity", 0.0)) if isinstance(obs, dict) else 0.0
    pitch_velocity = float(obs.get("pitch_velocity", 0.0)) if isinstance(obs, dict) else 0.0
    com_x = float(obs.get("com_x", 0.0)) if isinstance(obs, dict) else 0.0
    com_y = float(obs.get("com_y", 0.0)) if isinstance(obs, dict) else 0.0
    com_velocity_x = float(obs.get("com_velocity_x", 0.0)) if isinstance(obs, dict) else 0.0
    com_velocity_y = float(obs.get("com_velocity_y", 0.0)) if isinstance(obs, dict) else 0.0
    
    # Compute derived features
    tilt_mag = np.sqrt(roll**2 + pitch**2)
    vel_norm = np.sqrt(roll_velocity**2 + pitch_velocity**2)
    com_norm = np.sqrt(com_x**2 + com_y**2)
    com_vel_norm = np.sqrt(com_velocity_x**2 + com_velocity_y**2)
    
    # Extract additional features if available
    features_dict = step_data.get("features", {})
    instability_score = float(features_dict.get("instability_score", 0.0)) if isinstance(features_dict, dict) else 0.0
    risk_ema = float(features_dict.get("risk_ema", 0.0)) if isinstance(features_dict, dict) else 0.0
    
    # Extract phase/zone (encode as numeric: stable=0, warning=1, recovery=2, prefall=3, fail=4)
    core_state = step_data.get("core_state", {})
    phase = core_state.get("phase", "stable") if isinstance(core_state, dict) else "stable"
    phase_map = {"stable": 0.0, "warning": 1.0, "recovery": 2.0, "prefall": 3.0, "fail": 4.0}
    phase_encoded = phase_map.get(phase, 0.0)
    
    # Pack feature vector
    features = np.array([
        roll, pitch, roll_velocity, pitch_velocity,
        com_x, com_y, com_velocity_x, com_velocity_y,
        tilt_mag, vel_norm, com_norm, com_vel_norm,
        instability_score, risk_ema, phase_encoded
    ], dtype=np.float32)
    
    return features


class FailRiskModel(nn.Module):
    """
    Simple MLP for predicting failure risk.
    
    Input: feature vector (15 dims)
    Output: fail_risk ∈ [0, 1] via sigmoid
    """
    
    def __init__(self, input_size: int = 15, hidden_sizes: List[int] = [64, 32, 16]):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer (single scalar)
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())  # Output in [0, 1]
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: output fail_risk ∈ [0, 1]"""
        return self.net(x).squeeze(-1)


def load_episode_from_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """
    Load episode data from JSONL file.
    
    Expected format: one JSON object per line, with type="step" records.
    """
    episodes = []
    current_episode = []
    current_episode_id = None
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                record_type = record.get("type", "step")
                
                if record_type == "meta":
                    # Metadata record, skip
                    continue
                elif record_type == "step":
                    # Step record
                    episode_id = record.get("episode_id", 0)
                    
                    # If new episode, save previous and start new
                    if current_episode_id is not None and episode_id != current_episode_id:
                        if current_episode:
                            episodes.append(current_episode)
                        current_episode = []
                    
                    current_episode_id = episode_id
                    current_episode.append(record)
                elif record_type == "episode_summary":
                    # End of episode marker
                    if current_episode:
                        episodes.append(current_episode)
                        current_episode = []
                        current_episode_id = None
            except json.JSONDecodeError:
                continue
    
    # Add final episode if exists
    if current_episode:
        episodes.append(current_episode)
    
    return episodes


def prepare_dataset(episodes: List[List[Dict[str, Any]]], horizon_steps: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training dataset from episodes.
    
    Args:
        episodes: List of episode traces (each is a list of step records)
        horizon_steps: Look-ahead horizon for failure labels
    
    Returns:
        (features, labels) as numpy arrays
    """
    all_features = []
    all_labels = []
    
    for episode_trace in episodes:
        if len(episode_trace) < 2:
            continue  # Skip very short episodes
        
        # Compute labels for this episode
        labels = compute_fail_label(episode_trace, horizon_steps=horizon_steps)
        
        # Extract features for each step
        for i, step_data in enumerate(episode_trace):
            try:
                features = extract_features_from_step(step_data)
                all_features.append(features)
                all_labels.append(labels[i])
            except Exception as e:
                # Skip steps with missing data
                continue
    
    if len(all_features) == 0:
        raise ValueError("No valid training data found")
    
    features_array = np.array(all_features, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.float32)
    
    return features_array, labels_array


def train_fail_risk_model(
    dataset_paths: List[str],
    output_path: str = "models/edon_fail_risk_v1.pt",
    horizon_steps: int = 50,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    train_split: float = 0.9
) -> FailRiskModel:
    """
    Train fail-risk model from JSONL log files.
    
    Args:
        dataset_paths: List of paths to JSONL log files
        output_path: Path to save trained model
        horizon_steps: Look-ahead horizon for failure labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        train_split: Train/validation split ratio
    
    Returns:
        Trained FailRiskModel
    """
    print("="*70)
    print("Training Fail-Risk Model")
    print("="*70)
    
    # Load episodes from all dataset files
    print(f"\n[1/4] Loading episodes from {len(dataset_paths)} file(s)...")
    all_episodes = []
    for dataset_path in dataset_paths:
        path = Path(dataset_path)
        if not path.exists():
            print(f"  Warning: {dataset_path} not found, skipping")
            continue
        
        episodes = load_episode_from_jsonl(path)
        all_episodes.extend(episodes)
        print(f"  Loaded {len(episodes)} episodes from {path.name}")
    
    if len(all_episodes) == 0:
        raise ValueError("No episodes found in dataset paths")
    
    print(f"  Total episodes: {len(all_episodes)}")
    
    # Prepare dataset
    print(f"\n[2/4] Preparing dataset (horizon={horizon_steps} steps)...")
    features, labels = prepare_dataset(all_episodes, horizon_steps=horizon_steps)
    
    print(f"  Total samples: {len(features)}")
    print(f"  Feature shape: {features.shape}")
    print(f"  Positive labels: {np.sum(labels):.0f} ({100.0 * np.mean(labels):.1f}%)")
    
    # Train/validation split
    n_train = int(len(features) * train_split)
    indices = np.random.permutation(len(features))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    val_features = features[val_indices]
    val_labels = labels[val_indices]
    
    print(f"  Train samples: {len(train_features)}")
    print(f"  Val samples: {len(val_features)}")
    
    # Create model
    input_size = features.shape[1]
    model = FailRiskModel(input_size=input_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\n[3/4] Training for {epochs} epochs...")
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(train_features), batch_size):
            batch_features = torch.FloatTensor(train_features[i:i+batch_size]).to(device)
            batch_labels = torch.FloatTensor(train_labels[i:i+batch_size]).to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = train_loss / n_batches if n_batches > 0 else 0.0
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_features_tensor = torch.FloatTensor(val_features).to(device)
            val_outputs = model(val_features_tensor)
            val_labels_tensor = torch.FloatTensor(val_labels).to(device)
            val_loss = criterion(val_outputs, val_labels_tensor).item()
            
            # Compute accuracy and predictions for metrics
            val_preds = (val_outputs > 0.5).float()
            val_accuracy = (val_preds == val_labels_tensor).float().mean().item()
            
            # Compute AUC (simplified - using sklearn if available, else skip)
            try:
                from sklearn.metrics import roc_auc_score
                val_auc = roc_auc_score(val_labels.cpu().numpy() if hasattr(val_labels, 'cpu') else val_labels, 
                                       val_outputs.cpu().numpy() if hasattr(val_outputs, 'cpu') else val_outputs)
            except (ImportError, Exception):
                val_auc = None
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            auc_str = f", val_auc={val_auc:.4f}" if val_auc is not None else ""
            print(f"  Epoch {epoch + 1}/{epochs}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}, val_acc={val_accuracy:.4f}{auc_str}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_val_accuracy = val_accuracy
            best_val_auc = val_auc
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        # Fallback: use current model state
        best_val_accuracy = val_accuracy
        best_val_auc = val_auc
    
    print(f"\n[4/4] Saving model to {output_path}")
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "horizon_steps": horizon_steps,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "best_val_auc": best_val_auc if best_val_auc is not None else -1.0
    }, output_path)
    
    # Save metrics to text file
    metrics_path = output_path_obj.with_suffix('.txt')
    try:
        with open(metrics_path, 'w') as f:
            f.write("EDON v8 Fail-Risk Model Training Metrics\n")
            f.write("="*70 + "\n\n")
            f.write(f"Dataset: {len(dataset_paths)} file(s), {len(all_episodes)} episodes\n")
            f.write(f"Total samples: {len(features)}\n")
            f.write(f"Train samples: {len(train_features)}\n")
            f.write(f"Val samples: {len(val_features)}\n")
            f.write(f"Positive label rate: {100.0 * np.mean(labels):.1f}%\n")
            f.write(f"Horizon steps: {horizon_steps}\n")
            f.write(f"Epochs: {epochs}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Learning rate: {learning_rate}\n\n")
            f.write("Final Metrics:\n")
            f.write(f"  Best validation loss: {best_val_loss:.6f}\n")
            f.write(f"  Best validation accuracy: {best_val_accuracy:.4f}\n")
            if best_val_auc is not None:
                f.write(f"  Best validation AUC: {best_val_auc:.4f}\n")
            f.write(f"\nModel saved to: {output_path}\n")
    except Exception as e:
        print(f"  Warning: Could not save metrics file: {e}")
    
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Best validation accuracy: {best_val_accuracy:.4f}")
    if best_val_auc is not None:
        print(f"  Best validation AUC: {best_val_auc:.4f}")
    print(f"  Metrics saved to: {metrics_path}")
    print("="*70)
    print("Training complete!")
    
    return model

