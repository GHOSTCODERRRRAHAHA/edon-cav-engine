#!/usr/bin/env python3
"""
EDON v6 Training Script

Loads JSONL training logs and trains a PyTorch MLP to predict edon_delta.

Example: collect data
  python run_eval.py --mode edon --profile high_stress --episodes 50 --seed 42 --output data/edon_v61_highstress.json --edon-gain 1.0 --edon-arch v6_1_learned --edon-log-train

Example: train v6.1
  python training/train_edon_v6.py --log-dir logs --epochs 100 --batch-size 256 --output-dir models --model-name edon_v6_1

Example: eval v6.1 vs baseline (5 seeds) - PowerShell
  # Baseline runs:
  foreach ($s in 0,1,2,3,4) {
    python run_eval.py --mode baseline --profile high_stress --episodes 30 --seed $s --output results/baseline_highstress_seed_$s.json
  }
  
  # EDON v6.1 runs:
  foreach ($s in 0,1,2,3,4) {
    python run_eval.py --mode edon --profile high_stress --episodes 30 --seed $s --output results/edon_v61_highstress_seed_$s.json --edon-gain 1.0 --edon-arch v6_1_learned
  }
  
  # Compare:
  python training/compare_v61_vs_baseline.py
"""

import json
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import train_test_split


class EdonV6MLP(nn.Module):
    """
    PyTorch MLP for EDON v6 learned policy.
    
    Architecture:
    - Input: features + core_state + baseline_action
    - Hidden: 128 → 128 → 64
    - Output: edon_delta (same size as baseline_action)
    """
    
    def __init__(self, input_size: int, output_size: int):
        super(EdonV6MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # Linear output, no activation
        return x


def load_jsonl_files(log_dir: str = "logs") -> List[dict]:
    """
    Load all JSONL files from logs directory.
    
    Returns:
        List of step records (type == "step")
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    
    all_records = []
    jsonl_files = list(log_path.glob("*.jsonl"))
    
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {log_dir}")
    
    print(f"Found {len(jsonl_files)} JSONL file(s)")
    
    for jsonl_file in jsonl_files:
        print(f"  Loading: {jsonl_file.name}")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    # Only process step records
                    if record.get("type") == "step":
                        all_records.append(record)
                except json.JSONDecodeError as e:
                    print(f"    Warning: Skipping invalid JSON at line {line_num}: {e}")
                    continue
    
    print(f"Loaded {len(all_records)} step records")
    return all_records


def phase_to_one_hot(phase: str) -> np.ndarray:
    """Convert phase string to one-hot encoding."""
    phase_map = {"stable": 0, "warning": 1, "recovery": 2}
    one_hot = np.zeros(3)
    if phase in phase_map:
        one_hot[phase_map[phase]] = 1.0
    return one_hot


def prepare_dataset(records: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert records into input/target arrays with per-sample weights.
    
    Input vector:
    - tilt_mag (1)
    - vel_norm (1)
    - p_chaos (1)
    - p_stress (1)
    - risk_ema (1)
    - instability_score (1)
    - adaptive_gain (1)
    - phase one-hot (3)
    - baseline_action (variable)
    
    Target:
    - edon_delta (variable, same size as baseline_action)
    
    Weights:
    - Per-sample weights focusing on high-risk, pre-intervention states
    """
    inputs = []
    targets = []
    weights = []
    
    # First pass: determine action size
    action_size = None
    for record in records:
        baseline_action = np.array(record["baseline_action"])
        if action_size is None:
            action_size = len(baseline_action)
        elif len(baseline_action) != action_size:
            print(f"Warning: Inconsistent action size. Expected {action_size}, got {len(baseline_action)}")
            continue
    
    if action_size is None:
        raise ValueError("No valid records found")
    
    print(f"Action size: {action_size}")
    
    # Second pass: build input/target arrays and compute weights
    weight_stats = {"safe": [], "prefall": [], "fail": []}
    for record in records:
        try:
            features = record["features"]
            core_state = record["core_state"]
            baseline_action = np.array(record["baseline_action"])
            edon_delta = np.array(record["edon_delta"])
            
            # Skip if sizes don't match
            if len(baseline_action) != action_size or len(edon_delta) != action_size:
                continue
            
            # Compute per-sample weight based on instability_score and tilt_zone
            instab = float(core_state.get("instability_score", 0.0))
            tilt_zone = features.get("tilt_zone", "safe")
            
            # Zone bonus: make prefall king (highest priority)
            if tilt_zone == "prefall":
                zone_bonus = 2.0     # strongest focus – save it before fall
            elif tilt_zone == "fail":
                zone_bonus = 1.0     # still important, but less
            else:  # "safe" or other
                zone_bonus = 0.0
            
            # Weight formula: w = 1.0 + 3.0 * instab + zone_bonus
            # - safe & stable → ≈ 1.0
            # - unstable safe → 1–4
            # - prefall + high instab → 3–6+ (biggest weight)
            # - fail + high instab → 2–5+
            w = 1.0 + 3.0 * instab + zone_bonus
            
            # Clamp to avoid crazy outliers
            w = max(0.25, min(w, 6.0))
            
            weights.append(w)
            
            # Track weights by zone for debugging
            if tilt_zone in weight_stats:
                weight_stats[tilt_zone].append(w)
            else:
                weight_stats["safe"].append(w)
            
            # Build input vector
            input_vec = np.concatenate([
                [features["tilt_mag"]],
                [features["vel_norm"]],
                [features["p_chaos"]],
                [features["p_stress"]],
                [features["risk_ema"]],
                [core_state["instability_score"]],
                [core_state["adaptive_gain"]],
                phase_to_one_hot(core_state["phase"]),
                baseline_action
            ])
            
            inputs.append(input_vec)
            targets.append(edon_delta)
            
        except (KeyError, ValueError) as e:
            print(f"Warning: Skipping record due to error: {e}")
            continue
    
    inputs = np.array(inputs, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    weights = np.array(weights, dtype=np.float32)
    
    # Normalize weights so average weight is ~1.0
    mean_w = float(weights.mean()) if weights.size > 0 else 1.0
    if mean_w <= 0:
        mean_w = 1.0
    weights = weights / mean_w
    
    print(f"Dataset shape: inputs={inputs.shape}, targets={targets.shape}, weights={weights.shape}")
    print(f"Weights: min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}")
    
    # Debug: Show weight distribution by zone (before normalization)
    print(f"\n[DEBUG] Weight distribution by zone (raw, before normalization):")
    for zone, w_list in weight_stats.items():
        if len(w_list) > 0:
            w_arr = np.array(w_list)
            print(f"  {zone:8s}: count={len(w_list):6d}, mean={w_arr.mean():.3f}, min={w_arr.min():.3f}, max={w_arr.max():.3f}")
    
    # v6.1: Show weight stats after normalization (approximate per-zone)
    # Note: We normalized globally, so per-zone stats are approximate but informative
    zone_indices = {"safe": [], "prefall": [], "fail": []}
    for idx, record in enumerate(records):
        if idx >= len(weights):
            break
        try:
            tilt_zone = record.get("features", {}).get("tilt_zone", "safe")
            if tilt_zone in zone_indices:
                zone_indices[tilt_zone].append(idx)
        except:
            continue
    
    print(f"\n[EDON-V6.1] Weight stats after normalization:")
    for zone, indices in zone_indices.items():
        if len(indices) > 0:
            zone_weights = weights[indices]
            print(f"  {zone:8s}: n={len(indices):6d}, mean={zone_weights.mean():.3f}")
    print(f"  global_mean={weights.mean():.3f}, global_std={weights.std():.3f}")
    
    return inputs, targets, weights


class EdonV6Dataset(torch.utils.data.Dataset):
    """Dataset for EDON v6 training with per-sample weights."""
    
    def __init__(self, inputs, targets, weights):
        self.inputs = torch.FloatTensor(inputs)
        self.targets = torch.FloatTensor(targets)
        self.weights = torch.FloatTensor(weights)
    
    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.weights[idx]


def train_model(
    inputs: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    train_split: float = 0.9
) -> EdonV6MLP:
    """
    Train the EDON v6 MLP model.
    
    Returns:
        Trained model
    """
    # Split into train/validation (including weights)
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        inputs, targets, weights, test_size=1.0 - train_split, random_state=42
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Create model
    input_size = inputs.shape[1]
    output_size = targets.shape[1]
    model = EdonV6MLP(input_size, output_size)
    
    # Optimizer (no criterion needed, we'll compute weighted MSE manually)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create datasets and dataloaders
    train_dataset = EdonV6Dataset(X_train, y_train, w_train)
    val_dataset = EdonV6Dataset(X_val, y_val, w_val)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    device = torch.device("cpu")  # Use CPU for training
    
    # Training loop
    print("\nTraining...")
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        # Mini-batch training with weighted loss
        for batch_idx, (batch_X, batch_y, batch_w) in enumerate(train_loader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_X)
            
            # Weighted MSE: compute per-sample MSE, then weight and average
            mse_per_sample = ((preds - batch_y) ** 2).mean(dim=1)  # shape [batch]
            loss = (mse_per_sample * batch_w).mean()
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            # v6.1: Print weight stats for first batch of first epoch
            if epoch == 0 and batch_idx == 0:
                batch_w_np = batch_w.cpu().numpy()
                print(f"[EDON-V6.1] First batch weight stats: mean={batch_w_np.mean():.3f}, std={batch_w_np.std():.3f}, min={batch_w_np.min():.3f}, max={batch_w_np.max():.3f}")
        
        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
        
        # Validation (also weighted for consistency)
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_X, batch_y, batch_w in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                batch_w = batch_w.to(device)
                
                preds = model(batch_X)
                mse_per_sample = ((preds - batch_y) ** 2).mean(dim=1)
                batch_loss = (mse_per_sample * batch_w).mean()
                
                val_loss += batch_loss.item() * batch_X.size(0)
                val_count += batch_X.size(0)
        
        avg_val_loss = val_loss / val_count if val_count > 0 else 0.0
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
    
    return model


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train EDON v6 learned policy")
    parser.add_argument("--log-dir", type=str, default="logs",
                       help="Directory containing JSONL log files")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Directory to save trained model")
    parser.add_argument("--dataset-path", type=str, default="data/edon_v6_dataset.npz",
                       help="Path to save processed dataset")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--train-split", type=float, default=0.9,
                       help="Train/validation split ratio")
    parser.add_argument("--model-name", type=str, default="edon_v6_1",
                       help="Base filename for the saved model (without extension).")
    
    args = parser.parse_args()
    
    print("="*70)
    print("EDON v6 Training")
    print("="*70)
    
    # Load JSONL files
    print("\n[1/4] Loading JSONL files...")
    records = load_jsonl_files(args.log_dir)
    
    if len(records) == 0:
        print("ERROR: No step records found. Make sure you have run evaluations with --edon-log-train")
        return
    
    # Prepare dataset
    print("\n[2/4] Preparing dataset...")
    inputs, targets, weights = prepare_dataset(records)
    
    # Sanity checks
    assert inputs.shape[0] == targets.shape[0] == weights.shape[0], \
        f"Shape mismatch: inputs={inputs.shape[0]}, targets={targets.shape[0]}, weights={weights.shape[0]}"
    assert abs(weights.mean() - 1.0) < 0.01, \
        f"Weights mean should be ~1.0, got {weights.mean():.3f}"
    
    # Save dataset
    dataset_path = Path(args.dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dataset_path,
        inputs=inputs,
        targets=targets,
        weights=weights,
        input_size=inputs.shape[1],
        output_size=targets.shape[1]
    )
    print(f"Saved dataset to: {dataset_path}")
    
    # Train model
    print("\n[3/4] Training model...")
    model = train_model(
        inputs, targets, weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_split=args.train_split
    )
    
    # Save model
    print("\n[4/4] Saving model...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{args.model_name}.pt"
    
    # Check if model already exists
    if model_path.exists():
        print(f"[OK] Overwriting existing {model_path.name}")
    
    # Save model state dict and metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': inputs.shape[1],
        'output_size': targets.shape[1],
        'architecture': 'MLP_128_128_64'
    }, model_path)
    
    print(f"Model saved to: {model_path}")
    print(f"\nModel architecture:")
    print(f"  Input size: {inputs.shape[1]}")
    print(f"  Output size: {targets.shape[1]}")
    print(f"  Hidden layers: 128 -> 128 -> 64")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

