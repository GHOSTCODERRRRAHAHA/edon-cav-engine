#!/usr/bin/env python3
"""
XGBoost Baseline Training Script v2

Trains an XGBoost model to predict CAV state from OEM dataset features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import joblib
import json
import sys
from typing import Dict, Tuple
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import xgboost as xgb

warnings.filterwarnings('ignore')

# Configuration
DEFAULT_INPUT = "outputs/oem_sample_windows.csv"
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_MODEL_FILE = DEFAULT_MODEL_DIR / "cav_state_xgb_v2.joblib"
DEFAULT_SCALER_FILE = DEFAULT_MODEL_DIR / "cav_state_scaler_v2.joblib"
DEFAULT_SCHEMA_FILE = DEFAULT_MODEL_DIR / "cav_state_schema_v2.json"
RANDOM_STATE = 42


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load OEM dataset from CSV."""
    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} records")
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, int]]:
    """
    Prepare features and target for training.
    
    Features: eda_mean, bvp_std, acc_magnitude_mean, temp_c, humidity, aqi, local_hour
    Target: state (converted to numeric, 0-indexed based on actual classes present)
    
    Returns:
        Tuple of (X, y, state_map) where state_map maps original state names to 0-indexed numeric labels
    """
    # Feature columns
    feature_cols = [
        'eda_mean',
        'bvp_std',
        'acc_magnitude_mean',
        'temp_c',
        'humidity',
        'aqi',
        'local_hour'
    ]
    
    # Check if all features exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")
    
    X = df[feature_cols].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Target: state -> numeric (0-indexed based on actual classes)
    # Get unique states and create 0-indexed mapping
    unique_states = sorted(df['state'].unique())
    state_map = {state: idx for idx, state in enumerate(unique_states)}
    
    y = df['state'].map(state_map)
    
    # Check for unmapped states
    if y.isnull().any():
        unmapped = df[y.isnull()]['state'].unique()
        raise ValueError(f"Unmapped state values: {unmapped}")
    
    return X, y, state_map


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42
) -> Tuple[xgb.XGBClassifier, Dict]:
    """Train XGBoost classifier."""
    print("\nTraining XGBoost model...")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  learning_rate: {learning_rate}")
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Get feature importance
    feature_importance = dict(zip(X_train.columns, model.feature_importances_))
    
    return model, feature_importance


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, scaler: StandardScaler, state_map: Dict[str, int]) -> Dict:
    """Evaluate model performance."""
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    # Classification report - use actual state names from state_map
    reverse_state_map = {v: k for k, v in state_map.items()}
    state_names = [reverse_state_map[i] for i in sorted(reverse_state_map.keys())]
    report = classification_report(y_test, y_pred, target_names=state_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train XGBoost baseline model v2")
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help="Input CSV file path"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Model output directory"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (0.0-1.0)"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation set size (0.0-1.0)"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of XGBoost estimators"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum tree depth"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("XGBoost Baseline Training v2")
    print("=" * 60)
    print()
    
    # Load dataset
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)
    
    df = load_dataset(input_path)
    
    # Prepare features
    print("\nPreparing features...")
    X, y, state_map = prepare_features(df)
    print(f"Features: {list(X.columns)}")
    print(f"Target distribution:")
    state_counts = y.value_counts().sort_index()
    # Create reverse mapping (numeric -> state name)
    reverse_state_map = {v: k for k, v in state_map.items()}
    for state_id, count in state_counts.items():
        state_name = reverse_state_map[state_id]
        print(f"  {state_name:12s}: {count:5,} ({count/len(y)*100:5.1f}%)")
    
    # Split data
    print(f"\nSplitting data (test={args.test_size:.1%}, val={args.val_size:.1%})...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    
    val_size_adjusted = args.val_size / (1 - args.test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_size_adjusted), random_state=args.seed, stratify=y_temp
    )
    
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Val:   {len(X_val):,} samples")
    print(f"  Test:  {len(X_test):,} samples")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for XGBoost
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_df = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Train model
    model, feature_importance = train_model(
        X_train_df, y_train,
        X_val_df, y_val,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=args.seed
    )
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test_df, y_test, scaler, state_map)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    
    print("\nClassification Report:")
    # Get state names from reverse mapping
    reverse_state_map = {v: k for k, v in state_map.items()}
    state_names = [reverse_state_map[i] for i in sorted(reverse_state_map.keys())]
    for state in state_names:
        if state in metrics['classification_report']:
            report = metrics['classification_report'][state]
            print(f"  {state:12s}: precision={report['precision']:.3f}, recall={report['recall']:.3f}, f1={report['f1-score']:.3f}")
    
    print("\nFeature Importance:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature:20s}: {importance:.4f}")
    
    # Save model and artifacts
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_file = model_dir / "cav_state_xgb_v2.joblib"
    scaler_file = model_dir / "cav_state_scaler_v2.joblib"
    schema_file = model_dir / "cav_state_schema_v2.json"
    
    print(f"\nSaving model artifacts...")
    print(f"  Model: {model_file}")
    joblib.dump(model, model_file)
    
    print(f"  Scaler: {scaler_file}")
    joblib.dump(scaler, scaler_file)
    
    print(f"  Schema: {schema_file}")
    # Save state_map with both directions for reference
    # Convert numpy types to native Python types for JSON serialization
    schema = {
        'feature_names': list(X.columns),
        'state_map': state_map,  # state_name -> numeric_id
        'reverse_state_map': {v: k for k, v in state_map.items()},  # numeric_id -> state_name
        'feature_importance': {k: float(v) for k, v in feature_importance.items()},
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'f1_macro': float(metrics['f1_macro']),
            'f1_weighted': float(metrics['f1_weighted'])
        }
    }
    with open(schema_file, 'w') as f:
        json.dump(schema, f, indent=2)
    
    print()
    print("=" * 60)
    print("[SUCCESS] Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

