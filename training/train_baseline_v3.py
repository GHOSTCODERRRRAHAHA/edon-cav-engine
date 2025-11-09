#!/usr/bin/env python3
"""
XGBoost/LightGBM Baseline Training Script v3

Trains an optimized model with GridSearchCV on the 100K dataset.
Supports both XGBoost and LightGBM with hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import joblib
import json
import sys
from typing import Dict, Tuple, Optional
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import xgboost as xgb

# Try to import LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings('ignore')

# Configuration
DEFAULT_INPUT = "outputs/oem_100k_windows.parquet"
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_MODEL_FILE = DEFAULT_MODEL_DIR / "cav_state_xgb_v3.joblib"
DEFAULT_SCALER_FILE = DEFAULT_MODEL_DIR / "cav_state_scaler_v3.joblib"
DEFAULT_SCHEMA_FILE = DEFAULT_MODEL_DIR / "cav_state_schema_v3.json"
RANDOM_STATE = 42


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load OEM dataset from Parquet or CSV."""
    print(f"Loading dataset: {data_path}")
    
    if data_path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    
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
    unique_states = sorted(df['state'].unique())
    state_map = {state: idx for idx, state in enumerate(unique_states)}
    
    y = df['state'].map(state_map)
    
    # Check for unmapped states
    if y.isnull().any():
        unmapped = df[y.isnull()]['state'].unique()
        raise ValueError(f"Unmapped state values: {unmapped}")
    
    return X, y, state_map


def train_with_gridsearch(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: str = 'xgboost',
    cv: int = 3,
    n_jobs: int = -1
) -> Tuple:
    """Train model with GridSearchCV."""
    print(f"\nTraining {model_type.upper()} model with GridSearchCV...")
    print(f"  CV folds: {cv}")
    print(f"  n_jobs: {n_jobs}")
    
    if model_type.lower() == 'lightgbm' and not HAS_LIGHTGBM:
        print("[WARNING] LightGBM not available, falling back to XGBoost")
        model_type = 'xgboost'
    
    if model_type.lower() == 'lightgbm':
        # LightGBM parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [31, 50],
            'min_child_samples': [20, 30]
        }
        
        base_model = lgb.LGBMClassifier(
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=n_jobs
        )
    else:
        # XGBoost parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1],
            'min_child_weight': [1, 3],
            'subsample': [0.8, 1.0]
        }
        
        base_model = xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric='mlogloss',
            use_label_encoder=False,
            n_jobs=n_jobs
        )
    
    # GridSearchCV
    print("  Running GridSearchCV (this may take a while)...")
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=n_jobs,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"\n  Best parameters: {best_params}")
    print(f"  Best CV score (F1 weighted): {best_score:.4f}")
    
    # Get feature importance
    feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))
    
    return best_model, feature_importance, best_params


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
    
    # Classification report
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


def print_confusion_matrix(cm: np.ndarray, state_map: Dict[str, int]):
    """Print formatted confusion matrix."""
    reverse_state_map = {v: k for k, v in state_map.items()}
    state_names = [reverse_state_map[i] for i in sorted(reverse_state_map.keys())]
    
    print("\nConfusion Matrix:")
    print(" " * 15, end="")
    for state in state_names:
        print(f"{state:12s}", end="")
    print()
    
    for i, state in enumerate(state_names):
        print(f"{state:12s}", end="")
        for j in range(len(state_names)):
            print(f"{cm[i][j]:12d}", end="")
        print()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train optimized XGBoost/LightGBM model v3")
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help="Input Parquet or CSV file path"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Model output directory"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=['xgboost', 'lightgbm'],
        default='xgboost',
        help="Model type: xgboost or lightgbm"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (0.0-1.0)"
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Number of CV folds for GridSearchCV"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 for all cores)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("XGBoost/LightGBM Baseline Training v3 (GridSearchCV)")
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
    reverse_state_map = {v: k for k, v in state_map.items()}
    for state_id, count in state_counts.items():
        state_name = reverse_state_map[state_id]
        print(f"  {state_name:12s}: {count:5,} ({count/len(y)*100:5.1f}%)")
    
    # Split data
    print(f"\nSplitting data (test={args.test_size:.1%})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test:  {len(X_test):,} samples")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Split train into train/val for GridSearchCV
    X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(
        X_train_df, y_train, test_size=0.2, random_state=args.seed, stratify=y_train
    )
    
    # Train with GridSearchCV
    model, feature_importance, best_params = train_with_gridsearch(
        X_train_cv, y_train_cv,
        X_val_cv, y_val_cv,
        model_type=args.model_type,
        cv=args.cv,
        n_jobs=args.n_jobs
    )
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    metrics = evaluate_model(model, X_test_df, y_test, scaler, state_map)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    
    print("\nClassification Report:")
    state_names = [reverse_state_map[i] for i in sorted(reverse_state_map.keys())]
    for state in state_names:
        if state in metrics['classification_report']:
            report = metrics['classification_report'][state]
            print(f"  {state:12s}: precision={report['precision']:.3f}, recall={report['recall']:.3f}, f1={report['f1-score']:.3f}")
    
    # Confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    print_confusion_matrix(cm, state_map)
    
    print("\nFeature Importance:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature:20s}: {importance:.4f}")
    
    # Save model and artifacts
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_file = model_dir / "cav_state_xgb_v3.joblib"
    scaler_file = model_dir / "cav_state_scaler_v3.joblib"
    schema_file = model_dir / "cav_state_schema_v3.json"
    
    print(f"\nSaving model artifacts...")
    print(f"  Model: {model_file}")
    joblib.dump(model, model_file)
    
    print(f"  Scaler: {scaler_file}")
    joblib.dump(scaler, scaler_file)
    
    print(f"  Schema: {schema_file}")
    schema = {
        'feature_names': list(X.columns),
        'state_map': state_map,
        'reverse_state_map': {v: k for k, v in state_map.items()},
        'feature_importance': {k: float(v) for k, v in feature_importance.items()},
        'best_parameters': {k: (float(v) if isinstance(v, (np.integer, np.floating)) else v) for k, v in best_params.items()},
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'f1_macro': float(metrics['f1_macro']),
            'f1_weighted': float(metrics['f1_weighted'])
        },
        'confusion_matrix': metrics['confusion_matrix']
    }
    with open(schema_file, 'w') as f:
        json.dump(schema, f, indent=2)
    
    print()
    print("=" * 60)
    print("[SUCCESS] Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

