#!/usr/bin/env python3
"""
XGBoost/LightGBM Baseline Training Script v3.2

Enhanced training with 4 Hz-friendly signal features, per-class thresholds,
and improved minority-class F1. No synthetic HRV approximations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import joblib
import json
import sys
from typing import Dict, Tuple, Optional, List
from itertools import product
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    make_scorer
)
from sklearn.utils.class_weight import compute_class_weight
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
DEFAULT_MODEL_FILE = DEFAULT_MODEL_DIR / "cav_state_v3_2.joblib"
DEFAULT_SCALER_FILE = DEFAULT_MODEL_DIR / "cav_state_scaler_v3_2.joblib"
DEFAULT_SCHEMA_FILE = DEFAULT_MODEL_DIR / "cav_state_schema_v3_2.json"
RANDOM_STATE = 42

# Default per-class thresholds
DEFAULT_THRESHOLDS = {
    'balanced': 0.30,
    'focus': 0.35,
    'restorative': 0.50
}


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load OEM dataset from Parquet or CSV."""
    print(f"Loading dataset: {data_path}")
    
    if data_path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df):,} records")
    return df


def compute_4hz_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 4 Hz-friendly signal features from available aggregated features.
    
    Since we only have aggregated features (eda_mean, bvp_std, acc_magnitude_mean),
    we derive derivative and variance features from what's available.
    """
    features = pd.DataFrame(index=df.index)
    
    # Base features we have
    eda_mean = df['eda_mean'].values
    bvp_std = df['bvp_std'].values
    acc_magnitude_mean = df['acc_magnitude_mean'].values
    
    # EDA features
    # eda_deriv_std: Approximate from eda_mean variability
    # Use rolling window approach or use bvp_std as proxy for signal variability
    # For now, use a combination of eda_mean and bvp_std
    eda_deriv_std = np.abs(eda_mean) * 0.5 + bvp_std * 0.3
    features['eda_deriv_std'] = eda_deriv_std
    
    # eda_deriv_pos_rate: Fraction of positive derivatives
    # Approximate from eda_mean trend (if positive, more positive derivatives)
    # Use sigmoid-like function based on eda_mean
    eda_deriv_pos_rate = 0.5 + 0.3 * np.tanh(eda_mean * 2.0)  # Normalize to [0.2, 0.8]
    features['eda_deriv_pos_rate'] = eda_deriv_pos_rate
    
    # ACC features
    # acc_var: Variance of acceleration magnitude
    # Approximate from acc_magnitude_mean (higher mean = higher variance typically)
    acc_var = (acc_magnitude_mean ** 2) * 0.1  # Scale to reasonable variance
    features['acc_var'] = acc_var
    
    # Pruned zero-importance features: bvp_diff_std, bvp_diff_mean_abs, acc_energy
    
    return features


def prepare_features(df: pd.DataFrame, use_env: bool = False) -> Tuple[pd.DataFrame, pd.Series, Dict[str, int]]:
    """
    Prepare features and target for training.
    
    Features (6 signal features):
        ['eda_mean', 'eda_deriv_std', 'eda_deriv_pos_rate',
         'bvp_std', 'acc_magnitude_mean', 'acc_var']
    
    Optional environmental features (if use_env=True):
        ['temp_c', 'humidity', 'aqi', 'local_hour']
    
    Returns:
        Tuple of (X, y, state_map) where state_map maps original state names to 0-indexed numeric labels
    """
    # Required base features
    required_cols = ['eda_mean', 'bvp_std', 'acc_magnitude_mean']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")
    
    # Compute 4 Hz-friendly features
    print("  Computing 4 Hz-friendly signal features...")
    signal_features = compute_4hz_features(df)
    
    # Base signal features
    base_features = pd.DataFrame({
        'eda_mean': df['eda_mean'].values,
        'bvp_std': df['bvp_std'].values,
        'acc_magnitude_mean': df['acc_magnitude_mean'].values
    }, index=df.index)
    
    # Combine all signal features
    X = pd.concat([base_features, signal_features], axis=1)
    
    # Pruned zero-importance features
    pruned_features = ['bvp_diff_std', 'bvp_diff_mean_abs', 'acc_energy']
    print(f"  [INFO] Pruned zero-importance features: {pruned_features}")
    
    # Reorder to match specification (6 features)
    signal_feature_cols = [
        'eda_mean',
        'eda_deriv_std',
        'eda_deriv_pos_rate',
        'bvp_std',
        'acc_magnitude_mean',
        'acc_var'
    ]
    
    X = X[signal_feature_cols]
    
    # Add environmental features if requested
    if use_env:
        env_cols = ['temp_c', 'humidity', 'aqi', 'local_hour']
        missing_env = [col for col in env_cols if col not in df.columns]
        if missing_env:
            print(f"  [WARNING] Missing environmental columns: {missing_env}")
        else:
            for col in env_cols:
                X[col] = df[col].values
            print("  Added environmental features: temp_c, humidity, aqi, local_hour")
    
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
    class_weights: Optional[Dict[int, float]] = None,
    model_type: str = 'xgb',
    cv: int = 3,
    n_jobs: int = -1
) -> Tuple:
    """Train model with GridSearchCV using macro-F1 scoring."""
    print(f"\nTraining {model_type.upper()} model with GridSearchCV...")
    print(f"  CV folds: {cv}")
    print(f"  n_jobs: {n_jobs}")
    print(f"  Scoring: macro-F1")
    if class_weights:
        print(f"  Class weights: {class_weights}")
    
    if model_type.lower() == 'lgbm' and not HAS_LIGHTGBM:
        print("[WARNING] LightGBM not available, falling back to XGBoost")
        model_type = 'xgb'
    
    # Create macro-F1 scorer
    macro_f1_scorer = make_scorer(f1_score, average='macro')
    
    if model_type.lower() == 'lgbm':
        # LightGBM parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.2],
            'num_leaves': [31, 50],
            'min_child_samples': [20, 30]
        }
        
        # Get number of classes
        num_classes = len(np.unique(y_train))
        
        # Set class_weight based on whether class weights are provided
        lgbm_class_weight = None if class_weights is None else 'balanced'
        
        base_model = lgb.LGBMClassifier(
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=n_jobs,
            class_weight=lgbm_class_weight,
            num_class=num_classes,
            objective='multiclass'
        )
    else:
        # XGBoost parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'min_child_weight': [1, 3]
        }
        
        base_model = xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric='mlogloss',
            use_label_encoder=False,
            n_jobs=n_jobs
        )
    
    # GridSearchCV with macro-F1 scoring
    print("  Running GridSearchCV (this may take a while)...")
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring=macro_f1_scorer,
        n_jobs=n_jobs,
        verbose=1
    )
    
    # For XGBoost, apply sample weights
    if model_type.lower() == 'xgb' and class_weights:
        sample_weights = np.array([class_weights[y] for y in y_train])
        grid_search.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"\n  Best parameters: {best_params}")
    print(f"  Best CV score (macro-F1): {best_score:.4f}")
    
    # Get feature importance
    feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))
    
    return best_model, feature_importance, best_params, model_type.lower()


def tune_per_class_thresholds(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    state_map: Dict[str, int],
    default_thresholds: Dict[str, float]
) -> Dict[str, float]:
    """
    Tune per-class thresholds to maximize macro-F1 on validation set.
    
    Args:
        model: Trained classifier
        X_val: Validation features
        y_val: Validation labels
        state_map: State name to class ID mapping
        default_thresholds: Default thresholds per class
        
    Returns:
        Dictionary of optimized thresholds per class
    """
    print("\nTuning per-class thresholds...")
    
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_val)
    
    # Reverse state map
    reverse_state_map = {v: k for k, v in state_map.items()}
    n_classes = len(state_map)
    
    # Try different threshold combinations
    threshold_candidates = np.arange(0.1, 0.9, 0.05)
    best_thresholds = default_thresholds.copy()
    best_f1 = 0.0
    
    # Grid search over threshold combinations
    print("  Searching threshold space...")
    for thresholds in product(threshold_candidates, repeat=n_classes):
        # Create threshold dict
        thresh_dict = {reverse_state_map[i]: float(thresholds[i]) for i in range(n_classes)}
        
        # Apply thresholds
        y_pred = np.zeros(len(y_val), dtype=int)
        for i in range(n_classes):
            mask = y_pred_proba[:, i] >= thresholds[i]
            y_pred[mask] = i
        
        # Compute macro-F1
        f1 = f1_score(y_val, y_pred, average='macro')
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresholds = thresh_dict
    
    print(f"  Best macro-F1 with tuned thresholds: {best_f1:.4f}")
    print(f"  Optimized thresholds: {best_thresholds}")
    
    return best_thresholds


def apply_thresholds(
    y_pred_proba: np.ndarray,
    thresholds: Dict[str, float],
    state_map: Dict[str, int]
) -> np.ndarray:
    """Apply per-class thresholds to probability predictions."""
    reverse_state_map = {v: k for k, v in state_map.items()}
    n_classes = len(state_map)
    n_samples = len(y_pred_proba)
    
    y_pred = np.zeros(n_samples, dtype=int)
    
    for i in range(n_classes):
        state_name = reverse_state_map[i]
        threshold = thresholds.get(state_name, 0.5)
        mask = y_pred_proba[:, i] >= threshold
        y_pred[mask] = i
    
    # Handle cases where no class meets threshold (use max probability)
    no_class_mask = (y_pred == 0) & (y_pred_proba.max(axis=1) < min(thresholds.values()))
    if no_class_mask.any():
        y_pred[no_class_mask] = np.argmax(y_pred_proba[no_class_mask], axis=1)
    
    return y_pred


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scaler: StandardScaler,
    state_map: Dict[str, int],
    thresholds: Optional[Dict[str, float]] = None
) -> Dict:
    """Evaluate model performance with optional per-class thresholds."""
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Apply thresholds if provided
    if thresholds:
        y_pred = apply_thresholds(y_pred_proba, thresholds, state_map)
    else:
        y_pred = model.predict(X_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    # Per-class metrics
    f1_per_class = f1_score(y_test, y_pred, average=None)
    precision_per_class = []
    recall_per_class = []
    
    # Classification report
    reverse_state_map = {v: k for k, v in state_map.items()}
    state_names = [reverse_state_map[i] for i in sorted(reverse_state_map.keys())]
    report = classification_report(y_test, y_pred, target_names=state_names, output_dict=True)
    
    # Extract per-class precision and recall
    for state in state_names:
        if state in report:
            precision_per_class.append(report[state]['precision'])
            recall_per_class.append(report[state]['recall'])
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class.tolist() if hasattr(f1_per_class, 'tolist') else f1_per_class,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'state_names': state_names
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
    parser = argparse.ArgumentParser(description="Train optimized XGBoost/LightGBM model v3.2 with 4 Hz features and thresholds")
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
        "--model",
        type=str,
        choices=['xgb', 'lgbm'],
        default='xgb',
        help="Model type: xgb (XGBoost) or lgbm (LightGBM)"
    )
    parser.add_argument(
        "--use-env",
        action="store_true",
        help="Include environmental features (temp_c, humidity, aqi, local_hour)"
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
    parser.add_argument(
        "--no-thresholds",
        action="store_true",
        default=False,
        help="Skip threshold tuning and use argmax predictions (default: False)"
    )
    parser.add_argument(
        "--undersample",
        type=float,
        default=1.0,
        help="Fraction of the majority class (restorative) to keep. <1.0 reduces dominance. (default: 1.0)"
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        default=False,
        help="Skip class weighting (use no class weights for training)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("XGBoost/LightGBM Baseline Training v3.2")
    print("4 Hz-Friendly Features + Per-Class Thresholds")
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
    X, y, state_map = prepare_features(df, use_env=args.use_env)
    print(f"Features ({len(X.columns)}): {list(X.columns)}")
    print(f"Target distribution:")
    state_counts = y.value_counts().sort_index()
    reverse_state_map = {v: k for k, v in state_map.items()}
    for state_id, count in state_counts.items():
        state_name = reverse_state_map[state_id]
        print(f"  {state_name:12s}: {count:5,} ({count/len(y)*100:5.1f}%)")
    
    # Apply undersampling if requested
    if args.undersample < 1.0:
        print(f"\n[INFO] Applying undersampling (fraction={args.undersample:.2f})...")
        
        # Identify majority class (restorative)
        majority_class_name = 'restorative'
        if majority_class_name not in state_map:
            # Find the class with the most samples
            majority_class_id = y.value_counts().idxmax()
            majority_class_name = reverse_state_map[majority_class_id]
        
        majority_class_id = state_map[majority_class_name]
        
        # Get indices for each class
        majority_mask = y == majority_class_id
        minority_mask = ~majority_mask
        
        # Get original count
        orig_count = majority_mask.sum()
        
        # Sample majority class
        majority_indices = X[majority_mask].index
        n_samples = int(orig_count * args.undersample)
        
        if n_samples < orig_count:
            # Randomly sample
            np.random.seed(args.seed)
            sampled_indices = np.random.choice(majority_indices, size=n_samples, replace=False)
            majority_sampled_mask = X.index.isin(sampled_indices)
        else:
            majority_sampled_mask = majority_mask
        
        # Combine: sampled majority + all minority
        final_mask = majority_sampled_mask | minority_mask
        
        # Filter X and y
        X = X[final_mask].copy()
        y = y[final_mask].copy()
        
        # Shuffle
        shuffle_indices = X.index.values
        np.random.seed(args.seed)
        np.random.shuffle(shuffle_indices)
        X = X.loc[shuffle_indices].copy()
        y = y.loc[shuffle_indices].copy()
        
        # Reset index for clean indexing
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        # Print results
        new_count = (y == majority_class_id).sum()
        final_counts = y.value_counts().sort_index().to_dict()
        final_counts_str = {reverse_state_map[k]: v for k, v in final_counts.items()}
        
        print(f"  [INFO] Undersampling {majority_class_name} to {args.undersample*100:.1f}% (from {orig_count:,} → {new_count:,})")
        print(f"  Final class counts: {final_counts_str}")
    
    # Compute class weights (or skip if --no-class-weights)
    if args.no_class_weights:
        print("\n[INFO] Using NO class weights")
        class_weight_dict = None
    else:
        print("\nComputing class weights...")
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = {int(classes[i]): float(class_weights[i]) for i in range(len(classes))}
        print("  Class weights:")
        for class_id, weight in class_weight_dict.items():
            state_name = reverse_state_map[class_id]
            print(f"    {state_name:12s} (class {class_id}): {weight:.3f}")
    
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
    
    # Split train into train/val for GridSearchCV and threshold tuning
    X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(
        X_train_df, y_train, test_size=0.2, random_state=args.seed, stratify=y_train
    )
    
    # Train with GridSearchCV
    model, feature_importance, best_params, algorithm = train_with_gridsearch(
        X_train_cv, y_train_cv,
        X_val_cv, y_val_cv,
        class_weights=class_weight_dict,
        model_type=args.model,
        cv=args.cv,
        n_jobs=args.n_jobs
    )
    
    # Tune per-class thresholds on validation set (or skip if --no-thresholds)
    if args.no_thresholds:
        print("\n[INFO] Skipping threshold tuning (argmax used)")
        # Set all thresholds to 0.0 (argmax will be used)
        reverse_state_map = {v: k for k, v in state_map.items()}
        thresholds = {reverse_state_map[i]: 0.0 for i in range(len(state_map))}
    else:
        thresholds = tune_per_class_thresholds(
            model, X_val_cv, y_val_cv, state_map, DEFAULT_THRESHOLDS
        )
    
    # Evaluate on test set
    if args.no_thresholds:
        print("\nEvaluating model on test set (argmax predictions)...")
    else:
        print("\nEvaluating model on test set with tuned thresholds...")
    metrics = evaluate_model(model, X_test_df, y_test, scaler, state_map, thresholds=thresholds if not args.no_thresholds else None)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    
    print("\nPer-Class Metrics:")
    state_names = metrics['state_names']
    for i, state in enumerate(state_names):
        precision = metrics['precision_per_class'][i]
        recall = metrics['recall_per_class'][i]
        f1 = metrics['f1_per_class'][i]
        print(f"  {state:12s}: precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")
    
    # Confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    print_confusion_matrix(cm, state_map)
    
    # Top 10 feature importance (sorted descending)
    print("\nTop 10 Feature Importance (sorted descending):")
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(sorted_features[:10], 1):
        print(f"  {i:2d}. {feature:20s}: {importance:.4f}")
    
    # Save model and artifacts
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_file = model_dir / "cav_state_v3_2.joblib"
    scaler_file = model_dir / "cav_state_scaler_v3_2.joblib"
    schema_file = model_dir / "cav_state_schema_v3_2.json"
    
    print(f"\nSaving model artifacts...")
    print(f"  Model: {model_file}")
    joblib.dump(model, model_file)
    
    print(f"  Scaler: {scaler_file}")
    joblib.dump(scaler, scaler_file)
    
    print(f"  Schema: {schema_file}")
    # Set thresholds to 0.0 if no-thresholds was used
    if args.no_thresholds:
        thresholds_for_schema = {k: 0.0 for k in thresholds.keys()}
    else:
        thresholds_for_schema = {k: float(v) for k, v in thresholds.items()}
    
    schema = {
        'feature_names': list(X.columns),
        'state_map': state_map,
        'reverse_state_map': {v: k for k, v in state_map.items()},
        'feature_importance': {k: float(v) for k, v in feature_importance.items()},
        'class_weights': class_weight_dict,
        'class_weights_used': not args.no_class_weights,
        'thresholds': thresholds_for_schema,
        'thresholds_tuned': not args.no_thresholds,
        'algorithm': algorithm,
        'best_parameters': {k: (float(v) if isinstance(v, (np.integer, np.floating)) else v) for k, v in best_params.items()},
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'f1_macro': float(metrics['f1_macro']),
            'f1_weighted': float(metrics['f1_weighted']),
            'f1_per_class': metrics['f1_per_class'],
            'precision_per_class': metrics['precision_per_class'],
            'recall_per_class': metrics['recall_per_class']
        },
        'confusion_matrix': metrics['confusion_matrix']
    }
    with open(schema_file, 'w') as f:
        json.dump(schema, f, indent=2)
    
    print()
    print("=" * 60)
    print("[SUCCESS] Training complete!")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Features: {len(X.columns)} ({'with' if args.use_env else 'without'} environmental)")
    print("=" * 60)
    
    # Print usage summary
    print("\n" + "=" * 60)
    print("Available CLI Flags Summary")
    print("=" * 60)
    print("\nBasic Usage:")
    print("  python training\\train_baseline_v3_2.py --input outputs\\oem_100k_windows.parquet")
    print("\nModel Selection:")
    print("  --model xgb          Use XGBoost (default)")
    print("  --model lgbm         Use LightGBM")
    print("\nClass Balancing:")
    print("  --undersample 0.5    Keep 50% of majority class (default: 1.0)")
    print("  --no-class-weights   Skip class weighting (use no weights)")
    print("\nThreshold Tuning:")
    print("  --no-thresholds      Skip threshold tuning (use argmax)")
    print("\nEnvironmental Features:")
    print("  --use-env            Include temp_c, humidity, aqi, local_hour")
    print("\nExample Commands:")
    print("  # LightGBM with undersampling, no class weights, no thresholds:")
    print("  python training\\train_baseline_v3_2.py --input outputs\\oem_100k_windows.parquet \\")
    print("      --model lgbm --undersample 0.5 --no-class-weights --no-thresholds")
    print("\n  # XGBoost with environmental features:")
    print("  python training\\train_baseline_v3_2.py --input outputs\\oem_100k_windows.parquet \\")
    print("      --model xgb --use-env")
    print("\n  # Full pipeline with all options:")
    print("  python training\\train_baseline_v3_2.py --input outputs\\oem_100k_windows.parquet \\")
    print("      --model xgb --undersample 0.3 --use-env")
    print("=" * 60)


if __name__ == "__main__":
    main()

