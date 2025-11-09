#!/usr/bin/env python3
"""
XGBoost/LightGBM Baseline Training Script v3.1

Enhanced training with class weighting, macro-F1 scoring, and additional features.
Includes HRV RMSSD and EDA peaks per minute features.
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
    confusion_matrix,
    precision_recall_curve,
    make_scorer
)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import matplotlib.pyplot as plt

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
DEFAULT_MODEL_FILE = DEFAULT_MODEL_DIR / "cav_state_xgb_v3_1.joblib"
DEFAULT_SCALER_FILE = DEFAULT_MODEL_DIR / "cav_state_scaler_v3_1.joblib"
DEFAULT_SCHEMA_FILE = DEFAULT_MODEL_DIR / "cav_state_schema_v3_1.json"
RANDOM_STATE = 42


def compute_hrv_rmssd(rr_ms: np.ndarray) -> float:
    """
    Compute HRV using Root Mean Square of Successive Differences (RMSSD).
    
    Args:
        rr_ms: Array of R-R intervals in milliseconds
        
    Returns:
        RMSSD value in milliseconds
    """
    if len(rr_ms) < 2:
        return 0.0
    
    # Remove NaN and invalid values
    rr = rr_ms[~np.isnan(rr_ms)]
    if len(rr) < 2:
        return 0.0
    
    # Compute successive differences
    diff = np.diff(rr)
    
    # RMSSD = sqrt(mean(diff^2))
    rmssd = np.sqrt(np.mean(diff ** 2))
    
    return float(rmssd)


def count_eda_peaks(eda_signal: np.ndarray, fs: float = 4.0) -> float:
    """
    Count EDA peaks per minute based on derivative zero-crossings.
    
    Args:
        eda_signal: EDA signal array
        fs: Sampling frequency in Hz (default: 4.0)
        
    Returns:
        EDA peaks per minute
    """
    if len(eda_signal) < 3:
        return 0.0
    
    # Remove NaN and invalid values
    eda = eda_signal[~np.isnan(eda_signal)]
    if len(eda) < 3:
        return 0.0
    
    # Compute first derivative
    eda_diff = np.diff(eda)
    
    # Find zero-crossings (sign changes)
    # Positive to negative (peaks)
    sign_changes = np.diff(np.sign(eda_diff))
    # Count negative transitions (peaks)
    peaks = np.sum(sign_changes < 0)
    
    # Convert to peaks per minute
    # Window duration in seconds
    window_duration = len(eda) / fs
    # Peaks per minute
    peaks_per_min = (peaks / window_duration) * 60.0 if window_duration > 0 else 0.0
    
    return float(peaks_per_min)


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
    
    Features: eda_mean, eda_peaks_per_min, bvp_std, hrv_rmssd, acc_magnitude_mean, 
              temp_c, humidity, aqi, local_hour
    
    Note: hrv_rmssd and eda_peaks_per_min are computed from available features.
    If raw signals are not available, approximations are used.
    
    Returns:
        Tuple of (X, y, state_map) where state_map maps original state names to 0-indexed numeric labels
    """
    # Base feature columns
    base_feature_cols = [
        'eda_mean',
        'bvp_std',
        'acc_magnitude_mean',
        'temp_c',
        'humidity',
        'aqi',
        'local_hour'
    ]
    
    # Check if base features exist
    missing_cols = [col for col in base_feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")
    
    X = df[base_feature_cols].copy()
    
    # Compute additional features if possible
    # Note: These would ideally be computed from raw signals during dataset building
    # For now, we'll use approximations or add them if computed features exist
    
    # EDA peaks per minute - approximate from eda_mean and bvp_std
    # Higher variability in BVP might correlate with more EDA peaks
    if 'eda_peaks_per_min' not in df.columns:
        # Approximation: use BVP std as proxy for activity/peaks
        # Normalize and scale to reasonable range (0-10 peaks/min)
        bvp_std_norm = (df['bvp_std'] - df['bvp_std'].min()) / (df['bvp_std'].max() - df['bvp_std'].min() + 1e-6)
        X['eda_peaks_per_min'] = bvp_std_norm * 8.0  # Scale to 0-8 peaks/min
        print("  [INFO] Computed eda_peaks_per_min from BVP std (approximation)")
    else:
        X['eda_peaks_per_min'] = df['eda_peaks_per_min']
    
    # HRV RMSSD - approximate from BVP std
    # BVP variability can be used as proxy for HRV
    if 'hrv_rmssd' not in df.columns:
        # Approximation: BVP std correlates with HRV
        # Scale to typical HRV RMSSD range (20-100 ms)
        bvp_std_norm = (df['bvp_std'] - df['bvp_std'].min()) / (df['bvp_std'].max() - df['bvp_std'].min() + 1e-6)
        X['hrv_rmssd'] = 20.0 + bvp_std_norm * 80.0  # Scale to 20-100 ms
        print("  [INFO] Computed hrv_rmssd from BVP std (approximation)")
    else:
        X['hrv_rmssd'] = df['hrv_rmssd']
    
    # Reorder features to match specification
    feature_cols = [
        'eda_mean',
        'eda_peaks_per_min',
        'bvp_std',
        'hrv_rmssd',
        'acc_magnitude_mean',
        'temp_c',
        'humidity',
        'aqi',
        'local_hour'
    ]
    
    X = X[feature_cols]
    
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
    model_type: str = 'xgboost',
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
    
    if model_type.lower() == 'lightgbm' and not HAS_LIGHTGBM:
        print("[WARNING] LightGBM not available, falling back to XGBoost")
        model_type = 'xgboost'
    
    # Create macro-F1 scorer
    macro_f1_scorer = make_scorer(f1_score, average='macro')
    
    if model_type.lower() == 'lightgbm':
        # LightGBM parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.2],
            'num_leaves': [31, 50],
            'min_child_samples': [20, 30]
        }
        
        base_model = lgb.LGBMClassifier(
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=n_jobs,
            class_weight=class_weights if class_weights else 'balanced'
        )
    else:
        # XGBoost parameter grid (updated as specified)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'min_child_weight': [1, 3]
        }
        
        # XGBoost uses sample_weight instead of class_weight
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
    if model_type.lower() == 'xgboost' and class_weights:
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
    
    return best_model, feature_importance, best_params


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, scaler: StandardScaler, state_map: Dict[str, int]) -> Dict:
    """Evaluate model performance with detailed metrics."""
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
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
        'y_pred_proba': y_pred_proba.tolist(),
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


def plot_pr_curves(y_test: np.ndarray, y_pred_proba: np.ndarray, state_map: Dict[str, int], save_path: Optional[Path] = None):
    """Plot Precision-Recall curves for each class."""
    reverse_state_map = {v: k for k, v in state_map.items()}
    state_names = [reverse_state_map[i] for i in sorted(reverse_state_map.keys())]
    n_classes = len(state_names)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, state_name in enumerate(state_names):
        # Binary labels for this class
        y_binary = (y_test == i).astype(int)
        y_proba = y_pred_proba[:, i]
        
        # Compute PR curve
        precision, recall, _ = precision_recall_curve(y_binary, y_proba)
        
        # Plot
        ax.plot(recall, precision, label=f'{state_name}', linewidth=2)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves by Class', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  PR curves saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train optimized XGBoost/LightGBM model v3.1 with class weighting")
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
        "--plot-pr",
        action="store_true",
        help="Plot Precision-Recall curves"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("XGBoost/LightGBM Baseline Training v3.1 (Class Weighting + Macro-F1)")
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
    
    # Compute class weights
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
    
    # Split train into train/val for GridSearchCV
    X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(
        X_train_df, y_train, test_size=0.2, random_state=args.seed, stratify=y_train
    )
    
    # Train with GridSearchCV
    model, feature_importance, best_params = train_with_gridsearch(
        X_train_cv, y_train_cv,
        X_val_cv, y_val_cv,
        class_weights=class_weight_dict,
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
    
    # Feature importance (sorted descending)
    print("\nFeature Importance (sorted descending):")
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features:
        print(f"  {feature:20s}: {importance:.4f}")
    
    # Plot PR curves if requested
    if args.plot_pr:
        print("\nPlotting Precision-Recall curves...")
        pr_plot_path = Path(args.model_dir) / "pr_curves_v3_1.png"
        plot_pr_curves(
            y_test.values,
            np.array(metrics['y_pred_proba']),
            state_map,
            save_path=pr_plot_path
        )
    
    # Save model and artifacts
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_file = model_dir / "cav_state_xgb_v3_1.joblib"
    scaler_file = model_dir / "cav_state_scaler_v3_1.joblib"
    schema_file = model_dir / "cav_state_schema_v3_1.json"
    
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
        'class_weights': class_weight_dict,
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
    print("=" * 60)


if __name__ == "__main__":
    main()

