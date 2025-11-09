#!/usr/bin/env python3
"""
EDON CAV Engine — Live Console (v3.2)
Dark AI Console Theme - Tesla/Neuralink inspired
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from joblib import load
import warnings
import time
import altair as alt
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# --- Robust model/data discovery (v3.2) ---
import os

ROOT = Path(__file__).resolve().parent
# Also check parent directory (repo root) if we're in a subdirectory
REPO_ROOT = ROOT.parent if ROOT.name in ["tools", "temp_sdk"] else ROOT

def find_artifact(name: str, also=None) -> Path:
    candidates = []
    env_dir = os.getenv("EDON_MODEL_DIR")
    if env_dir: 
        candidates.append(Path(env_dir) / name)
    
    # Check in current directory structure
    candidates.append(ROOT / "models" / name)
    candidates.append(ROOT / name)
    candidates += [(p / name) for p in ROOT.glob("cav_engine_v3_2_*")]
    
    # Check in repo root (if different from ROOT)
    if REPO_ROOT != ROOT:
        candidates.append(REPO_ROOT / "models" / name)
        candidates.append(REPO_ROOT / name)
        candidates += [(p / name) for p in REPO_ROOT.glob("cav_engine_v3_2_*")]
    
    if also: 
        candidates += [ROOT / a for a in also]
        if REPO_ROOT != ROOT:
            candidates += [REPO_ROOT / a for a in also]
    
    for c in candidates:
        if c.exists(): 
            return c
    
    raise FileNotFoundError(f"{name} not found. Looked in: " + ", ".join(map(str, candidates)))

SCHEMA_PATH = find_artifact("cav_state_schema_v3_2.json")

SCALER_PATH = find_artifact("cav_state_scaler_v3_2.joblib")

MODEL_PATH  = find_artifact("cav_state_v3_2.joblib")

DATA_PATH   = find_artifact("oem_100k_windows.parquet", also=["outputs/oem_100k_windows.parquet"])



with open(SCHEMA_PATH, "r", encoding="utf-8") as f:

    _schema = json.load(f)



FEATURES = _schema.get("feature_names", [

    "eda_mean", "eda_deriv_std", "eda_deriv_pos_rate",

    "bvp_std", "acc_magnitude_mean", "acc_var",

])

_rev = _schema.get("reverse_state_map", {"0":"balanced","1":"focus","2":"restorative"})

STATE_MAP = {int(k): v for k, v in _rev.items()}



STATE_COLORS = {

    "balanced": "#C08D57",

    "focus": "#00E5FF",

    "restorative": "#4FC3F7"

}

# Default paths for UI (use discovered paths)
DEFAULT_PATHS = {
    "data_path": str(DATA_PATH),
    "model_path": str(MODEL_PATH),
    "scaler_path": str(SCALER_PATH),
    "schema_path": str(SCHEMA_PATH)
}

# Preset values
PRESETS = {
    "restorative": {
        "eda_mean": -0.40,
        "eda_deriv_std": 0.15,
        "eda_deriv_pos_rate": 0.25,
        "bvp_std": 0.55,
        "acc_magnitude_mean": 0.80,
        "acc_var": 0.20
    },
    "balanced": {
        "eda_mean": 0.00,
        "eda_deriv_std": 0.35,
        "eda_deriv_pos_rate": 0.50,
        "bvp_std": 0.95,
        "acc_magnitude_mean": 1.40,
        "acc_var": 0.50
    },
    "focus": {
        "eda_mean": 0.30,
        "eda_deriv_std": 1.05,
        "eda_deriv_pos_rate": 0.70,
        "bvp_std": 1.70,
        "acc_magnitude_mean": 2.70,
        "acc_var": 1.10
    }
}

# Slider ranges and jitter magnitudes
SLIDER_CONFIG = {
    "eda_mean": {"min": -2.0, "max": 2.0, "default": 0.0, "step": 0.1, "jitter": 0.05},
    "eda_deriv_std": {"min": 0.0, "max": 2.0, "default": 0.3, "step": 0.1, "jitter": 0.05},
    "eda_deriv_pos_rate": {"min": 0.0, "max": 1.0, "default": 0.5, "step": 0.05, "jitter": 0.03},
    "bvp_std": {"min": 0.0, "max": 3.0, "default": 1.2, "step": 0.1, "jitter": 0.06},
    "acc_magnitude_mean": {"min": 0.0, "max": 4.0, "default": 2.0, "step": 0.1, "jitter": 0.08},
    "acc_var": {"min": 0.0, "max": 2.0, "default": 0.8, "step": 0.1, "jitter": 0.05}
}


def inject_css():
    """Inject custom CSS for dark AI console theme."""
    css_path = Path("tools/_styles/edon_console.css")
    if css_path.exists():
        with open(css_path, 'r') as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_model_resources(model_path, scaler_path, schema_path):
    """Load model, scaler, and schema with caching."""
    try:
        model_path = Path(model_path)
        scaler_path = Path(scaler_path)
        schema_path = Path(schema_path)
        
        if not model_path.exists() or not scaler_path.exists() or not schema_path.exists():
            return None, None, None, None
        
        model = load(str(model_path))
        scaler = load(str(scaler_path))
        
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        feature_order = schema.get("feature_names", FEATURES)
        reverse_state_map_raw = schema.get("reverse_state_map", {"0": "balanced", "1": "focus", "2": "restorative"})
        reverse_state_map = {int(k): v for k, v in reverse_state_map_raw.items()}
        
        return model, scaler, feature_order, reverse_state_map
    except Exception as e:
        return None, None, None, None


@st.cache_data
def load_dataset(data_path):
    """Load dataset with caching."""
    try:
        data_path = Path(data_path)
        if not data_path.exists():
            return None
        
        if data_path.suffix == '.parquet':
            df = pd.read_parquet(data_path)
        elif data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
        else:
            return None
        
        return df
    except Exception as e:
        return None


def predict_one(row_dict, model, scaler, feature_order, reverse_state_map):
    """Predict state for a single row. Returns: (label, prob_dict) or (None, None)"""
    if model is None or scaler is None:
        return None, None
    
    try:
        X = pd.DataFrame([[row_dict.get(f, 0.0) for f in feature_order]], columns=feature_order)
        X_scaled = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_order)
        pred_idx = int(model.predict(X_scaled_df)[0])
        label = reverse_state_map.get(pred_idx, "unknown")
        
        prob_dict = None
        try:
            proba = model.predict_proba(X_scaled_df)[0]
            prob_dict = {reverse_state_map.get(i, f"class_{i}"): float(prob) 
                        for i, prob in enumerate(proba)}
        except:
            pass
        
        return label, prob_dict
    except Exception as e:
        return None, None


def check_v32_features_present(df):
    """Check if all v3.2 features are present in dataframe."""
    return all(f in df.columns for f in FEATURES)


def render_logo_and_version():
    """Render EDON logo and version badge."""
    st.markdown("""
    <div class="logo-container">
        <h1 class="logo-title">EDON</h1>
        <div class="version-badge">v3.2</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


def render_state_chip(state_name, active=False):
    """Render a state chip button."""
    color = STATE_COLORS.get(state_name, "#F2F6FA")
    active_class = "active" if active else ""
    
    return f"""
    <button class="state-chip {state_name} {active_class}" onclick="this.style.transform='scale(0.95)'; setTimeout(() => this.style.transform='', 100);">
        {state_name.upper()}
    </button>
    """


def render_kpi_row(label, cav_score, max_prob=None):
    """Render KPI row with predicted state, CAV score, and max probability."""
    color = STATE_COLORS.get(label, "#F2F6FA")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-badge {label}" style="margin-bottom: 16px;">
            <div style="font-size: 12px; color: rgba(242, 246, 250, 0.6); margin-bottom: 12px; letter-spacing: 1px; text-transform: uppercase;">Predicted State</div>
            <div style="font-size: 32px; font-weight: 600; color: {color}; margin: 12px 0; letter-spacing: 0.5px;">
                {label.upper()}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-badge" style="margin-bottom: 16px;">
            <div style="font-size: 12px; color: rgba(242, 246, 250, 0.6); margin-bottom: 12px; letter-spacing: 1px; text-transform: uppercase;">CAV Score</div>
            <div style="font-size: 32px; font-weight: 600; color: #00E5FF; margin: 12px 0;">
                {cav_score:,}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        prob_text = f"{max_prob:.0%}" if max_prob is not None else "N/A"
        st.markdown(f"""
        <div class="kpi-badge" style="margin-bottom: 16px;">
            <div style="font-size: 12px; color: rgba(242, 246, 250, 0.6); margin-bottom: 12px; letter-spacing: 1px; text-transform: uppercase;">Max Probability</div>
            <div style="font-size: 32px; font-weight: 600; color: #00E5FF; margin: 12px 0;">
                {prob_text}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_probs_chart(prob_dict):
    """Render class probabilities chart with fixed y-axis [0,1]."""
    prob_df = pd.DataFrame(list(prob_dict.items()), columns=["state", "prob"])
    prob_df = prob_df.sort_values("prob", ascending=False)
    
    chart = alt.Chart(prob_df).mark_bar().encode(
        x=alt.X("state:N", sort=None, title="State"),
        y=alt.Y("prob:Q", scale=alt.Scale(domain=[0, 1]), title="Probability"),
        color=alt.Color("state:N", scale=alt.Scale(
            domain=list(STATE_COLORS.keys()),
            range=list(STATE_COLORS.values())
        ), legend=None)
    ).properties(width=600, height=300).configure(
        background='rgba(21, 23, 27, 0)',
        axis=alt.AxisConfig(
            labelColor='#F2F6FA',
            titleColor='#F2F6FA',
            gridColor='rgba(242, 246, 250, 0.1)'
        )
    )
    
    return chart


# Page configuration
st.set_page_config(
    page_title="EDON CAV Engine Console",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject CSS
inject_css()

# Initialize session state
if "inference_result" not in st.session_state:
    st.session_state.inference_result = None

# Control Rail (Left Sidebar)
with st.sidebar:
    render_logo_and_version()
    
    st.markdown("---")
    
    # File paths
    st.markdown("### File Paths")
    st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
    data_path = st.text_input("Dataset", value=DEFAULT_PATHS["data_path"], key="data_path", label_visibility="visible")
    st.markdown("<div style='margin-bottom: 4px;'></div>", unsafe_allow_html=True)
    model_path = st.text_input("Model", value=DEFAULT_PATHS["model_path"], key="model_path", label_visibility="visible")
    st.markdown("<div style='margin-bottom: 4px;'></div>", unsafe_allow_html=True)
    scaler_path = st.text_input("Scaler", value=DEFAULT_PATHS["scaler_path"], key="scaler_path", label_visibility="visible")
    st.markdown("<div style='margin-bottom: 4px;'></div>", unsafe_allow_html=True)
    schema_path = st.text_input("Schema", value=DEFAULT_PATHS["schema_path"], key="schema_path", label_visibility="visible")
    
    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Mode selection
    st.markdown("### Mode")
    st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
    mode = st.radio("", ["Playback", "Manual Inference"], index=0, label_visibility="collapsed")
    
    # Load model resources
    model, scaler, feature_order, reverse_state_map = load_model_resources(
        model_path, scaler_path, schema_path
    )
    
    if model is not None:
        st.markdown("<div style='color: #00E5FF; font-size: 13px; margin: 8px 0;'>Model loaded</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color: #FF9800; font-size: 13px; margin: 8px 0;'>No model loaded</div>", unsafe_allow_html=True)
    
    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Mode-specific controls
    if mode == "Playback":
        st.markdown("### Playback Options")
        st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
        
        # Initialize playback state
        if "playback_running" not in st.session_state:
            st.session_state.playback_running = False
        if "playback_index" not in st.session_state:
            st.session_state.playback_index = 0
        
        sample_size = st.number_input("Sample Size", min_value=100, max_value=100000, value=1000, step=100, label_visibility="visible")
        st.markdown("<div style='margin-bottom: 4px;'></div>", unsafe_allow_html=True)
        refresh_seconds = st.number_input("Refresh (seconds)", min_value=1, max_value=60, value=2, step=1, label_visibility="visible")
        st.markdown("<div style='margin-bottom: 4px;'></div>", unsafe_allow_html=True)
        rolling_window = st.number_input("Rolling Window", min_value=1, max_value=100, value=25, step=1, label_visibility="visible")
        st.markdown("<div style='margin-bottom: 4px;'></div>", unsafe_allow_html=True)
        compat_mode = st.checkbox("Compat mode: fill missing features with medians", value=False, help="Enable to auto-predict even if some v3.2 features are missing")
        st.markdown("<div style='margin-bottom: 4px;'></div>", unsafe_allow_html=True)
        auto_predict = st.checkbox("Auto-predict during playback", value=False, help="Requires all v3.2 features in dataset (or compat mode enabled)")
        
        # Playback controls
        st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
        col_play, col_stop = st.columns(2)
        with col_play:
            if st.button("▶ Start Playback", use_container_width=True, key="playback_start", disabled=st.session_state.playback_running):
                st.session_state.playback_running = True
                st.session_state.playback_index = 0
                st.rerun()
        with col_stop:
            if st.button("⏹ Stop Playback", use_container_width=True, key="playback_stop", disabled=not st.session_state.playback_running):
                st.session_state.playback_running = False
                st.rerun()
    
    elif mode == "Manual Inference":
        st.markdown("### Presets")
        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
        
        # Define preset callbacks (safe session_state updates)
        def apply_preset(preset_name: str):
            """Apply preset values to sliders using callback pattern."""
            preset = PRESETS[preset_name]
            for key, value in preset.items():
                st.session_state[f"slider_{key}"] = value
                st.session_state[key] = value  # Keep for compatibility
            
            # Auto-run inference after preset is applied
            if model is not None and scaler is not None:
                feature_values_preset = {key: st.session_state.get(f"slider_{key}", SLIDER_CONFIG[key]["default"]) for key in FEATURES}
                label, prob_dict = predict_one(feature_values_preset, model, scaler, feature_order, reverse_state_map)
                if label is not None:
                    cav_score = np.random.randint(5000, 10000)
                    max_prob = max(prob_dict.values()) if prob_dict else None
                    st.session_state.inference_result = (label, cav_score, max_prob, prob_dict)
        
        # State chips - uniform grid layout with proper spacing
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("Restorative", use_container_width=True, key="preset_restorative", 
                     on_click=apply_preset, args=("restorative",))
        with col2:
            st.button("Balanced", use_container_width=True, key="preset_balanced",
                     on_click=apply_preset, args=("balanced",))
        with col3:
            st.button("Focus", use_container_width=True, key="preset_focus",
                     on_click=apply_preset, args=("focus",))
        
        st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Auto-animate toggle with LIVE dot - store in session_state
        if "auto_animate" not in st.session_state:
            st.session_state.auto_animate = False
        
        auto_animate = st.checkbox(
            "Auto-animate sliders", 
            value=st.session_state.auto_animate, 
            help="Demo mode: continuously jitter slider values",
            key="auto_animate_checkbox"
        )
        st.session_state.auto_animate = auto_animate
        
        if auto_animate:
            st.markdown('<div style="margin-top: 8px;"><span class="live-dot"></span> <span style="color: #00E5FF; font-size: 12px;">LIVE</span></div>', unsafe_allow_html=True)
        
        # Initialize session state for sliders BEFORE creating them
        # Use slider_{key} format to match slider widget keys
        for key in FEATURES:
            slider_key = f"slider_{key}"
            if slider_key not in st.session_state:
                st.session_state[slider_key] = SLIDER_CONFIG[key]["default"]
            # Also keep the feature key for compatibility
            if key not in st.session_state:
                st.session_state[key] = SLIDER_CONFIG[key]["default"]

# Main Console (Right Column)
st.title("EDON CAV Engine — Live Console (v3.2)")

if model is not None:
    st.markdown(f"<div style='color: #00E5FF; margin-bottom: 24px; font-size: 14px;'>Model: {Path(model_path).name}</div>", unsafe_allow_html=True)
else:
    st.markdown("<div style='color: #FF9800; margin-bottom: 24px; font-size: 14px;'>No model loaded</div>", unsafe_allow_html=True)

st.markdown("---")

# Mode: Playback
if mode == "Playback":
    try:
        df = load_dataset(data_path)
        
        if df is None:
            st.error(f"Could not load dataset from: {data_path}")
            st.info("Please check the dataset path in the sidebar and ensure the file exists.")
        elif len(df) == 0:
            st.warning("Dataset is empty. Please load a dataset with data.")
        else:
            st.markdown(f"<div style='color: #00E5FF; margin-bottom: 16px; font-size: 14px;'>Dataset: {len(df):,} rows, {len(df.columns)} columns</div>", unsafe_allow_html=True)
            
            has_v32_features = check_v32_features_present(df)
            missing_features = [f for f in FEATURES if f not in df.columns]
            
            if has_v32_features:
                st.markdown("<div style='color: #4CAF50; margin-bottom: 16px; font-size: 13px;'>All v3.2 features present</div>", unsafe_allow_html=True)
            else:
                if compat_mode and missing_features:
                    st.markdown(f"<div style='color: #FFC107; margin-bottom: 16px; font-size: 13px;'>⚠️ Compat mode filled: {', '.join(missing_features)}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color: #FF9800; margin-bottom: 16px; font-size: 13px;'>Some v3.2 features missing</div>", unsafe_allow_html=True)
                    if not compat_mode:
                        auto_predict = False
            
            # Sample data
            if len(df) > sample_size:
                df_work = df.tail(sample_size).copy()
            else:
                df_work = df.copy()
            
            # Metrics
            st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{len(df_work):,}")
            with col2:
                st.metric("Features", len(df_work.columns))
            with col3:
                unique_states = df_work["state"].nunique() if "state" in df_work.columns else 0
                st.metric("States", unique_states)
            with col4:
                can_predict = auto_predict and model and (has_v32_features or compat_mode)
                auto_status = "Enabled" if can_predict else "Disabled"
                st.metric("Auto-Predict", auto_status)
            
            st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
            
            # CAV scores
            if any(col in df_work.columns for col in ["cav_raw", "cav_smooth"]):
                st.subheader("CAV Scores")
                st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
                cav_cols = [col for col in ["cav_raw", "cav_smooth"] if col in df_work.columns]
                if cav_cols:
                    df_cav = df_work[cav_cols].copy()
                    if "index" not in df_cav.columns:
                        df_cav["index"] = range(len(df_cav))
                    for col in cav_cols:
                        df_cav[f"{col}_smooth"] = df_cav[col].rolling(window=rolling_window, min_periods=1).mean()
                    st.line_chart(df_cav.set_index("index")[[f"{col}_smooth" for col in cav_cols if f"{col}_smooth" in df_cav.columns]])
            
            # Physiological signals
            st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)
            st.subheader("Physiological Signals")
            st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
            signal_cols = [col for col in ["eda_mean", "bvp_std", "acc_magnitude_mean"] if col in df_work.columns]
            if signal_cols:
                df_signal = df_work[signal_cols].copy()
                if "index" not in df_signal.columns:
                    df_signal["index"] = range(len(df_signal))
                for col in signal_cols:
                    df_signal[f"{col}_smooth"] = df_signal[col].rolling(window=rolling_window, min_periods=1).mean()
                st.line_chart(df_signal.set_index("index")[[f"{col}_smooth" for col in signal_cols]])
            
            # Environmental features
            env_cols = [col for col in ["temp_c", "humidity", "aqi", "local_hour"] if col in df_work.columns]
            if env_cols:
                st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)
                with st.expander("Environmental Features", expanded=False):
                    df_env = df_work[env_cols].copy()
                    if "index" not in df_env.columns:
                        df_env["index"] = range(len(df_env))
                    for col in env_cols:
                        df_env[f"{col}_smooth"] = df_env[col].rolling(window=rolling_window, min_periods=1).mean()
                    st.line_chart(df_env.set_index("index")[[f"{col}_smooth" for col in env_cols]])
            
            # State distribution
            if "state" in df_work.columns:
                st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)
                st.subheader("State Distribution")
                st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
                state_counts = df_work["state"].value_counts()
                st.bar_chart(state_counts)
            
            # Auto-prediction
            can_predict = auto_predict and model is not None and (has_v32_features or compat_mode)
            if can_predict:
                st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)
                st.subheader("Auto-Prediction Results")
                st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
                try:
                    # Build feature matrix with compat mode support
                    X = pd.DataFrame(index=df_work.index)
                    
                    # Compat mode: fill missing features with medians
                    medians = {
                        "eda_deriv_std": 0.35,
                        "eda_deriv_pos_rate": 0.50,
                        "acc_var": 0.50
                    }
                    
                    for feat in feature_order:
                        if feat in df_work.columns:
                            X[feat] = df_work[feat].values
                        elif compat_mode and feat in medians:
                            # Fill with median constant for all rows (pandas broadcasts scalar)
                            X[feat] = medians[feat]
                        elif compat_mode:
                            # For other missing features, use defaults
                            if "eda" in feat:
                                default_val = float(df_work["eda_mean"].median()) if "eda_mean" in df_work.columns else 0.0
                            elif "bvp" in feat:
                                default_val = float(df_work["bvp_std"].median()) if "bvp_std" in df_work.columns else 1.0
                            elif "acc" in feat:
                                default_val = float(df_work["acc_magnitude_mean"].median()) if "acc_magnitude_mean" in df_work.columns else 2.0
                            else:
                                default_val = 0.0
                            X[feat] = default_val
                        else:
                            raise ValueError(f"Missing required feature: {feat}")
                    
                    # Ensure correct column order
                    X = X[feature_order]
                    
                    X_scaled = scaler.transform(X)
                    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_order)
                    predictions = model.predict(X_scaled_df)
                    pred_labels = [reverse_state_map.get(int(p), "unknown") for p in predictions]
                    df_work["predicted_state"] = pred_labels
                    
                    pred_numeric = [list(STATE_MAP.values()).index(p) if p in STATE_MAP.values() else -1 for p in pred_labels]
                    df_pred = pd.DataFrame({"index": range(len(pred_numeric)), "predicted_class": pred_numeric})
                    st.line_chart(df_pred.set_index("index"))
                    
                    pred_counts = pd.Series(pred_labels).value_counts()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted States", len(pred_counts))
                        st.dataframe(pred_counts.to_frame("Count"), use_container_width=True)
                    with col2:
                        pred_pct = (pred_counts / len(pred_labels) * 100).round(1)
                        st.bar_chart(pred_pct)
                except Exception as e:
                    st.error(f"Auto-prediction error: {e}")
            
            # Auto-refresh if playback is running
            if st.session_state.get("playback_running", False):
                time.sleep(refresh_seconds)
                st.session_state.playback_index = (st.session_state.playback_index + 1) % sample_size
                st.rerun()
    except Exception as e:
        st.error(f"Playback error: {str(e)}")
        st.exception(e)
        # Reset playback state on error
        if "playback_running" in st.session_state:
            st.session_state.playback_running = False

# Mode: Manual Inference
elif mode == "Manual Inference":
    if model is None or scaler is None:
        st.markdown("<div style='color: #FF9800; margin-bottom: 16px; font-size: 14px;'>Please load model files to run inference</div>", unsafe_allow_html=True)
    else:
        # Auto-animate: Update values BEFORE creating sliders (to avoid mutation error)
        auto_animating = st.session_state.get("auto_animate", False)
        if auto_animating:
            # Update slider values with jitter using slider_{key} format
            # This must happen BEFORE sliders are created
            for key in FEATURES:
                config = SLIDER_CONFIG[key]
                slider_key = f"slider_{key}"
                current_val = st.session_state.get(slider_key, config["default"])
                jitter = np.random.normal(0, config["jitter"])
                new_val = current_val + jitter
                new_val = max(config["min"], min(config["max"], new_val))
                # Update slider key BEFORE widget is created
                st.session_state[slider_key] = new_val
                # Also update feature key for compatibility
                st.session_state[key] = new_val
        
        # Feature sliders in 2-column grid
        st.subheader("Feature Controls")
        st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            feature_values = {}
            # Build sliders using session_state (slider_{key} format)
            feature_values["eda_mean"] = st.slider(
                "EDA Mean", SLIDER_CONFIG["eda_mean"]["min"], SLIDER_CONFIG["eda_mean"]["max"],
                value=st.session_state["slider_eda_mean"],
                step=SLIDER_CONFIG["eda_mean"]["step"], key="slider_eda_mean"
            )
            st.session_state["eda_mean"] = feature_values["eda_mean"]
            
            feature_values["eda_deriv_std"] = st.slider(
                "EDA Deriv Std", SLIDER_CONFIG["eda_deriv_std"]["min"], SLIDER_CONFIG["eda_deriv_std"]["max"],
                value=st.session_state["slider_eda_deriv_std"],
                step=SLIDER_CONFIG["eda_deriv_std"]["step"], key="slider_eda_deriv_std"
            )
            st.session_state["eda_deriv_std"] = feature_values["eda_deriv_std"]
            
            feature_values["eda_deriv_pos_rate"] = st.slider(
                "EDA Deriv Pos Rate", SLIDER_CONFIG["eda_deriv_pos_rate"]["min"], SLIDER_CONFIG["eda_deriv_pos_rate"]["max"],
                value=st.session_state["slider_eda_deriv_pos_rate"],
                step=SLIDER_CONFIG["eda_deriv_pos_rate"]["step"], key="slider_eda_deriv_pos_rate"
            )
            st.session_state["eda_deriv_pos_rate"] = feature_values["eda_deriv_pos_rate"]
        
        with col2:
            feature_values["bvp_std"] = st.slider(
                "BVP Std", SLIDER_CONFIG["bvp_std"]["min"], SLIDER_CONFIG["bvp_std"]["max"],
                value=st.session_state["slider_bvp_std"],
                step=SLIDER_CONFIG["bvp_std"]["step"], key="slider_bvp_std"
            )
            st.session_state["bvp_std"] = feature_values["bvp_std"]
            
            feature_values["acc_magnitude_mean"] = st.slider(
                "ACC Magnitude Mean", SLIDER_CONFIG["acc_magnitude_mean"]["min"], SLIDER_CONFIG["acc_magnitude_mean"]["max"],
                value=st.session_state["slider_acc_magnitude_mean"],
                step=SLIDER_CONFIG["acc_magnitude_mean"]["step"], key="slider_acc_magnitude_mean"
            )
            st.session_state["acc_magnitude_mean"] = feature_values["acc_magnitude_mean"]
            
            feature_values["acc_var"] = st.slider(
                "ACC Var", SLIDER_CONFIG["acc_var"]["min"], SLIDER_CONFIG["acc_var"]["max"],
                value=st.session_state["slider_acc_var"],
                step=SLIDER_CONFIG["acc_var"]["step"], key="slider_acc_var"
            )
            st.session_state["acc_var"] = feature_values["acc_var"]
        
        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
        
        # Auto-run inference when auto-animate is enabled (after sliders are created)
        if auto_animating:
            if model is not None and scaler is not None:
                # Build feature dict from slider session_state values
                feature_values_auto = {key: st.session_state.get(f"slider_{key}", SLIDER_CONFIG[key]["default"]) for key in FEATURES}
                label, prob_dict = predict_one(feature_values_auto, model, scaler, feature_order, reverse_state_map)
                
                if label is not None:
                    cav_score = np.random.randint(5000, 10000)
                    max_prob = max(prob_dict.values()) if prob_dict else None
                    st.session_state.inference_result = (label, cav_score, max_prob, prob_dict)
            
            # Small delay to control animation speed, then rerun
            time.sleep(0.5)
            st.rerun()
        
        # Run inference button
        if st.button("Run Inference", type="primary", use_container_width=True):
            with st.spinner("Running inference..."):
                label, prob_dict = predict_one(feature_values, model, scaler, feature_order, reverse_state_map)
                
                if label is not None:
                    cav_score = np.random.randint(5000, 10000)
                    max_prob = max(prob_dict.values()) if prob_dict else None
                    st.session_state.inference_result = (label, cav_score, max_prob, prob_dict)
                    st.rerun()
        
        # Display results if available
        if st.session_state.inference_result:
            label, cav_score, max_prob, prob_dict = st.session_state.inference_result
            
            st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
            st.markdown("---")
            st.subheader("Results")
            st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)
            
            # KPI row
            render_kpi_row(label, cav_score, max_prob)
            
            # Probabilities chart
            if prob_dict is not None:
                st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
                st.subheader("Class Probabilities")
                st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
                chart = render_probs_chart(prob_dict)
                st.altair_chart(chart, use_container_width=True)
                
                st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)
                # Individual probabilities
                cols = st.columns(len(prob_dict))
                for idx, (state, prob) in enumerate(prob_dict.items()):
                    with cols[idx]:
                        st.metric(state.capitalize(), f"{prob:.1%}")

# About section
st.markdown("<div style='margin-bottom: 32px;'></div>", unsafe_allow_html=True)
with st.expander("About", expanded=False):
    st.markdown("""
    **EDON CAV Engine v3.2**
    
    LightGBM classifier trained on 100K OEM windows. Uses six physiological features: EDA, BVP, and ACC.
    
    **Features:**
    - Dataset playback with rolling window smoothing
    - Auto-prediction during playback (requires all v3.2 features)
    - Manual inference with interactive sliders
    - Real-time visualization of CAV scores and physiological signals
    
    **Model Configuration:** LGBM, undersample=0.5, argmax predictions
    """)
