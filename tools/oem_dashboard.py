#!/usr/bin/env python3
"""
EDON OEM Dashboard - Streamlit-based monitoring dashboard.

Shows:
- Model information (name, SHA256, PCA dims)
- Live CAV score stream
- State transitions (color-coded)
- System memory summary and telemetry stats
"""

import streamlit as st
import requests
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta

# Configuration
API_URL = "http://127.0.0.1:8000"
REFRESH_INTERVAL = 3  # seconds

# Page config
st.set_page_config(
    page_title="EDON CAV Engine - OEM Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'cav_history' not in st.session_state:
    st.session_state.cav_history = []
if 'state_history' not in st.session_state:
    st.session_state.state_history = []
if 'max_history' not in st.session_state:
    st.session_state.max_history = 100  # Keep last 100 points


def fetch_model_info() -> Optional[Dict]:
    """Fetch model information from API."""
    try:
        response = requests.get(f"{API_URL}/models/info", timeout=2.0)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error fetching model info: {e}")
    return None


def fetch_telemetry() -> Optional[Dict]:
    """Fetch telemetry statistics from API."""
    try:
        response = requests.get(f"{API_URL}/telemetry", timeout=2.0)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error fetching telemetry: {e}")
    return None


def fetch_memory_summary() -> Optional[Dict]:
    """Fetch memory summary from API."""
    try:
        response = requests.get(f"{API_URL}/memory/summary", timeout=2.0)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        # Memory endpoint might not be available
        pass
    return None


def get_state_color(state: str) -> str:
    """Get color for state."""
    colors = {
        "overload": "#FF4444",  # Red
        "balanced": "#44FF44",  # Green
        "focus": "#4444FF",     # Blue
        "restorative": "#FFAA44"  # Orange
    }
    return colors.get(state.lower(), "#888888")


def create_cav_chart(cav_history: List[Dict]) -> go.Figure:
    """Create CAV score line chart."""
    if not cav_history:
        fig = go.Figure()
        fig.add_annotation(
            text="No data yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    df = pd.DataFrame(cav_history)
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("CAV Score", "State"),
        row_heights=[0.7, 0.3]
    )
    
    # CAV score line
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['cav_smooth'],
            mode='lines+markers',
            name='CAV Smooth',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # State bars (color-coded)
    if 'state' in df.columns:
        for state in df['state'].unique():
            state_df = df[df['state'] == state]
            fig.add_trace(
                go.Scatter(
                    x=state_df['timestamp'],
                    y=[state] * len(state_df),
                    mode='markers',
                    name=state,
                    marker=dict(
                        color=get_state_color(state),
                        size=10,
                        symbol='square'
                    ),
                    showlegend=True
                ),
                row=2, col=1
            )
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="CAV Score", row=1, col=1)
    fig.update_yaxes(title_text="State", row=2, col=1)
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def main():
    """Main dashboard function."""
    st.title("📊 EDON CAV Engine - OEM Dashboard")
    st.markdown("---")
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (3s)", value=True)
    if auto_refresh:
        time.sleep(REFRESH_INTERVAL)
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Model Information Section
    st.header("🔧 Model Information")
    model_info = fetch_model_info()
    
    if model_info:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model Name", model_info.get("name", "unknown"))
        with col2:
            st.metric("Features", model_info.get("features", "N/A"))
        with col3:
            st.metric("Window Size", f"{model_info.get('window', 'N/A')} samples")
        with col4:
            st.metric("PCA Dimensions", model_info.get("pca_dim", "N/A"))
        
        st.code(f"SHA256: {model_info.get('sha256', 'unknown')[:32]}...")
    else:
        st.warning("⚠️ Could not fetch model information")
    
    st.markdown("---")
    
    # Telemetry Section
    st.header("📈 System Telemetry")
    telemetry = fetch_telemetry()
    
    if telemetry:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Requests", f"{telemetry.get('request_count', 0):,}")
        with col2:
            st.metric("Avg Latency", f"{telemetry.get('avg_latency_ms', 0):.2f} ms")
        with col3:
            uptime_hours = telemetry.get('uptime_seconds', 0) / 3600
            st.metric("Uptime", f"{uptime_hours:.2f} hours")
        with col4:
            if telemetry.get('request_count', 0) > 0:
                req_per_sec = telemetry.get('request_count', 0) / max(telemetry.get('uptime_seconds', 1), 1)
                st.metric("Req/sec", f"{req_per_sec:.2f}")
    else:
        st.warning("⚠️ Could not fetch telemetry")
    
    st.markdown("---")
    
    # Memory Summary Section
    st.header("💾 Memory Summary")
    memory = fetch_memory_summary()
    
    if memory:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", f"{memory.get('total_records', 0):,}")
            st.metric("Window Hours", memory.get('window_hours', 24))
        with col2:
            if 'overall_stats' in memory:
                stats = memory['overall_stats']
                st.metric("CAV Mean", f"{stats.get('cav_mean', 0):.2f}")
                st.metric("CAV Std", f"{stats.get('cav_std', 0):.2f}")
    else:
        st.info("ℹ️ Memory summary not available")
    
    st.markdown("---")
    
    # CAV Score Stream Section
    st.header("📊 Live CAV Score Stream")
    
    # Try to get recent CAV data (this would need to be implemented via a streaming endpoint or stored data)
    # For now, show placeholder
    if len(st.session_state.cav_history) > 0:
        fig = create_cav_chart(st.session_state.cav_history)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ No CAV data yet. Start sending requests to /cav or /oem/cav/batch to see live stream.")
    
    # State Distribution
    if st.session_state.state_history:
        st.subheader("State Distribution")
        state_counts = pd.Series(st.session_state.state_history).value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=state_counts.index,
            values=state_counts.values,
            marker=dict(colors=[get_state_color(s) for s in state_counts.index])
        )])
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption(f"API: {API_URL}")


if __name__ == "__main__":
    main()

