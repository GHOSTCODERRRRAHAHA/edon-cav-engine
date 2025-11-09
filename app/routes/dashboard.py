"""
EDON OEM Evaluation Dashboard

Real-time visualization of CAV and adaptive memory data.
Uses Plotly Dash for interactive charts and auto-refreshes every 5 seconds.
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
from functools import lru_cache
import time

# Initialize Dash app
dash_app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://codepen.io/chriddyp/pen/bWLwgP.css",
        {
            "href": "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
            "rel": "stylesheet"
        }
    ],
    suppress_callback_exceptions=True
)

# API base URL
API_BASE_URL = "http://localhost:8000"

# In-memory cache for recent CAV data (last 100 records)
recent_cav_data: List[Dict] = []


def get_memory_summary() -> Dict:
    """Fetch memory summary from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/memory/summary", timeout=2)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {}


def get_recent_cav_data() -> pd.DataFrame:
    """Get recent CAV data from memory or cache."""
    global recent_cav_data
    
    # Try to get from memory summary
    summary = get_memory_summary()
    if summary and summary.get('total_records', 0) > 0:
        # We'll use the memory engine's data
        # For now, return cached data
        pass
    
    # Return cached data as DataFrame
    if recent_cav_data:
        df = pd.DataFrame(recent_cav_data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    return pd.DataFrame()


def update_recent_cav_data(cav_smooth: int, state: str, adaptive: Optional[Dict] = None):
    """Update recent CAV data cache."""
    global recent_cav_data
    
    record = {
        'timestamp': datetime.now(),
        'cav_smooth': cav_smooth,
        'state': state,
        'z_cav': adaptive.get('z_cav', 0.0) if adaptive else 0.0,
        'sensitivity': adaptive.get('sensitivity', 1.0) if adaptive else 1.0,
        'env_weight_adj': adaptive.get('env_weight_adj', 1.0) if adaptive else 1.0
    }
    
    recent_cav_data.append(record)
    
    # Keep only last 100 records
    if len(recent_cav_data) > 100:
        recent_cav_data = recent_cav_data[-100:]


# Dashboard layout
dash_app.layout = html.Div([
    # Header
    html.Div([
        html.H1("EDON", style={
            'color': '#ffffff',
            'fontSize': '32px',
            'fontWeight': '700',
            'margin': '0',
            'fontFamily': 'Inter, sans-serif'
        }),
        html.P("Symbiotic Intelligence", style={
            'color': '#a0a0a0',
            'fontSize': '14px',
            'margin': '0',
            'fontFamily': 'Inter, sans-serif'
        })
    ], style={
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'padding': '20px 30px',
        'borderRadius': '8px 8px 0 0',
        'marginBottom': '20px'
    }),
    
    # Tabs
    dcc.Tabs(id="tabs", value="live", children=[
        dcc.Tab(label="Live CAV", value="live"),
        dcc.Tab(label="Adaptive Memory", value="memory"),
        dcc.Tab(label="Environment Context", value="environment"),
        dcc.Tab(label="System Status", value="status"),
    ], style={
        'fontFamily': 'Inter, sans-serif',
        'fontSize': '14px'
    }),
    
    # Content
    html.Div(id="tab-content", style={
        'padding': '20px',
        'backgroundColor': '#1a1a1a',
        'minHeight': '600px',
        'borderRadius': '0 0 8px 8px'
    }),
    
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # 5 seconds
        n_intervals=0
    ),
    
    # Store for data
    dcc.Store(id='memory-data-store'),
    dcc.Store(id='cav-data-store')
], style={
    'fontFamily': 'Inter, sans-serif',
    'backgroundColor': '#0f0f0f',
    'color': '#ffffff',
    'minHeight': '100vh',
    'padding': '20px'
})


@dash_app.callback(
    Output('memory-data-store', 'data'),
    Output('cav-data-store', 'data'),
    Input('interval-component', 'n_intervals')
)
def update_data_stores(n):
    """Update data stores on interval."""
    memory_summary = get_memory_summary()
    cav_df = get_recent_cav_data()
    
    return memory_summary, cav_df.to_dict('records') if not cav_df.empty else []


@dash_app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    Input('memory-data-store', 'data'),
    Input('cav-data-store', 'data')
)
def render_tab_content(tab, memory_data, cav_data):
    """Render content based on selected tab."""
    
    if tab == "live":
        return render_live_cav_tab(cav_data, memory_data)
    elif tab == "memory":
        return render_memory_tab(memory_data)
    elif tab == "environment":
        return render_environment_tab(memory_data)
    elif tab == "status":
        return render_status_tab()
    else:
        return html.Div("Unknown tab")


def render_live_cav_tab(cav_data: List, memory_data: Dict) -> html.Div:
    """Render Live CAV tab with real-time charts."""
    
    df = pd.DataFrame(cav_data) if cav_data else pd.DataFrame()
    
    # CAV over time chart
    cav_chart = dcc.Graph(
        id='cav-time-chart',
        figure={
            'data': [
                go.Scatter(
                    x=df['timestamp'] if not df.empty and 'timestamp' in df.columns else [],
                    y=df['cav_smooth'] if not df.empty and 'cav_smooth' in df.columns else [],
                    mode='lines+markers',
                    name='CAV Smooth',
                    line=dict(color='#667eea', width=2),
                    marker=dict(size=6)
                )
            ] if not df.empty else [],
            'layout': go.Layout(
                title='CAV Over Time',
                xaxis=dict(title='Time', color='#ffffff'),
                yaxis=dict(title='CAV Score', range=[0, 10000], color='#ffffff'),
                plot_bgcolor='#1a1a1a',
                paper_bgcolor='#1a1a1a',
                font=dict(color='#ffffff', family='Inter'),
                height=300
            )
        }
    )
    
    # State frequency chart
    if not df.empty and 'state' in df.columns:
        state_counts = df['state'].value_counts()
        state_chart = dcc.Graph(
            figure={
                'data': [
                    go.Bar(
                        x=state_counts.index,
                        y=state_counts.values,
                        marker=dict(color=['#667eea', '#764ba2', '#f093fb', '#4facfe'])
                    )
                ],
                'layout': go.Layout(
                    title='State Frequency',
                    xaxis=dict(title='State', color='#ffffff'),
                    yaxis=dict(title='Count', color='#ffffff'),
                    plot_bgcolor='#1a1a1a',
                    paper_bgcolor='#1a1a1a',
                    font=dict(color='#ffffff', family='Inter'),
                    height=300
                )
            }
        )
    else:
        state_chart = html.Div("No data available", style={'color': '#a0a0a0', 'padding': '20px'})
    
    # Adaptive sensitivity gauge
    if not df.empty and 'sensitivity' in df.columns:
        current_sensitivity = df['sensitivity'].iloc[-1] if len(df) > 0 else 1.0
    else:
        current_sensitivity = 1.0
    
    sensitivity_gauge = dcc.Graph(
        figure={
            'data': [
                go.Indicator(
                    mode="gauge+number",
                    value=current_sensitivity,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Adaptive Sensitivity", 'font': {'color': '#ffffff'}},
                    gauge={
                        'axis': {'range': [None, 1.5]},
                        'bar': {'color': '#667eea'},
                        'steps': [
                            {'range': [0, 1.0], 'color': '#2a2a2a'},
                            {'range': [1.0, 1.5], 'color': '#3a3a3a'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 1.25
                        }
                    }
                )
            ],
            'layout': go.Layout(
                plot_bgcolor='#1a1a1a',
                paper_bgcolor='#1a1a1a',
                font=dict(color='#ffffff', family='Inter'),
                height=300
            )
        }
    )
    
    return html.Div([
        html.Div([
            html.Div([cav_chart], style={'width': '66%', 'display': 'inline-block', 'padding': '10px'}),
            html.Div([sensitivity_gauge], style={'width': '33%', 'display': 'inline-block', 'padding': '10px'})
        ]),
        html.Div([state_chart], style={'padding': '10px'})
    ])


def render_memory_tab(memory_data: Dict) -> html.Div:
    """Render Adaptive Memory tab with hourly heatmap."""
    
    if not memory_data or not memory_data.get('hourly_stats'):
        return html.Div("No memory data available", style={'color': '#a0a0a0', 'padding': '20px'})
    
    hourly_stats = memory_data.get('hourly_stats', {})
    
    # Prepare heatmap data
    hours = list(range(24))
    cav_means = [hourly_stats.get(h, {}).get('cav_mean', 0) for h in hours]
    
    # Create heatmap
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=[cav_means],
        x=hours,
        y=['CAV Mean'],
        colorscale='Viridis',
        showscale=True
    ))
    
    heatmap_fig.update_layout(
        title='Hourly CAV Baseline (24h Rolling Window)',
        xaxis=dict(title='Hour of Day', color='#ffffff'),
        yaxis=dict(title='', color='#ffffff'),
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        font=dict(color='#ffffff', family='Inter'),
        height=200
    )
    
    # Overall statistics
    overall_stats = memory_data.get('overall_stats', {})
    
    stats_html = html.Div([
        html.H3("Overall Statistics", style={'color': '#ffffff', 'marginTop': '20px'}),
        html.Div([
            html.Div([
                html.P("Total Records", style={'color': '#a0a0a0', 'margin': '0'}),
                html.H2(f"{memory_data.get('total_records', 0)}", style={'color': '#667eea', 'margin': '0'})
            ], style={'display': 'inline-block', 'margin': '10px', 'padding': '15px', 'backgroundColor': '#2a2a2a', 'borderRadius': '8px', 'minWidth': '150px'}),
            html.Div([
                html.P("CAV Mean", style={'color': '#a0a0a0', 'margin': '0'}),
                html.H2(f"{overall_stats.get('cav_mean', 0):.1f}", style={'color': '#667eea', 'margin': '0'})
            ], style={'display': 'inline-block', 'margin': '10px', 'padding': '15px', 'backgroundColor': '#2a2a2a', 'borderRadius': '8px', 'minWidth': '150px'}),
            html.Div([
                html.P("CAV Std", style={'color': '#a0a0a0', 'margin': '0'}),
                html.H2(f"{overall_stats.get('cav_std', 0):.1f}", style={'color': '#667eea', 'margin': '0'})
            ], style={'display': 'inline-block', 'margin': '10px', 'padding': '15px', 'backgroundColor': '#2a2a2a', 'borderRadius': '8px', 'minWidth': '150px'})
        ])
    ])
    
    # Clear memory button
    clear_button = html.Button(
        "Clear Memory",
        id='clear-memory-button',
        n_clicks=0,
        style={
            'backgroundColor': '#dc3545',
            'color': '#ffffff',
            'border': 'none',
            'padding': '10px 20px',
            'borderRadius': '5px',
            'cursor': 'pointer',
            'fontFamily': 'Inter',
            'marginTop': '20px'
        }
    )
    
    return html.Div([
        dcc.Graph(figure=heatmap_fig),
        stats_html,
        clear_button,
        html.Div(id='clear-memory-output')
    ])


def render_environment_tab(memory_data: Dict) -> html.Div:
    """Render Environment Context tab."""
    return html.Div([
        html.H3("Environment Context", style={'color': '#ffffff'}),
        html.P("Environment data visualization coming soon...", style={'color': '#a0a0a0'})
    ])


def render_status_tab() -> html.Div:
    """Render System Status tab."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        health = response.json() if response.status_code == 200 else {}
    except Exception:
        health = {}
    
    try:
        response = requests.get(f"{API_BASE_URL}/telemetry", timeout=2)
        telemetry = response.json() if response.status_code == 200 else {}
    except Exception:
        telemetry = {}
    
    return html.Div([
        html.H3("System Status", style={'color': '#ffffff'}),
        html.Div([
            html.Div([
                html.P("API Status", style={'color': '#a0a0a0', 'margin': '0'}),
                html.H3("✓ Online" if health.get('ok') else "✗ Offline", 
                       style={'color': '#28a745' if health.get('ok') else '#dc3545', 'margin': '0'})
            ], style={'display': 'inline-block', 'margin': '10px', 'padding': '15px', 'backgroundColor': '#2a2a2a', 'borderRadius': '8px', 'minWidth': '150px'}),
            html.Div([
                html.P("Requests", style={'color': '#a0a0a0', 'margin': '0'}),
                html.H3(f"{telemetry.get('request_count', 0)}", style={'color': '#667eea', 'margin': '0'})
            ], style={'display': 'inline-block', 'margin': '10px', 'padding': '15px', 'backgroundColor': '#2a2a2a', 'borderRadius': '8px', 'minWidth': '150px'}),
            html.Div([
                html.P("Avg Latency", style={'color': '#a0a0a0', 'margin': '0'}),
                html.H3(f"{telemetry.get('avg_latency_ms', 0):.1f}ms", style={'color': '#667eea', 'margin': '0'})
            ], style={'display': 'inline-block', 'margin': '10px', 'padding': '15px', 'backgroundColor': '#2a2a2a', 'borderRadius': '8px', 'minWidth': '150px'})
        ])
    ])


@dash_app.callback(
    Output('clear-memory-output', 'children'),
    Input('clear-memory-button', 'n_clicks')
)
def clear_memory(n_clicks):
    """Handle clear memory button click."""
    if n_clicks > 0:
        try:
            response = requests.post(f"{API_BASE_URL}/memory/clear", timeout=5)
            if response.status_code == 200:
                return html.Div("Memory cleared successfully", style={'color': '#28a745', 'marginTop': '10px'})
            else:
                return html.Div("Failed to clear memory", style={'color': '#dc3545', 'marginTop': '10px'})
        except Exception as e:
            return html.Div(f"Error: {str(e)}", style={'color': '#dc3545', 'marginTop': '10px'})
    return html.Div()


# Function to get Dash app for mounting
def get_dash_app():
    """Return Dash app instance for mounting in FastAPI."""
    return dash_app


# Function to update CAV data from API responses
def update_cav_from_response(response_data: Dict):
    """Update recent CAV data from API response."""
    if 'cav_smooth' in response_data and 'state' in response_data:
        adaptive = response_data.get('adaptive', {})
        update_recent_cav_data(
            cav_smooth=response_data['cav_smooth'],
            state=response_data['state'],
            adaptive=adaptive
        )

