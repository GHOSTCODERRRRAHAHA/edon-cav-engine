# EDON OEM Evaluation Dashboard

## Overview

The EDON Evaluation Dashboard provides real-time visualization of CAV and adaptive memory data. It enables OEM partners to evaluate EDON's intelligence in real-time through interactive charts and statistics.

## Access

**URL**: http://localhost:8000/dashboard

## Running the Dashboard

1. **Start the FastAPI server:**
   ```bash
   uvicorn app.main:app --reload
   ```

2. **Open dashboard in browser:**
   ```
   http://localhost:8000/dashboard
   ```

## Dashboard Features

### Tabs

#### 1. Live CAV
- **CAV Over Time**: Line chart showing CAV scores over time
- **State Frequency**: Bar chart showing distribution of states
- **Adaptive Sensitivity Gauge**: Real-time sensitivity indicator

#### 2. Adaptive Memory
- **Hourly CAV Baseline Heatmap**: 24-hour rolling window visualization
- **Overall Statistics**: Total records, CAV mean, CAV std
- **Clear Memory Button**: Reset memory for testing

#### 3. Environment Context
- Environment data visualization (coming soon)

#### 4. System Status
- API health status
- Request count
- Average latency
- Uptime

## Data Sources

The dashboard fetches data from:
- **Memory Engine**: `/memory/summary` endpoint
- **Recent CAV Data**: In-memory cache (last 100 records)
- **System Status**: `/health` and `/telemetry` endpoints

## Auto-Refresh

The dashboard automatically refreshes every **5 seconds** to show the latest data.

## Charts

### CAV Over Time
- Shows smoothed CAV scores over time
- Range: 0-10000
- Updates as new data arrives

### State Frequency
- Bar chart showing count of each state
- States: `overload`, `balanced`, `focus`, `restorative`
- Updates based on recent data

### Hourly CAV Baseline Heatmap
- Shows hourly CAV baselines from memory engine
- 24-hour rolling window
- Color-coded by CAV value

### Adaptive Sensitivity Gauge
- Shows current adaptive sensitivity
- Range: 0-1.5
- Threshold at 1.25 (increased sensitivity)

## How OEMs Interpret Outputs

### CAV Scores
- **0-3000**: Overload state (high stress, poor conditions)
- **3000-7000**: Balanced state (normal operating)
- **7000-9000**: Focus state (optimal performance)
- **9000-10000**: Restorative state (recovery/rest)

### Adaptive Adjustments
- **Z-score**: Deviation from baseline
  - `|z| < 1.0`: Normal variation
  - `1.0 ≤ |z| < 1.5`: Moderate deviation
  - `|z| ≥ 1.5`: Significant deviation
- **Sensitivity**: Response speed
  - `1.0`: Normal
  - `>1.0`: Increased (faster state changes)
- **Env Weight Adj**: Environment influence
  - `1.0`: Normal
  - `<1.0`: Reduced (when AQI is consistently poor)

### Memory Statistics
- **Total Records**: Number of CAV responses in last 24h
- **CAV Mean**: Average CAV score
- **CAV Std**: Standard deviation (variability)
- **State Distribution**: Percentage of each state

## Technical Details

### Technology
- **Backend**: FastAPI
- **Frontend**: Plotly Dash
- **Charts**: Plotly.js
- **Styling**: Dark theme, Inter font

### Performance
- **Refresh Rate**: 5 seconds
- **Data Cache**: Last 100 records in memory
- **API Calls**: Cached to reduce load

### Browser Compatibility
- Chrome/Edge (recommended)
- Firefox
- Safari

## Troubleshooting

### Dashboard Not Loading
1. Check if server is running: `curl http://localhost:8000/health`
2. Check browser console for errors
3. Verify dashboard route: `http://localhost:8000/dashboard`

### No Data Showing
1. Make some CAV requests to populate data
2. Check memory summary: `curl http://localhost:8000/memory/summary`
3. Clear memory and start fresh: `POST /memory/clear`

### Charts Not Updating
1. Check auto-refresh interval (5 seconds)
2. Verify API endpoints are responding
3. Check browser console for errors

## Future Enhancements

- Real-time WebSocket updates
- Historical data export
- Customizable time ranges
- Multi-user support
- Advanced filtering

