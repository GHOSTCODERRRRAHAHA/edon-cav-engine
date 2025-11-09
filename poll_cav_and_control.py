#!/usr/bin/env python3
"""
CAV Polling Client and Control Loop

Polls the CAV API every 30 seconds and maps states to LED/voice control.
"""

import time
import json
import datetime as dt
import requests
import numpy as np
from pathlib import Path
import sys

API = "http://localhost:8000/cav"
LED_API = "http://localhost:5050/led"
HOP_SEC = 30
WIN_LEN = 240  # 60s @ 4Hz
LOG_FILE = Path(__file__).parent / "cav_log.csv"
SENSORS_DIR = Path(__file__).parent.parent.parent / "sensors"

# Cross-platform serial port detection
if sys.platform == "win32":
    SERIAL_PORT = "COM3"
else:
    SERIAL_PORT = "/dev/ttyUSB0"


def led_for_state(state: str):
    """
    Map CAV state to LED control parameters.
    
    Args:
        state: State string (overload, balanced, focus, restorative)
        
    Returns:
        Dictionary with LED control parameters
    """
    if state == "overload":
        return {"color": "#B87333", "pattern": "pulse", "bpm": 6, "brightness": 30}
    if state == "balanced":
        return {"color": "#FFFFFF", "pattern": "steady", "bpm": 0, "brightness": 50}
    if state == "focus":
        return {"color": "#FFFFFF", "pattern": "pulse", "bpm": 10, "brightness": 80}
    if state == "restorative":
        return {"color": "#FFBF80", "pattern": "low", "bpm": 0, "brightness": 15}
    return {"color": "#FFFFFF", "pattern": "steady", "bpm": 0, "brightness": 40}


def get_sensor_window():
    """
    Read 60 seconds (240 samples) of real sensor data.
    
    Priority:
    1. Read from sensors/ folder (CSV files with columns: EDA, TEMP, BVP, ACC_x, ACC_y, ACC_z)
    2. Read from serial device (COM3 on Windows, /dev/ttyUSB0 on Linux/macOS)
    3. Fall back to fake data if neither available
    
    Returns:
        Dictionary with window data and environmental parameters
    """
    # Try reading from sensors/ folder
    if SENSORS_DIR.exists():
        csv_files = sorted(SENSORS_DIR.glob("*.csv"))
        if csv_files:
            try:
                import pandas as pd
                # Read most recent CSV file
                df = pd.read_csv(csv_files[-1])
                
                # Check if we have enough data
                if len(df) >= WIN_LEN:
                    # Get last WIN_LEN samples
                    window_df = df.tail(WIN_LEN)
                    
                    # Extract sensor columns
                    required_cols = ['EDA', 'TEMP', 'BVP', 'ACC_x', 'ACC_y', 'ACC_z']
                    if all(col in window_df.columns for col in required_cols):
                        return {
                            "EDA": window_df['EDA'].values.tolist(),
                            "TEMP": window_df['TEMP'].values.tolist(),
                            "BVP": window_df['BVP'].values.tolist(),
                            "ACC_x": window_df['ACC_x'].values.tolist(),
                            "ACC_y": window_df['ACC_y'].values.tolist(),
                            "ACC_z": window_df['ACC_z'].values.tolist(),
                            "temp_c": 22.0,  # TODO: Get from environmental sensor
                            "humidity": 45.0,  # TODO: Get from environmental sensor
                            "aqi": 35,  # TODO: Get from air quality API
                            "local_hour": dt.datetime.now().hour
                        }
            except Exception as e:
                print(f"Warning: Could not read from sensors folder: {e}")
    
    # Try reading from serial device
    try:
        import serial
        ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)
        
        # Read data (assuming CSV format over serial)
        samples = []
        start_time = time.time()
        
        while len(samples) < WIN_LEN and (time.time() - start_time) < 65:  # 60s + buffer
            line = ser.readline().decode('utf-8').strip()
            if line:
                try:
                    # Parse CSV line: EDA,TEMP,BVP,ACC_x,ACC_y,ACC_z
                    values = [float(x) for x in line.split(',')]
                    if len(values) >= 6:
                        samples.append(values)
                except ValueError:
                    continue
        
        ser.close()
        
        if len(samples) >= WIN_LEN:
            samples = samples[-WIN_LEN:]  # Take last WIN_LEN samples
            return {
                "EDA": [s[0] for s in samples],
                "TEMP": [s[1] for s in samples],
                "BVP": [s[2] for s in samples],
                "ACC_x": [s[3] for s in samples],
                "ACC_y": [s[4] for s in samples],
                "ACC_z": [s[5] for s in samples],
                "temp_c": 22.0,  # TODO: Get from environmental sensor
                "humidity": 45.0,  # TODO: Get from environmental sensor
                "aqi": 35,  # TODO: Get from air quality API
                "local_hour": dt.datetime.now().hour
            }
    except ImportError:
        # pyserial not installed
        pass
    except Exception:
        # Serial device not available
        pass
    
    # Fall back to fake data
    print("Warning: Using fake sensor data (no real sensors found)")
    return fake_window()


def fake_window():
    """
    Generate fake sensor window for testing.
    
    Returns:
        Dictionary with window data and environmental parameters
    """
    rng = np.random.default_rng()
    return {
        "EDA": rng.normal(0.0, 0.2, WIN_LEN).tolist(),
        "TEMP": rng.normal(32.0, 0.2, WIN_LEN).tolist(),
        "BVP": rng.normal(0.0, 0.5, WIN_LEN).tolist(),
        "ACC_x": rng.normal(0.0, 0.05, WIN_LEN).tolist(),
        "ACC_y": rng.normal(0.0, 0.05, WIN_LEN).tolist(),
        "ACC_z": rng.normal(1.0, 0.05, WIN_LEN).tolist(),
        "temp_c": 22.0,
        "humidity": 45.0,
        "aqi": 35,
        "local_hour": dt.datetime.now().hour
    }


def send_to_led_controller(led):
    """
    Send LED control parameters to LED microcontroller API.
    
    Args:
        led: Dictionary with LED control parameters (color, pattern, bpm, brightness)
    """
    try:
        response = requests.post(LED_API, json=led, timeout=2)
        response.raise_for_status()
        print(f"  ✓ LED updated: {led['color']} {led['pattern']} @ {led['brightness']}%")
    except requests.exceptions.ConnectionError:
        # LED controller not available - silently skip (expected in development)
        pass
    except requests.exceptions.RequestException:
        # Other errors - silently skip
        pass


def adjust_voice_style(state):
    """
    Adjust voice style based on CAV state.
    
    Args:
        state: Current CAV state (overload, balanced, focus, restorative)
    """
    voice_modes = {
        "overload": "Calm",
        "balanced": "Neutral",
        "focus": "Focus",
        "restorative": "Recharge"
    }
    
    voice_mode = voice_modes.get(state, "Neutral")
    print(f"  ✓ Voice style: {voice_mode} mode")


def init_log_file():
    """Initialize CSV log file with headers."""
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("timestamp,cav_raw,cav_smooth,state,bio,env,circadian,p_stress\n")


def log_to_csv(timestamp, cav_raw, cav_smooth, state, parts):
    """
    Append CAV data to CSV log file.
    
    Args:
        timestamp: ISO timestamp string
        cav_raw: Raw CAV score
        cav_smooth: Smoothed CAV score
        state: State string
        parts: Parts dictionary
    """
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        p = parts
        f.write(
            f'{timestamp},{cav_raw},{cav_smooth},{state},'
            f'{p.get("bio", 0.0)},{p.get("env", 0.0)},{p.get("circadian", 0.0)},{p.get("p_stress", 0.0)}\n'
        )


def main():
    """Main polling loop."""
    print("Polling CAV every 30s… Ctrl+C to stop.")
    print(f"API: {API}")
    print(f"Log file: {LOG_FILE}")
    print()
    
    # Initialize log file
    init_log_file()
    
    # Track last state for change detection
    last_state = None
    
    try:
        while True:
            # Get window data from real sensors
            payload = get_sensor_window()
            
            # Call CAV API
            try:
                r = requests.post(API, json=payload, timeout=10)
                r.raise_for_status()
                out = r.json()
            except requests.exceptions.RequestException as e:
                print(f"{dt.datetime.now().isoformat()}  ERROR: {e}")
                time.sleep(HOP_SEC)
                continue
            
            # Extract results
            state = out.get("state", "unknown")
            cav_raw = out.get("cav_raw", 0)
            cav_smooth = out.get("cav_smooth", 0)
            parts = out.get("parts", {})
            
            # Log to CSV
            timestamp = dt.datetime.now().isoformat()
            log_to_csv(timestamp, cav_raw, cav_smooth, state, parts)
            
            # Check for state change
            state_changed = (last_state is not None and state != last_state)
            
            # Print status
            status_line = (
                f"{timestamp}  "
                f"state={state:12s}  "
                f"cav={cav_raw:5d}/{cav_smooth:5d}  "
                f"parts={parts}"
            )
            
            if state_changed:
                status_line += f"  [STATE CHANGED: {last_state} → {state}]"
            
            print(status_line)
            
            # Update LED and voice on state change
            if state_changed or last_state is None:
                led = led_for_state(state)
                send_to_led_controller(led)
                adjust_voice_style(state)
            
            # Update last state
            last_state = state
            
            time.sleep(HOP_SEC)
    
    except KeyboardInterrupt:
        print("\nCAV client stopped safely.")


if __name__ == "__main__":
    main()

