import os
import csv
import time
import json
from collections import deque
from datetime import datetime, timezone
from urllib import request, error
import csv as _csv
import pathlib as _pl
import pathlib as _pl2
import sys

# Add src to path for API clients
# Script is in project root, so src/ is directly accessible
project_root = _pl.Path(__file__).parent
sys.path.insert(0, str(project_root))
from src.api_clients import get_weather_data, get_air_quality, get_circadian_data

# Windows beep feedback (optional)
try:
    import winsound as _ws
except Exception:
    _ws = None

# Configuration
URL = "http://localhost:8000/cav"
CSV_PATH = os.path.join(os.path.dirname(__file__), "sensors", "real_wesad.csv")
WINDOW = 240       # samples per window (per the schema)
POST_EVERY = 8     # send once per N new samples (reduce server load)
SLEEP_BETWEEN = 0  # seconds between row reads; set >0 to slow down

# Location configuration (for environmental APIs)
# Can be set via environment variables: LOCATION_LAT, LOCATION_LON, LOCATION_TIMEZONE
LOCATION_LAT = float(os.getenv("LOCATION_LAT", "40.7128"))  # Default: New York City
LOCATION_LON = float(os.getenv("LOCATION_LON", "-74.0060"))
LOCATION_TIMEZONE = os.getenv("LOCATION_TIMEZONE", "America/New_York")

# Cache environmental data to avoid hitting API rate limits
_env_cache = {"data": None, "timestamp": 0}
ENV_CACHE_TTL = 300  # Cache for 5 minutes (300 seconds)

# Log paths
LOG_PATH = _pl.Path(__file__).with_name("cav_responses.csv")
EVENT_LOG = _pl2.Path(__file__).with_name("cav_events.csv")

# Module-level state tracking
_last_state = None


def get_environmental_data():
    """
    Fetch environmental data from APIs (weather, air quality, circadian).
    Uses caching to avoid hitting API rate limits.
    
    Returns:
        Dictionary with temp_c, humidity, aqi, local_hour
    """
    global _env_cache
    
    # Check cache
    current_time = time.time()
    if _env_cache["data"] and (current_time - _env_cache["timestamp"]) < ENV_CACHE_TTL:
        return _env_cache["data"]
    
    # Fetch fresh data
    try:
        # Get weather data (temp, humidity)
        weather = get_weather_data(LOCATION_LAT, LOCATION_LON)
        
        # Get air quality data (AQI)
        air_quality = get_air_quality(LOCATION_LAT, LOCATION_LON)
        
        # Get circadian data (hour)
        circadian = get_circadian_data(LOCATION_LAT, LOCATION_LON, LOCATION_TIMEZONE)
        
        env_data = {
            "temp_c": weather.get("temp_c", 22.0),
            "humidity": weather.get("humidity", 50.0),
            "aqi": air_quality.get("aqi", 50),
            "local_hour": circadian.get("hour", datetime.now().hour)
        }
        
        # Update cache
        _env_cache["data"] = env_data
        _env_cache["timestamp"] = current_time
        
        return env_data
    except Exception as e:
        print(f"Warning: Could not fetch environmental data: {e}. Using defaults.")
        # Return defaults and cache them briefly
        env_data = {
            "temp_c": 22.0,
            "humidity": 50.0,
            "aqi": 50,
            "local_hour": datetime.now().hour
        }
        _env_cache["data"] = env_data
        _env_cache["timestamp"] = current_time
        return env_data


def _beep(pattern="ok"):
    """Play beep pattern for audio feedback."""
    if _ws is None:
        return
    try:
        if pattern == "ok":
            _ws.Beep(880, 120)
            _ws.Beep(1320, 120)
        elif pattern == "calm":
            _ws.Beep(660, 200)
        elif pattern == "alert":
            _ws.Beep(440, 250)
            _ws.Beep(440, 250)
            _ws.Beep(440, 300)
        else:
            _ws.Beep(523, 120)
    except Exception:
        pass


def log_event(ts_iso, state, cav):
    """Log state change events to CSV."""
    first = not EVENT_LOG.exists()
    with EVENT_LOG.open("a", newline="") as f:
        w = _csv.writer(f)
        if first:
            w.writerow(["timestamp", "state", "cav"])
        w.writerow([ts_iso, state, cav])


def log_response(ts_iso: str, code, text):
    """Log API responses to CSV."""
    first = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="") as f:
        w = _csv.writer(f)
        if first:
            w.writerow(["timestamp", "status", "response"])
        w.writerow([ts_iso, code, (text or "")[:500]])


def send(payload: dict):
    """Send payload to CAV API and return status code and response text."""
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=5) as resp:
            return resp.getcode(), resp.read().decode("utf-8")
    except error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        return e.code, body
    except Exception as e:
        return None, str(e)


def main():
    """Main streaming loop."""
    global _last_state
    
    if not os.path.exists(CSV_PATH):
        print(f"? CSV not found: {CSV_PATH}")
        return
    
    # Sliding buffers
    buf = {
        "EDA": deque(maxlen=WINDOW),
        "TEMP": deque(maxlen=WINDOW),
        "BVP": deque(maxlen=WINDOW),
        "ACC_x": deque(maxlen=WINDOW),
        "ACC_y": deque(maxlen=WINDOW),
        "ACC_z": deque(maxlen=WINDOW),
    }
    
    sent = 0
    step = 0
    print(f"?? Streaming from {CSV_PATH}")
    print(f"Location: {LOCATION_LAT}, {LOCATION_LON} ({LOCATION_TIMEZONE})")
    
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        required = ["EDA", "TEMP", "BVP", "ACC_x", "ACC_y", "ACC_z"]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            print(f"? Missing columns: {missing}")
            return
        
        for row in reader:
            try:
                buf["EDA"].append(float(row["EDA"]))
                buf["TEMP"].append(float(row["TEMP"]))
                buf["BVP"].append(float(row["BVP"]))
                buf["ACC_x"].append(float(row["ACC_x"]))
                buf["ACC_y"].append(float(row["ACC_y"]))
                buf["ACC_z"].append(float(row["ACC_z"]))
            except Exception:
                # skip malformed rows
                continue
            
            step += 1
            
            # Only start posting once we have a full window
            if len(buf["EDA"]) < WINDOW:
                continue
            
            if step % POST_EVERY != 0:
                if SLEEP_BETWEEN:
                    time.sleep(SLEEP_BETWEEN)
                continue
            
            # Get environmental data from APIs
            env_data = get_environmental_data()
            
            payload = {
                # arrays of 240 samples each
                "EDA": list(buf["EDA"]),
                "TEMP": list(buf["TEMP"]),
                "BVP": list(buf["BVP"]),
                "ACC_x": list(buf["ACC_x"]),
                "ACC_y": list(buf["ACC_y"]),
                "ACC_z": list(buf["ACC_z"]),
                # required env fields (from APIs)
                "temp_c": env_data["temp_c"],
                "humidity": env_data["humidity"],
                "aqi": env_data["aqi"],
                # optional but nice-to-send
                "local_hour": env_data["local_hour"],
            }
            
            ts_iso = datetime.now(timezone.utc).isoformat()
            code, text = send(payload)
            log_response(ts_iso, code, text)
            
            # Pretty summary of server response
            try:
                obj = json.loads(text) if text else {}
            except Exception:
                obj = {}
            
            state = obj.get("state") or obj.get("decision") or obj.get("mode")
            cav = obj.get("cav_raw") or obj.get("cav_smooth") or obj.get("cav") or obj.get("score") or obj.get("cav_score")
            parts = obj.get("parts") or obj.get("components")
            
            if code == 200:
                print(f"POST /cav -> 200  window={WINDOW}  hour={payload['local_hour']}  sent={sent+1}")
                summary = []
                if state is not None:
                    summary.append(f"state={state}")
                if cav is not None:
                    summary.append(f"cav={cav}")
                if isinstance(parts, dict):
                    top = sorted(parts.items(), key=lambda kv: kv[1], reverse=True)[:2]
                    summary.append("parts=" + ",".join(f"{k}:{v:.2f}" for k, v in top))
                if summary:
                    print("  " + " | ".join(summary))
                
                # --- state-change actions ---
                if state is not None and state != _last_state:
                    ts_iso2 = datetime.now(timezone.utc).isoformat()
                    log_event(ts_iso2, state, cav if cav is not None else "")
                    if str(state).lower().startswith(("rest", "calm", "recharge")):
                        _beep("calm")
                    elif str(state).lower().startswith(("alert", "focus", "active")):
                        _beep("alert")
                    else:
                        _beep("ok")
                    _last_state = state
            else:
                print(f"POST /cav -> {code}  (detail below)")
                print((text or "")[:400])
            
            sent += 1
            if SLEEP_BETWEEN:
                time.sleep(SLEEP_BETWEEN)
    
    print(f"? Done. Windows sent: {sent}")


if __name__ == "__main__":
    main()
