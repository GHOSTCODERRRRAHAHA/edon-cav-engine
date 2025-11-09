# Environmental API Integration

The CAV system now fetches real environmental data from external APIs to enhance context-aware scoring.

## APIs Used

1. **OpenWeatherMap** - Temperature and humidity
2. **AirNow (EPA)** - Air Quality Index (AQI)
3. **WorldTimeAPI** - Local time and circadian data (free, no key required)

## Configuration

### Location Settings

Set your location via environment variables or modify the defaults in the scripts:

```powershell
# Windows PowerShell
$env:LOCATION_LAT = "40.7128"      # Your latitude
$env:LOCATION_LON = "-74.0060"     # Your longitude
$env:LOCATION_TIMEZONE = "America/New_York"  # Your timezone
```

```bash
# Linux/macOS
export LOCATION_LAT="40.7128"
export LOCATION_LON="-74.0060"
export LOCATION_TIMEZONE="America/New_York"
```

### API Keys

Create a `.env` file in the project root:

```env
# OpenWeatherMap API Key
OPENWEATHER_API_KEY=your_openweather_api_key_here

# AirNow API Key (EPA)
AIRNOW_API_KEY=your_airnow_api_key_here

# Default timezone (optional)
DEFAULT_TIMEZONE=America/New_York
```

**Note:** The API clients will use default values if keys are not provided, but real data is recommended for accurate CAV scores.

## Caching

Environmental data is cached for **5 minutes** (300 seconds) to:
- Avoid hitting API rate limits
- Reduce latency
- Save API quota

The cache is automatically refreshed when it expires.

## Default Location

If no location is specified, the system defaults to:
- **Latitude:** 40.7128 (New York City)
- **Longitude:** -74.0060
- **Timezone:** America/New_York

## Testing

To test the API integration:

```powershell
# Test with default location (NYC)
python data/raw/wesad/poll_cav_and_control.py

# Test with custom location
$env:LOCATION_LAT = "37.7749"  # San Francisco
$env:LOCATION_LON = "-122.4194"
python data/raw/wesad/poll_cav_and_control.py
```

The scripts will print the location being used at startup:
```
Location: 40.7128, -74.0060 (America/New_York)
```

## Fallback Behavior

If API calls fail:
- Weather data defaults to: temp=22°C, humidity=50%
- Air quality defaults to: AQI=50 (moderate)
- Circadian data falls back to local system time

Warnings will be printed if API calls fail, but the system will continue operating with defaults.

