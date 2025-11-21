# API Keys Status

## API Keys Required (Optional - Have Defaults)

### 1. EDON_API_TOKEN
- **Purpose**: Authentication for `/oem/*` endpoints
- **Location**: Environment variable `EDON_API_TOKEN`
- **Status**: ❌ **NOT SET** (authentication disabled by default)
- **How to set**:
  ```powershell
  $env:EDON_AUTH_ENABLED = "true"
  $env:EDON_API_TOKEN = "your-secret-token-here"
  ```
- **Note**: Authentication is **disabled by default** (development mode)

### 2. OPENWEATHER_API_KEY
- **Purpose**: Fetch real weather data (temperature, humidity)
- **Location**: Environment variable `OPENWEATHER_API_KEY`
- **Status**: ❌ **NOT SET** (uses defaults: temp=22°C, humidity=50%)
- **How to get**: Sign up at https://openweathermap.org/api (free tier available)
- **How to set**:
  ```powershell
  $env:OPENWEATHER_API_KEY = "your-api-key-here"
  ```
- **Fallback**: Uses default values if not set

### 3. AIRNOW_API_KEY
- **Purpose**: Fetch real air quality data (AQI)
- **Location**: Environment variable `AIRNOW_API_KEY`
- **Status**: ❌ **NOT SET** (uses defaults: AQI=50)
- **How to get**: Sign up at https://www.airnow.gov/developers/ (free, EPA)
- **How to set**:
  ```powershell
  $env:AIRNOW_API_KEY = "your-api-key-here"
  ```
- **Fallback**: Uses default values if not set

## Current Status

**No API keys are currently configured.** The system will work with default values:
- Weather: temp=22°C, humidity=50%
- Air Quality: AQI=50 (moderate)
- Authentication: Disabled (all endpoints open)

## Setting Up API Keys

### Option 1: Environment Variables (Temporary)
```powershell
# Set for current session
$env:OPENWEATHER_API_KEY = "your-key"
$env:AIRNOW_API_KEY = "your-key"
$env:EDON_API_TOKEN = "your-token"
$env:EDON_AUTH_ENABLED = "true"
```

### Option 2: Create .env File (Recommended)
Create `.env` file in project root:
```env
# OpenWeatherMap API Key
OPENWEATHER_API_KEY=your_openweather_api_key_here

# AirNow API Key (EPA)
AIRNOW_API_KEY=your_airnow_api_key_here

# EDON API Token (for /oem/* authentication)
EDON_API_TOKEN=your-secret-token-here
EDON_AUTH_ENABLED=false
```

**Note**: The code currently reads from `os.getenv()`, so you'd need to load the .env file manually or use `python-dotenv` package.

## Files That Reference API Keys

- `app/middleware/auth.py` - EDON_API_TOKEN
- `src/api_clients.py` - OPENWEATHER_API_KEY, AIRNOW_API_KEY
- `edon-cav-engine/src/api_clients.py` - OPENWEATHER_API_KEY, AIRNOW_API_KEY

## Summary

✅ **System works without API keys** (uses defaults)
⚠️ **No API keys found** in environment or files
📝 **API keys are optional** - system has fallback defaults
🔐 **Authentication is disabled** by default (development mode)

