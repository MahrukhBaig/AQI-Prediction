"""
FETCH FULL 2025 KARACHI AQI DATA AND SAVE AS EXCEL
"""
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Setup API client
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Karachi coordinates
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011

print("=" * 50)
print("FETCHING KARACHI AQI DATA - FULL YEAR 2025")
print("=" * 50)

# ============================================
# PART 1: GET AIR QUALITY DATA
# ============================================
print("\n1. Fetching air quality data...")

air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
aq_params = {
    "latitude": KARACHI_LAT,
    "longitude": KARACHI_LON,
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",  # FULL YEAR 2025
    "hourly": [
        "pm10", "pm2_5",
        "carbon_monoxide", "nitrogen_dioxide",
        "ozone", "sulphur_dioxide",
        "us_aqi", "european_aqi"
    ]
}

aq_response = openmeteo.weather_api(air_quality_url, params=aq_params)[0]
aq_hourly = aq_response.Hourly()

# Get the length of air quality data
aq_length = len(aq_hourly.Variables(0).ValuesAsNumpy())

# Create timestamps
aq_timestamps = pd.date_range(
    start=pd.to_datetime(aq_hourly.Time(), unit="s", utc=True),
    periods=aq_length,
    freq=pd.Timedelta(seconds=aq_hourly.Interval()),
    inclusive="left"
)

# Create dataframe for air quality
aq_df = pd.DataFrame({
    "timestamp": aq_timestamps,
    "pm10": aq_hourly.Variables(0).ValuesAsNumpy(),
    "pm2_5": aq_hourly.Variables(1).ValuesAsNumpy(),
    "carbon_monoxide": aq_hourly.Variables(2).ValuesAsNumpy(),
    "nitrogen_dioxide": aq_hourly.Variables(3).ValuesAsNumpy(),
    "ozone": aq_hourly.Variables(4).ValuesAsNumpy(),
    "sulphur_dioxide": aq_hourly.Variables(5).ValuesAsNumpy(),
    "us_aqi": aq_hourly.Variables(6).ValuesAsNumpy(),
    "european_aqi": aq_hourly.Variables(7).ValuesAsNumpy()
})

print(f"   ✅ Got {len(aq_df)} rows of air quality data")

# ============================================
# PART 2: GET WEATHER DATA
# ============================================
print("\n2. Fetching weather data...")

weather_url = "https://archive-api.open-meteo.com/v1/archive"
weather_params = {
    "latitude": KARACHI_LAT,
    "longitude": KARACHI_LON,
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "pressure_msl",
        "precipitation",
        "cloudcover"
    ]
}

weather_response = openmeteo.weather_api(weather_url, params=weather_params)[0]
weather_hourly = weather_response.Hourly()

# Get the length of weather data
weather_length = len(weather_hourly.Variables(0).ValuesAsNumpy())

# Create timestamps for weather
weather_timestamps = pd.date_range(
    start=pd.to_datetime(weather_hourly.Time(), unit="s", utc=True),
    periods=weather_length,
    freq=pd.Timedelta(seconds=weather_hourly.Interval()),
    inclusive="left"
)

# Create dataframe for weather
weather_df = pd.DataFrame({
    "timestamp": weather_timestamps,
    "temperature": weather_hourly.Variables(0).ValuesAsNumpy(),
    "humidity": weather_hourly.Variables(1).ValuesAsNumpy(),
    "wind_speed": weather_hourly.Variables(2).ValuesAsNumpy(),
    "pressure": weather_hourly.Variables(3).ValuesAsNumpy(),
    "precipitation": weather_hourly.Variables(4).ValuesAsNumpy(),
    "cloudcover": weather_hourly.Variables(5).ValuesAsNumpy()
})

print(f"   ✅ Got {len(weather_df)} rows of weather data")

# ============================================
# PART 3: MERGE AND SAVE AS EXCEL
# ============================================
print("\n3. Merging data...")

# Merge on timestamp
merged_df = pd.merge(aq_df, weather_df, on="timestamp", how="inner")

# Add city column
merged_df.insert(0, "city", "Karachi")

print(f"   ✅ Merged: {len(merged_df)} total rows")

# ============================================
# PART 4: SAVE AS EXCEL FILE
# ============================================
print("\n4. Saving as Excel file...")
merged_df.to_csv("karachi_aqi_2025.csv", index=False)

print("✅ Saved to karachi_aqi_2025.csv")