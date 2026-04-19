"""
HOURLY FETCH SCRIPT - Runs every hour via GitHub Actions
Fetches ONLY the most recent hour and appends to existing CSV
"""
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
import os

# Setup API client
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Karachi coordinates
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011

CSV_FILENAME = "karachi_aqi_2025.csv"

print("=" * 50)
print("HOURLY AQI FETCH - KARACHI")
print(f"Run at: {datetime.now()}")
print("=" * 50)

# ============================================
# STEP 1: Find the last timestamp in existing CSV
# ============================================
if os.path.exists(CSV_FILENAME):
    existing_df = pd.read_csv(CSV_FILENAME)
    existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
    last_timestamp = existing_df['timestamp'].max()
    print(f"\n📁 Existing file found")
    print(f"   Last data timestamp: {last_timestamp}")
    print(f"   Total rows so far: {len(existing_df)}")
    
    # Start from the hour AFTER the last timestamp
    start_date = (last_timestamp + timedelta(hours=1)).strftime("%Y-%m-%d")
    print(f"   Fetching from: {start_date}")
else:
    print(f"\n⚠️ No existing file found. Please run the historical fetch first.")
    exit(1)

# ============================================
# STEP 2: Fetch ONLY new data (last hour)
# ============================================
end_date = datetime.now().strftime("%Y-%m-%d")

print(f"\n📅 Fetching new data from {start_date} to {end_date}")

# Fetch Air Quality Data
air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
aq_params = {
    "latitude": KARACHI_LAT,
    "longitude": KARACHI_LON,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
               "ozone", "sulphur_dioxide", "us_aqi", "european_aqi"]
}

print("   → Fetching air quality data...")
aq_response = openmeteo.weather_api(air_quality_url, params=aq_params)[0]
aq_hourly = aq_response.Hourly()

aq_length = len(aq_hourly.Variables(0).ValuesAsNumpy())
aq_timestamps = pd.date_range(
    start=pd.to_datetime(aq_hourly.Time(), unit="s", utc=True),
    periods=aq_length,
    freq=pd.Timedelta(seconds=aq_hourly.Interval()),
    inclusive="left"
)

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

# Fetch Weather Data
weather_url = "https://archive-api.open-meteo.com/v1/archive"
weather_params = {
    "latitude": KARACHI_LAT,
    "longitude": KARACHI_LON,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
               "pressure_msl", "precipitation", "cloudcover"]
}

print("   → Fetching weather data...")
weather_response = openmeteo.weather_api(weather_url, params=weather_params)[0]
weather_hourly = weather_response.Hourly()

weather_length = len(weather_hourly.Variables(0).ValuesAsNumpy())
weather_timestamps = pd.date_range(
    start=pd.to_datetime(weather_hourly.Time(), unit="s", utc=True),
    periods=weather_length,
    freq=pd.Timedelta(seconds=weather_hourly.Interval()),
    inclusive="left"
)

weather_df = pd.DataFrame({
    "timestamp": weather_timestamps,
    "temperature": weather_hourly.Variables(0).ValuesAsNumpy(),
    "humidity": weather_hourly.Variables(1).ValuesAsNumpy(),
    "wind_speed": weather_hourly.Variables(2).ValuesAsNumpy(),
    "pressure": weather_hourly.Variables(3).ValuesAsNumpy(),
    "precipitation": weather_hourly.Variables(4).ValuesAsNumpy(),
    "cloudcover": weather_hourly.Variables(5).ValuesAsNumpy()
})

# Merge new data
new_data = pd.merge(aq_df, weather_df, on="timestamp", how="inner")
new_data.insert(0, "city", "Karachi")

print(f"   ✅ Fetched {len(new_data)} new rows")

# ============================================
# STEP 3: Append to existing CSV
# ============================================
if len(new_data) > 0:
    # Combine existing + new
    updated_df = pd.concat([existing_df, new_data], ignore_index=True)
    # Remove duplicates (just in case)
    updated_df = updated_df.drop_duplicates(subset=['timestamp'], keep='last')
    # Sort by timestamp
    updated_df = updated_df.sort_values('timestamp')
    
    # Save back to CSV
    updated_df.to_csv(CSV_FILENAME, index=False)
    
    print(f"\n✅ File updated!")
    print(f"   New total rows: {len(updated_df)}")
    print(f"   Date range now: {updated_df['timestamp'].min()} to {updated_df['timestamp'].max()}")
else:
    print("\n✅ No new data to append")

print("\n" + "=" * 50)
print("✅ HOURLY FETCH COMPLETE!")
print("=" * 50)