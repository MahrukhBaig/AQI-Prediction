"""
HOURLY FETCH SCRIPT - Simplified & Foolproof
"""
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
import os
import sys

# Setup API client
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Karachi coordinates
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011

CSV_FILENAME = "karachi_aqi_2025.csv"

print("=" * 60)
print("HOURLY AQI FETCH - KARACHI")
print("=" * 60)

# ============================================
# STEP 1: Read CSV and get last timestamp
# ============================================
if not os.path.exists(CSV_FILENAME):
    print(f"❌ Error: {CSV_FILENAME} not found!")
    sys.exit(1)

existing_df = pd.read_csv(CSV_FILENAME)
existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])

# Convert to naive datetime (remove timezone if any)
if existing_df['timestamp'].dt.tz is not None:
    existing_df['timestamp'] = existing_df['timestamp'].dt.tz_localize(None)

last_timestamp = existing_df['timestamp'].max()
print(f"\n📁 Last data timestamp: {last_timestamp}")
print(f"   Total rows: {len(existing_df)}")

# ============================================
# STEP 2: Get current time (naive, no timezone)
# ============================================
now_naive = datetime.now()
print(f"   Current time: {now_naive}")

# Calculate next hour to fetch
next_hour = last_timestamp + timedelta(hours=1)
print(f"   Next hour to fetch: {next_hour}")

# ============================================
# STEP 3: CRITICAL CHECK - Is next_hour in the future?
# ============================================
if next_hour > now_naive:
    print(f"\n✅ No new data needed.")
    print(f"   Next hour ({next_hour}) is in the future.")
    print(f"   Current time ({now_naive}) is earlier.")
    print("   Exiting successfully.")
    sys.exit(0)

# ============================================
# STEP 4: If we get here, we need to fetch data
# ============================================
start_date = next_hour.strftime("%Y-%m-%d")
end_date = now_naive.strftime("%Y-%m-%d")

print(f"\n📅 Fetching from {start_date} to {end_date}")

# Safety check
if start_date > end_date:
    print(f"❌ Error: start_date > end_date")
    sys.exit(1)

# ============================================
# STEP 5: Fetch Air Quality Data
# ============================================
try:
    air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": KARACHI_LAT,
        "longitude": KARACHI_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                   "ozone", "sulphur_dioxide", "us_aqi", "european_aqi"]
    }

    print("   → Fetching air quality...")
    aq_response = openmeteo.weather_api(air_quality_url, params=aq_params)[0]
    aq_hourly = aq_response.Hourly()
    
    aq_length = len(aq_hourly.Variables(0).ValuesAsNumpy())
    
    if aq_length == 0:
        print("   No new data")
        sys.exit(0)
    
    aq_timestamps = pd.date_range(
        start=pd.to_datetime(aq_hourly.Time(), unit="s"),
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
    
    # ============================================
    # STEP 6: Fetch Weather Data
    # ============================================
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": KARACHI_LAT,
        "longitude": KARACHI_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                   "pressure_msl", "precipitation", "cloudcover"]
    }
    
    print("   → Fetching weather...")
    weather_response = openmeteo.weather_api(weather_url, params=weather_params)[0]
    weather_hourly = weather_response.Hourly()
    
    weather_length = len(weather_hourly.Variables(0).ValuesAsNumpy())
    
    weather_timestamps = pd.date_range(
        start=pd.to_datetime(weather_hourly.Time(), unit="s"),
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
    
    # ============================================
    # STEP 7: Merge and Save
    # ============================================
    new_data = pd.merge(aq_df, weather_df, on="timestamp", how="inner")
    new_data.insert(0, "city", "Karachi")
    
    # Remove duplicates
    new_data = new_data[~new_data['timestamp'].isin(existing_df['timestamp'])]
    
    if len(new_data) == 0:
        print("   No new data after deduplication")
        sys.exit(0)
    
    updated_df = pd.concat([existing_df, new_data], ignore_index=True)
    updated_df = updated_df.sort_values('timestamp')
    updated_df.to_csv(CSV_FILENAME, index=False)
    
    print(f"\n✅ SUCCESS! Added {len(new_data)} new rows")
    print(f"   Total rows now: {len(updated_df)}")
    print(f"   Date range: {updated_df['timestamp'].min()} to {updated_df['timestamp'].max()}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

print("\n✅ HOURLY FETCH COMPLETE!")