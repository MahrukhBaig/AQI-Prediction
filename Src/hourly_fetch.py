import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
import os
import sys

# Setup
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011
CSV_FILENAME = "karachi_aqi_2025.csv"

print("=" * 50)
print("HOURLY AQI FETCH - KARACHI")
print(f"Run at: {datetime.now()}")
print("=" * 50)

# Read CSV
if not os.path.exists(CSV_FILENAME):
    print(f"Error: {CSV_FILENAME} not found!")
    sys.exit(1)

df = pd.read_csv(CSV_FILENAME)
df['timestamp'] = pd.to_datetime(df['timestamp'])
last_ts = df['timestamp'].max()
print(f"Last timestamp: {last_ts}")
print(f"Current time: {datetime.now()}")

# Calculate next hour
next_hour = last_ts + timedelta(hours=1)

# Check if next hour is in the future
if next_hour > datetime.now():
    print("\n✅ No new data needed. Next hour is in the future.")
    sys.exit(0)

# If we get here, fetch data
start_date = next_hour.strftime("%Y-%m-%d")
end_date = datetime.now().strftime("%Y-%m-%d")
print(f"\nFetching from {start_date} to {end_date}")

# Fetch air quality
url = "https://air-quality-api.open-meteo.com/v1/air-quality"
params = {
    "latitude": KARACHI_LAT,
    "longitude": KARACHI_LON,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
               "ozone", "sulphur_dioxide", "us_aqi", "european_aqi"]
}

response = openmeteo.weather_api(url, params=params)[0]
hourly = response.Hourly()
timestamps = pd.date_range(
    start=pd.to_datetime(hourly.Time(), unit="s"),
    periods=len(hourly.Variables(0).ValuesAsNumpy()),
    freq=pd.Timedelta(seconds=hourly.Interval())
)

aq_df = pd.DataFrame({
    "timestamp": timestamps,
    "pm10": hourly.Variables(0).ValuesAsNumpy(),
    "pm2_5": hourly.Variables(1).ValuesAsNumpy(),
    "us_aqi": hourly.Variables(6).ValuesAsNumpy(),
})

# Fetch weather
weather_url = "https://archive-api.open-meteo.com/v1/archive"
weather_params = {
    "latitude": KARACHI_LAT,
    "longitude": KARACHI_LON,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
}

weather_response = openmeteo.weather_api(weather_url, params=weather_params)[0]
weather_hourly = weather_response.Hourly()
weather_df = pd.DataFrame({
    "timestamp": timestamps,
    "temperature": weather_hourly.Variables(0).ValuesAsNumpy(),
    "humidity": weather_hourly.Variables(1).ValuesAsNumpy(),
    "wind_speed": weather_hourly.Variables(2).ValuesAsNumpy(),
})

# Merge and save
new_data = pd.merge(aq_df, weather_df, on="timestamp")
new_data.insert(0, "city", "Karachi")
updated = pd.concat([df, new_data], ignore_index=True)
updated = updated.drop_duplicates(subset=['timestamp'])
updated.to_csv(CSV_FILENAME, index=False)

print(f"✅ Added {len(new_data)} rows. Total: {len(updated)}")
