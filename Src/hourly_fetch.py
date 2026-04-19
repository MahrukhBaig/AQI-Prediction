import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta, timezone
import os
import sys

# Setup
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

KARACHI_LAT  = 24.8607
KARACHI_LON  = 67.0011
CSV_FILENAME = "karachi_aqi_2025.csv"

print("=" * 50)
print("HOURLY AQI FETCH - KARACHI")
print(f"Run at: {datetime.now(timezone.utc)}")
print("=" * 50)

# Read existing CSV
if not os.path.exists(CSV_FILENAME):
    print(f"Error: {CSV_FILENAME} not found!")
    sys.exit(1)

df = pd.read_csv(CSV_FILENAME)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
last_ts   = df['timestamp'].max()
next_hour = last_ts + timedelta(hours=1)

print(f"Last timestamp in CSV : {last_ts}")
print(f"Current UTC time      : {datetime.now(timezone.utc)}")
print(f"Next hour to fetch    : {next_hour}")

# Check if new data is even available yet
if next_hour > datetime.now(timezone.utc):
    print("\nNo new data needed — next hour is still in the future.")
    sys.exit(0)

# Date range to fetch
start_date = next_hour.strftime("%Y-%m-%d")
end_date   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
print(f"\nFetching from {start_date} to {end_date}...")

# PART 1: Air Quality
aq_response = openmeteo.weather_api(
    "https://air-quality-api.open-meteo.com/v1/air-quality",
    params={
        "latitude":   KARACHI_LAT,
        "longitude":  KARACHI_LON,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly": [
            "pm10", "pm2_5",
            "carbon_monoxide", "nitrogen_dioxide",
            "ozone", "sulphur_dioxide",
            "us_aqi", "european_aqi"
        ]
    }
)[0]

aq_hourly = aq_response.Hourly()
aq_len    = len(aq_hourly.Variables(0).ValuesAsNumpy())

aq_df = pd.DataFrame({
    "timestamp":        pd.date_range(
                            start=pd.to_datetime(aq_hourly.Time(), unit="s", utc=True),
                            periods=aq_len,
                            freq=pd.Timedelta(seconds=aq_hourly.Interval()),
                            inclusive="left"
                        ),
    "pm10":             aq_hourly.Variables(0).ValuesAsNumpy(),
    "pm2_5":            aq_hourly.Variables(1).ValuesAsNumpy(),
    "carbon_monoxide":  aq_hourly.Variables(2).ValuesAsNumpy(),
    "nitrogen_dioxide": aq_hourly.Variables(3).ValuesAsNumpy(),
    "ozone":            aq_hourly.Variables(4).ValuesAsNumpy(),
    "sulphur_dioxide":  aq_hourly.Variables(5).ValuesAsNumpy(),
    "us_aqi":           aq_hourly.Variables(6).ValuesAsNumpy(),
    "european_aqi":     aq_hourly.Variables(7).ValuesAsNumpy(),
})

print(f"   Air quality rows fetched: {len(aq_df)}")

# PART 2: Weather — use forecast API (works for recent + current data)
weather_response = openmeteo.weather_api(
    "https://api.open-meteo.com/v1/forecast",
    params={
        "latitude":      KARACHI_LAT,
        "longitude":     KARACHI_LON,
        "past_days":     2,
        "forecast_days": 1,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "pressure_msl",
            "precipitation",
            "cloud_cover"
        ]
    }
)[0]

w_hourly = weather_response.Hourly()
w_len    = len(w_hourly.Variables(0).ValuesAsNumpy())

weather_df = pd.DataFrame({
    "timestamp":     pd.date_range(
                         start=pd.to_datetime(w_hourly.Time(), unit="s", utc=True),
                         periods=w_len,
                         freq=pd.Timedelta(seconds=w_hourly.Interval()),
                         inclusive="left"
                     ),
    "temperature":   w_hourly.Variables(0).ValuesAsNumpy(),
    "humidity":      w_hourly.Variables(1).ValuesAsNumpy(),
    "wind_speed":    w_hourly.Variables(2).ValuesAsNumpy(),
    "pressure":      w_hourly.Variables(3).ValuesAsNumpy(),
    "precipitation": w_hourly.Variables(4).ValuesAsNumpy(),
    "cloudcover":    w_hourly.Variables(5).ValuesAsNumpy(),
})

print(f"   Weather rows fetched: {len(weather_df)}")

# PART 3: Merge, filter, deduplicate, save
new_data = pd.merge(aq_df, weather_df, on="timestamp", how="inner")
new_data.insert(0, "city", "Karachi")
new_data = new_data[new_data["timestamp"] > last_ts]

if new_data.empty:
    print("\nNo new rows after filtering — CSV is already up to date.")
    sys.exit(0)

updated = pd.concat([df, new_data], ignore_index=True)
updated = updated.drop_duplicates(subset=["timestamp"])
updated.sort_values("timestamp", inplace=True)
updated.to_csv(CSV_FILENAME, index=False)

print(f"\nAdded {len(new_data)} new rows. Total rows now: {len(updated)}")
