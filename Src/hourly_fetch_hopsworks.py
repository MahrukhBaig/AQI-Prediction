"""
HOURLY FETCH SCRIPT - Direct to Hopsworks
"""

# ============================================
# IMPORTS
# ============================================
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta, timezone
import hopsworks
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ============================================
# ENV SETUP
# ============================================
ROOT_DIR = Path(__file__).resolve().parent.parent
env_path = ROOT_DIR / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# Remove any hidden spaces/newlines
if HOPSWORKS_API_KEY:
    HOPSWORKS_API_KEY = HOPSWORKS_API_KEY.strip()

if not HOPSWORKS_API_KEY:
    print("❌ API key missing")
    sys.exit(1)

# ============================================
# SETUP OPEN-METEO
# ============================================
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011

print("=" * 60)
print("HOURLY AQI FETCH - DIRECT TO HOPSWORKS")
print(f"Run at: {datetime.now(timezone.utc)}")
print("=" * 60)

# ============================================
# CONNECT TO HOPSWORKS
# ============================================
print("\n1. Connecting to Hopsworks...")

try:
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        host="eu-west.cloud.hopsworks.ai"
    )
    fs = project.get_feature_store()
    print(f"   ✅ Connected to project: {project.name}")
except Exception as e:
    print(f"   ❌ Failed to connect: {e}")
    sys.exit(1)

# ============================================
# FEATURE GROUP
# ============================================
try:
    fg = fs.get_feature_group("karachi_aqi_features", version=1)
    print("   ✅ Feature group found")
except Exception as e:
    print(f"   ❌ Feature group error: {e}")
    sys.exit(1)

# ============================================
# LAST TIMESTAMP
# ============================================
df_existing = fg.read()
df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'])
last_ts = df_existing['timestamp'].max()

print(f"   📅 Last timestamp: {last_ts}")

# ============================================
# CHECK NEW DATA
# ============================================
next_hour = last_ts + timedelta(hours=1)
now_utc = datetime.now(timezone.utc)

print(f"   ⏰ Next hour: {next_hour}")
print(f"   🕐 Current: {now_utc}")

if next_hour > now_utc:
    print("\n✅ No new data needed — next hour is in the future.")
    sys.exit(0)

# ============================================
# FETCH DATA
# ============================================
start_date = next_hour.strftime("%Y-%m-%d")
end_date = now_utc.strftime("%Y-%m-%d")

print(f"\n📅 Fetching {start_date} → {end_date}")

# AQ DATA
aq_response = openmeteo.weather_api(
    "https://air-quality-api.open-meteo.com/v1/air-quality",
    params={
        "latitude": KARACHI_LAT,
        "longitude": KARACHI_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "pm10", "pm2_5",
            "carbon_monoxide", "nitrogen_dioxide",
            "ozone", "sulphur_dioxide",
            "us_aqi", "european_aqi"
        ]
    }
)[0]

aq_hourly = aq_response.Hourly()
aq_len = len(aq_hourly.Variables(0).ValuesAsNumpy())

if aq_len == 0:
    print("⚠️ No AQ data")
    sys.exit(0)

aq_timestamps = pd.date_range(
    start=pd.to_datetime(aq_hourly.Time(), unit="s", utc=True),
    periods=aq_len,
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
    "european_aqi": aq_hourly.Variables(7).ValuesAsNumpy(),
})

# WEATHER DATA
weather_response = openmeteo.weather_api(
    "https://archive-api.open-meteo.com/v1/archive",
    params={
        "latitude": KARACHI_LAT,
        "longitude": KARACHI_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "pressure_msl",
            "precipitation",
            "cloudcover"
        ]
    }
)[0]

w_hourly = weather_response.Hourly()
w_len = len(w_hourly.Variables(0).ValuesAsNumpy())

weather_timestamps = pd.date_range(
    start=pd.to_datetime(w_hourly.Time(), unit="s", utc=True),
    periods=w_len,
    freq=pd.Timedelta(seconds=w_hourly.Interval()),
    inclusive="left"
)

weather_df = pd.DataFrame({
    "timestamp": weather_timestamps,
    "temperature": w_hourly.Variables(0).ValuesAsNumpy(),
    "humidity": w_hourly.Variables(1).ValuesAsNumpy(),
    "wind_speed": w_hourly.Variables(2).ValuesAsNumpy(),
    "pressure": w_hourly.Variables(3).ValuesAsNumpy(),
    "precipitation": w_hourly.Variables(4).ValuesAsNumpy(),
    "cloudcover": w_hourly.Variables(5).ValuesAsNumpy(),
})

# ============================================
# MERGE
# ============================================
new_data = pd.merge(aq_df, weather_df, on="timestamp", how="inner")
new_data.insert(0, "city", "Karachi")

new_data = new_data[new_data["timestamp"] > last_ts]

if new_data.empty:
    print("⚠️ No new rows")
    sys.exit(0)

print(f"✅ New rows: {len(new_data)}")

# ============================================
# CONVERT FLOAT TO DOUBLE (float64)
# ============================================
numeric_cols = new_data.select_dtypes(include=['float32', 'float64']).columns

for col in numeric_cols:
    new_data[col] = new_data[col].astype('float64')

# ============================================
# INSERT INTO HOPSWORKS
# ============================================
print("\n📤 Inserting into Hopsworks...")

try:
    fg.insert(new_data)
    print("   ✅ Data inserted successfully!")
except Exception as e:
    print(f"   ❌ Insert error: {e}")
    sys.exit(1)

# ============================================
# DONE
# ============================================
print("\n" + "=" * 60)
print("✅ PIPELINE SUCCESS")
print(f"   Inserted rows: {len(new_data)}")
print("=" * 60)