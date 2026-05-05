"""
EXPORT DATA FROM HOPSWORKS TO CSV
Reads all data from Hopsworks feature group and saves to CSV.
If Hopsworks access is unavailable, the CSV is updated directly from Open-Meteo.
"""
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
env_path = ROOT_DIR / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
if HOPSWORKS_API_KEY:
    HOPSWORKS_API_KEY = HOPSWORKS_API_KEY.strip()

CSV_PATH = ROOT_DIR / "karachi_aqi_2025.csv"
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011
OPEN_METEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"


def ensure_utc_timestamp(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
    return df


def fetch_open_meteo_data(start_date, end_date):
    params = {
        "latitude": KARACHI_LAT,
        "longitude": KARACHI_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
            "ozone", "sulphur_dioxide", "us_aqi", "european_aqi"
        ]),
        "timezone": "UTC"
    }
    response = requests.get(OPEN_METEO_AQ_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json().get("hourly", {})
    if not data or "time" not in data:
        raise RuntimeError("Open-Meteo air quality API returned no hourly data")

    timestamps = pd.to_datetime(data["time"])
    timestamps = timestamps.tz_localize('UTC')

    return pd.DataFrame({
        "timestamp": timestamps,
        "pm10": data.get("pm10", []),
        "pm2_5": data.get("pm2_5", []),
        "carbon_monoxide": data.get("carbon_monoxide", []),
        "nitrogen_dioxide": data.get("nitrogen_dioxide", []),
        "ozone": data.get("ozone", []),
        "sulphur_dioxide": data.get("sulphur_dioxide", []),
        "us_aqi": data.get("us_aqi", []),
        "european_aqi": data.get("european_aqi", [])
    })


def fetch_open_meteo_weather(start_date, end_date):
    params = {
        "latitude": KARACHI_LAT,
        "longitude": KARACHI_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
            "pressure_msl", "precipitation", "cloudcover"
        ]),
        "timezone": "UTC"
    }
    response = requests.get(OPEN_METEO_WEATHER_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json().get("hourly", {})
    if not data or "time" not in data:
        raise RuntimeError("Open-Meteo weather archive API returned no hourly data")

    timestamps = pd.to_datetime(data["time"])
    timestamps = timestamps.tz_localize('UTC')

    return pd.DataFrame({
        "timestamp": timestamps,
        "temperature": data.get("temperature_2m", []),
        "humidity": data.get("relative_humidity_2m", []),
        "wind_speed": data.get("wind_speed_10m", []),
        "pressure": data.get("pressure_msl", []),
        "precipitation": data.get("precipitation", []),
        "cloudcover": data.get("cloudcover", [])
    })


def merge_api_data(aq_df, weather_df):
    merged = pd.merge(aq_df, weather_df, on="timestamp", how="inner")
    merged.insert(0, "city", "Karachi")
    return merged


def update_csv_from_api(existing_df=None):
    if existing_df is not None and not existing_df.empty:
        existing_df = ensure_utc_timestamp(existing_df)
        last_ts = existing_df['timestamp'].max()
        start_date = (last_ts + timedelta(hours=1)).date()
    else:
        start_date = (datetime.now(timezone.utc) - timedelta(days=7)).date()
        last_ts = None

    end_date = datetime.now(timezone.utc).date()
    if start_date > end_date:
        print("✅ CSV already contains the latest hourly data.")
        return existing_df

    print(f"Fetching API data from {start_date} to {end_date}...")
    aq_df = fetch_open_meteo_data(str(start_date), str(end_date))
    weather_df = fetch_open_meteo_weather(str(start_date), str(end_date))
    api_df = merge_api_data(aq_df, weather_df)

    if last_ts is not None:
        api_df = api_df[api_df['timestamp'] > last_ts]

    if api_df.empty:
        print("✅ No new API rows to add.")
        combined_df = existing_df
    else:
        combined_df = pd.concat([existing_df, api_df], ignore_index=True) if existing_df is not None else api_df
        combined_df = combined_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        print(f"✅ Added {len(api_df)} new rows to CSV.")

    if combined_df is not None:
        combined_df.to_csv(CSV_PATH, index=False)
        print(f"✅ CSV updated: {CSV_PATH} ({len(combined_df)} rows)")
    return combined_df


def export_from_hopsworks():
    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY is not set")

    import hopsworks

    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        host="eu-west.cloud.hopsworks.ai"
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group("karachi_aqi_features", version=1)
    df = fg.read()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df.to_csv(CSV_PATH, index=False)
    print(f"✅ Exported {len(df)} rows to {CSV_PATH}")
    return df


def load_csv():
    if not CSV_PATH.exists():
        return None
    df = pd.read_csv(CSV_PATH, parse_dates=['timestamp'])
    return ensure_utc_timestamp(df)


if __name__ == "__main__":
    print("=" * 50)
    print("EXPORTING DATA FROM HOPSWORKS TO CSV")
    print("=" * 50)

    try:
        export_from_hopsworks()
    except Exception as e:
        print(f"⚠️ Hopsworks export failed: {e}")
        print("➡️ Falling back to updating CSV from the Open-Meteo API.")
        existing_df = load_csv()
        update_csv_from_api(existing_df)
