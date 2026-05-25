"""
FEATURE ENGINEERING FOR AQI PREDICTION
Creates 43 features and registers Feature View
"""
import hopsworks
import pandas as pd
import numpy as np
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# ============================================
# ENV SETUP
# ============================================
ROOT_DIR = Path(__file__).resolve().parent.parent
env_path = ROOT_DIR / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"✅ Loaded .env from: {env_path}")
else:
    print("⚠️ .env file not found")

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if HOPSWORKS_API_KEY:
    HOPSWORKS_API_KEY = HOPSWORKS_API_KEY.strip()

if not HOPSWORKS_API_KEY:
    print("❌ API key missing")
    sys.exit(1)

print(f"🔐 Using API Key: {HOPSWORKS_API_KEY[:5]}*****")

print("=" * 70)
print("FEATURE ENGINEERING - KARACHI AQI")
print("=" * 70)

# ============================================
# STEP 1: Connect to Hopsworks
# ============================================
print("\n📡 1. Connecting to Hopsworks...")

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
# STEP 2: Get Raw Feature Group
# ============================================
try:
    raw_fg = fs.get_feature_group("karachi_aqi_features", version=1)
    print("   ✅ Raw feature group found")
except Exception as e:
    print(f"   ❌ Feature group error: {e}")
    sys.exit(1)

# ============================================
# STEP 3: Load raw data
# ============================================
print("\n📂 2. Loading raw AQI data from Hopsworks...")

try:
    df = raw_fg.read()
    print(f"   ✅ Loaded {len(df)} rows")
    print(f"   📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
except Exception as e:
    print(f"   ❌ Failed to load data: {e}")
    sys.exit(1)

# Sort by timestamp
df = df.sort_values('timestamp')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# ============================================
# STEP 4: Create TIME-BASED features
# ============================================
print("\n⏰ 3. Creating time-based features...")

df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['day_of_year'] = df['timestamp'].dt.dayofyear
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                      (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
df['season'] = df['month'].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 
                                 6:2, 7:2, 8:2, 9:3, 10:3, 11:3}).astype(int)

print(f"   ✅ Added: hour, day_of_week, month, day_of_year, is_weekend, is_rush_hour, season")

# ============================================
# STEP 5: Create LAG features
# ============================================
print("\n🕐 4. Creating lag features...")

df['aqi_lag_1h'] = df['us_aqi'].shift(1)
df['aqi_lag_3h'] = df['us_aqi'].shift(3)
df['aqi_lag_6h'] = df['us_aqi'].shift(6)
df['aqi_lag_12h'] = df['us_aqi'].shift(12)
df['aqi_lag_24h'] = df['us_aqi'].shift(24)
df['pm25_lag_6h'] = df['pm2_5'].shift(6)
df['pm25_lag_24h'] = df['pm2_5'].shift(24)

print(f"   ✅ Added: aqi_lag_1h,3h,6h,12h,24h and pm25_lag_6h,24h")

# ============================================
# STEP 6: Create ROLLING statistics
# ============================================
print("\n📊 5. Creating rolling statistics...")

df['aqi_rolling_mean_3h'] = df['us_aqi'].rolling(window=3).mean()
df['aqi_rolling_mean_6h'] = df['us_aqi'].rolling(window=6).mean()
df['aqi_rolling_mean_12h'] = df['us_aqi'].rolling(window=12).mean()
df['aqi_rolling_mean_24h'] = df['us_aqi'].rolling(window=24).mean()
df['aqi_rolling_std_6h'] = df['us_aqi'].rolling(window=6).std()
df['aqi_rolling_std_24h'] = df['us_aqi'].rolling(window=24).std()
df['aqi_rolling_max_6h'] = df['us_aqi'].rolling(window=6).max()
df['aqi_rolling_min_6h'] = df['us_aqi'].rolling(window=6).min()

print(f"   ✅ Added: rolling means, std dev, max/min for various windows")

# ============================================
# STEP 7: Create RATIO and INTERACTION features
# ============================================
print("\n🔢 6. Creating ratio and interaction features...")

df['pm25_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 0.01)
df['temp_humidity'] = df['temperature'] * df['humidity']
df['wind_pressure'] = df['wind_speed'] * df['pressure']
df['aqi_change'] = df['us_aqi'].diff()
df['aqi_change_rate'] = df['us_aqi'].pct_change() * 100

print(f"   ✅ Added: pm25_pm10_ratio, temp_humidity, wind_pressure, aqi_change, aqi_change_rate")

# ============================================
# STEP 8: Create WEATHER impact features
# ============================================
print("\n🌤️ 7. Creating weather impact features...")

df['temp_squared'] = df['temperature'] ** 2
df['wind_category'] = pd.cut(df['wind_speed'], bins=[0, 2, 5, 10, 100], labels=[0, 1, 2, 3]).astype(int)
df['is_raining'] = (df['precipitation'] > 0.1).astype(int)
df['high_pressure'] = (df['pressure'] > 1015).astype(int)

print(f"   ✅ Added: temp_squared, wind_category, is_raining, high_pressure")

# ============================================
# STEP 9: Clean data
# ============================================
print("\n🧹 8. Cleaning data...")

before = len(df)
df = df.dropna()
after = len(df)
print(f"   📊 Dropped {before - after} rows with null values")
print(f"   📊 Final rows: {after}")

# ============================================
# STEP 10: Convert data types
# ============================================
print("\n🔄 9. Converting data types for Hopsworks...")

float32_cols = df.select_dtypes(include=['float32']).columns
for col in float32_cols:
    df[col] = df[col].astype('float64')

int_cols = df.select_dtypes(include=['int64']).columns
for col in int_cols:
    df[col] = df[col].astype('int32')

print(f"   ✅ Converted {len(float32_cols)} columns to float64")
print(f"   ✅ Converted {len(int_cols)} columns to int32")

# ============================================
# STEP 11: Save to Hopsworks
# ============================================
print("\n💾 10. Saving engineered features to Hopsworks...")

try:
    engineered_fg = fs.get_or_create_feature_group(
        name="karachi_aqi_engineered_features",
        version=1,
        description="Engineered features for AQI prediction including time features, lags, rolling statistics, and weather interactions",
        primary_key=['timestamp'],
        event_time='timestamp',
    )
    
    engineered_fg.insert(df, write_options={"start_offline_materialization": True})
    print(f"   ✅ Saved {len(df)} rows to engineered feature group")
    
except Exception as e:
    print(f"   ❌ Failed to save: {e}")
    sys.exit(1)

# ============================================
# STEP 12: Create Feature View (Production)
# ============================================
print("\n🔍 11. Creating Feature View...")

try:
    # Get the engineered feature group
    engineered_fg = fs.get_feature_group("karachi_aqi_engineered_features", version=1)
    
    # Create Feature View (saved query)
    feature_view = fs.create_feature_view(
        name="karachi_aqi_final_view",
        version=1,
        description="Complete engineered features for AQI prediction - 43 features including time, lags, rolling stats, and weather interactions",
        query=engineered_fg.select_all()
    )
    
    print(f"   ✅ Feature View created: {feature_view.name} v{feature_view.version}")
    
except Exception as e:
    print(f"   ⚠️ Could not create Feature View: {e}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("✅ FEATURE ENGINEERING COMPLETE!")
print("=" * 70)
print(f"\n📊 Summary:")
print(f"   📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"   📊 Final rows: {after}")
print(f"   📋 Final columns: {len(df.columns)}")
print(f"\n📁 Feature Group: karachi_aqi_engineered_features v1")
print(f"📁 Feature View: karachi_aqi_final_view v1")
print("\n🎯 Next step: Train ML models!")
print("=" * 70)