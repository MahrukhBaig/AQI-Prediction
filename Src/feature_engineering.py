"""
FEATURE ENGINEERING FOR AQI PREDICTION
Converts raw data into ML-ready features
"""
import hopsworks
import pandas as pd
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv
import sys

# Load API key
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not API_KEY:
    print("❌ API key not found!")
    sys.exit(1)

print("=" * 60)
print("FEATURE ENGINEERING - KARACHI AQI")
print("=" * 60)

# ============================================
# STEP 1: Connect to Hopsworks
# ============================================
print("\n1. Connecting to Hopsworks...")

try:
    project = hopsworks.login(api_key_value=API_KEY)
    fs = project.get_feature_store()
    print(f"   ✅ Connected to project: {project.name}")
except Exception as e:
    print(f"   ❌ Failed to connect: {e}")
    sys.exit(1)

# ============================================
# STEP 2: Load raw data
# ============================================
print("\n2. Loading raw AQI data from Hopsworks...")

try:
    fg = fs.get_feature_group("karachi_aqi_features", version=1)
    df = fg.read()
    print(f"   ✅ Loaded {len(df)} rows")
except Exception as e:
    print(f"   ❌ Failed to load data: {e}")
    sys.exit(1)

# Sort by timestamp
df = df.sort_values('timestamp')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# ============================================
# STEP 3: Create TIME-BASED features
# ============================================
print("\n3. Creating time-based features...")

df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
df['month'] = df['timestamp'].dt.month
df['day_of_year'] = df['timestamp'].dt.dayofyear
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Rush hour: 7-9 AM and 5-7 PM
df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                      (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)

print(f"   ✅ Added: hour, day_of_week, month, day_of_year, is_weekend, is_rush_hour")

# ============================================
# STEP 4: Create LAG features (past AQI values)
# ============================================
print("\n4. Creating lag features...")

df['aqi_lag_1h'] = df['us_aqi'].shift(1)
df['aqi_lag_6h'] = df['us_aqi'].shift(6)
df['aqi_lag_12h'] = df['us_aqi'].shift(12)
df['aqi_lag_24h'] = df['us_aqi'].shift(24)

print(f"   ✅ Added: aqi_lag_1h, aqi_lag_6h, aqi_lag_12h, aqi_lag_24h")

# ============================================
# STEP 5: Create ROLLING statistics
# ============================================
print("\n5. Creating rolling statistics...")

df['aqi_rolling_mean_6h'] = df['us_aqi'].rolling(window=6).mean()
df['aqi_rolling_std_6h'] = df['us_aqi'].rolling(window=6).std()
df['aqi_rolling_mean_24h'] = df['us_aqi'].rolling(window=24).mean()
df['aqi_rolling_std_24h'] = df['us_aqi'].rolling(window=24).std()

print(f"   ✅ Added: aqi_rolling_mean_6h, aqi_rolling_std_6h, aqi_rolling_mean_24h, aqi_rolling_std_24h")

# ============================================
# STEP 6: Create RATIO features
# ============================================
print("\n6. Creating ratio features...")

# PM2.5 to PM10 ratio (small particles vs larger particles)
df['pm25_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 0.01)

# Temperature-Humidity interaction (heat index proxy)
df['temp_humidity'] = df['temperature'] * df['humidity']

# Wind-Pressure interaction
df['wind_pressure'] = df['wind_speed'] * df['pressure']

# Change from previous hour
df['aqi_change'] = df['us_aqi'].diff()

print(f"   ✅ Added: pm25_pm10_ratio, temp_humidity, wind_pressure, aqi_change")

# ============================================
# STEP 7: Clean data (remove nulls from lag/rolling)
# ============================================
print("\n7. Cleaning data...")

before = len(df)
df = df.dropna()
after = len(df)
dropped = before - after
print(f"   Dropped {dropped} rows with null values (from lag/rolling features)")
print(f"   Final rows: {after}")

# ============================================
# STEP 8: Save to Hopsworks
# ============================================
print("\n8. Saving engineered features to Hopsworks...")

# Convert float32 to float64 for Hopsworks compatibility
for col in df.select_dtypes(include=['float32']).columns:
    df[col] = df[col].astype('float64')

try:
    engineered_fg = fs.get_or_create_feature_group(
        name="karachi_aqi_engineered_features",
        version=1,
        description="Engineered features for AQI prediction including time features, lags, rolling statistics, and ratios",
        primary_key=['timestamp'],
        event_time='timestamp',
    )
    
    engineered_fg.insert(df)
    print(f"   ✅ Saved {len(df)} rows to engineered feature group")
except Exception as e:
    print(f"   ❌ Failed to save: {e}")
    sys.exit(1)

# ============================================
# STEP 9: Verification
# ============================================
print("\n9. Verifying saved data...")

try:
    sample = engineered_fg.read(limit=5)
    print("\n   Preview of engineered features:")
    print(sample[['timestamp', 'hour', 'day_of_week', 'is_weekend', 'aqi_lag_1h', 'aqi_rolling_mean_6h']].head())
    
    # Show all columns
    print(f"\n   Total columns in feature group: {len(sample.columns)}")
    print(f"   Columns: {sample.columns.tolist()}")
    
except Exception as e:
    print(f"   ⚠️ Could not verify: {e}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 60)
print("✅ FEATURE ENGINEERING COMPLETE!")
print("=" * 60)
print(f"\n📊 Summary:")
print(f"   Original rows: {before}")
print(f"   Final rows: {after}")
print(f"   Original columns: 15")
print(f"   Final columns: {len(df.columns)}")
print(f"\n📁 New feature group:")
print(f"   Name: karachi_aqi_engineered_features")
print(f"   Version: 1")
print("\n🔜 Next step: Train ML model using these features!")
print("=" * 60)