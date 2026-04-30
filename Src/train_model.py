"""
TRAIN AQI PREDICTION MODEL - RANDOM FOREST
Uses SAME connection method as working hourly_fetch_hopsworks.py
"""
import hopsworks
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# ============================================
# ENV SETUP (IDENTICAL TO WORKING HOURLY SCRIPT)
# ============================================
ROOT_DIR = Path(__file__).resolve().parent.parent
env_path = ROOT_DIR / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"✅ Loaded .env from: {env_path}")
else:
    print("⚠️ .env file not found")

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# CRITICAL: Remove hidden spaces/newlines (SAME AS HOURLY SCRIPT)
if HOPSWORKS_API_KEY:
    HOPSWORKS_API_KEY = HOPSWORKS_API_KEY.strip()

if not HOPSWORKS_API_KEY:
    print("❌ API key missing")
    sys.exit(1)

print(f"🔐 Using API Key: {HOPSWORKS_API_KEY[:5]}*****")

print("=" * 70)
print("🌲 RANDOM FOREST - AQI PREDICTION MODEL")
print("=" * 70)

# ============================================
# STEP 1: Connect to Hopsworks (EXACT SAME METHOD)
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
# STEP 2: Load Engineered Features
# ============================================
print("\n📂 2. Loading engineered features...")

try:
    # First try to get engineered features
    fg = fs.get_feature_group("karachi_aqi_engineered_features", version=1)
    df = fg.read()
    print(f"   ✅ Loaded {len(df)} rows from engineered features")
    print(f"   📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
except Exception as e:
    print(f"   ⚠️ Engineered features not found: {e}")
    print("   Trying raw features instead...")
    try:
        fg = fs.get_feature_group("karachi_aqi_features", version=1)
        df = fg.read()
        print(f"   ✅ Loaded {len(df)} rows from raw features")
    except Exception as e2:
        print(f"   ❌ Failed to load data: {e2}")
        sys.exit(1)

df['timestamp'] = pd.to_datetime(df['timestamp'])

# ============================================
# STEP 3: Prepare Features (X) and Target (y)
# ============================================
print("\n🎯 3. Preparing features and target...")

# Columns to exclude (not features)
exclude_cols = ['timestamp', 'city', 'us_aqi', 'european_aqi']

# Feature columns = all columns except excluded ones
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df['us_aqi']

print(f"   ✅ Features: {len(feature_cols)} columns")
print(f"   ✅ Target: us_aqi (Air Quality Index)")

# ============================================
# STEP 4: Split Data (Time Series Split)
# ============================================
print("\n✂️ 4. Splitting data (80% train, 20% test)...")

# Sort by timestamp FIRST (important for time series!)
df_sorted = df.sort_values('timestamp')
X_sorted = X.loc[df_sorted.index]
y_sorted = y.loc[df_sorted.index]

# Use first 80% for training, last 20% for testing
split_idx = int(len(df_sorted) * 0.8)
X_train = X_sorted[:split_idx]
X_test = X_sorted[split_idx:]
y_train = y_sorted[:split_idx]
y_test = y_sorted[split_idx:]

print(f"   📊 Training set: {len(X_train)} rows")
print(f"   📊 Test set: {len(X_test)} rows")
print(f"   📅 Training: {df_sorted['timestamp'].iloc[0]} to {df_sorted['timestamp'].iloc[split_idx-1]}")
print(f"   📅 Testing: {df_sorted['timestamp'].iloc[split_idx]} to {df_sorted['timestamp'].iloc[-1]}")

# ============================================
# STEP 5: Train Random Forest Model
# ============================================
print("\n🌲 5. Training Random Forest model...")

rf_model = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=15,          # Maximum depth of each tree
    min_samples_split=5,   # Minimum samples to split a node
    min_samples_leaf=2,    # Minimum samples at leaf node
    random_state=42,       # For reproducible results
    n_jobs=-1              # Use all CPU cores
)

rf_model.fit(X_train, y_train)
print("   ✅ Model training complete!")

# ============================================
# STEP 6: Make Predictions
# ============================================
print("\n🔮 6. Making predictions...")

y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# ============================================
# STEP 7: Evaluate Model Performance
# ============================================
print("\n📊 7. Model Performance:")

# Training metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"\n   📈 Training Set:")
print(f"      RMSE: {train_rmse:.2f} (average error in AQI points)")
print(f"      MAE: {train_mae:.2f} (mean absolute error)")
print(f"      R² Score: {train_r2:.3f} (1.0 = perfect)")

# Test metrics
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\n   📉 Test Set:")
print(f"      RMSE: {test_rmse:.2f} (average error in AQI points)")
print(f"      MAE: {test_mae:.2f} (mean absolute error)")
print(f"      R² Score: {test_r2:.3f} (1.0 = perfect)")

# ============================================
# STEP 8: Feature Importance (What affects AQI most?)
# ============================================
print("\n🔍 8. Feature Importance Analysis:")

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   📊 Top 15 most important features:")
print("   " + "-" * 55)
for i, row in importance_df.head(15).iterrows():
    bar = "█" * int(row['importance'] * 50)
    print(f"   {row['feature']:25s} {row['importance']:.4f} {bar}")

# ============================================
# STEP 9: Save Model
# ============================================
print("\n💾 9. Saving model...")

# Create models directory if not exists
model_dir = ROOT_DIR / "models"
model_dir.mkdir(exist_ok=True)

# Save model
model_path = model_dir / "random_forest_aqi_model.pkl"
joblib.dump(rf_model, model_path)
print(f"   ✅ Model saved to: {model_path}")

# ============================================
# STEP 10: Save to Hopsworks Model Registry
# ============================================
print("\n📤 10. Saving to Hopsworks Model Registry...")

try:
    mr = project.get_model_registry()
    
    version = int(datetime.now().strftime("%Y%m%d"))
    print(f"   Debug: ModelRegistry has sklearn={hasattr(mr, 'sklearn')} python={hasattr(mr, 'python')} tensorflow={hasattr(mr, 'tensorflow')}")
    model_registry_obj = mr.python.create_model(
        name="aqi_predictor_random_forest",
        version=version,
        description="Random Forest model for AQI prediction in Karachi using engineered features",
        metrics={
            "train_rmse": float(train_rmse),
            "train_mae": float(train_mae),
            "train_r2": float(train_r2),
            "test_rmse": float(test_rmse),
            "test_mae": float(test_mae),
            "test_r2": float(test_r2)
        },
        input_example=X_test[:1]
    )
    print(f"   Debug: model_registry_obj type = {type(model_registry_obj)} has save={hasattr(model_registry_obj, 'save')}")
    model_registry_obj.save(str(model_path))
    print(f"   ✅ Model saved to Hopsworks Model Registry v{version}!")
except Exception as e:
    print(f"   ⚠️ Could not save to registry: {e}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 70)
print(f"\n📊 Model Performance Summary:")
print(f"   Random Forest - AQI Predictor")
print(f"   {'─' * 40}")
print(f"   Test RMSE: {test_rmse:.2f} AQI points")
print(f"   Test MAE:  {test_mae:.2f} AQI points")
print(f"   Test R²:   {test_r2:.3f}")
print(f"\n📁 Files created:")
print(f"   👉 Model: models/random_forest_aqi_model.pkl")
