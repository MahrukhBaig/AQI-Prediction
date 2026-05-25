"""
TRAIN XGBOOST MODEL - AQI PREDICTION (Production Ready)
Includes Training Dataset versioning for reproducibility
"""
import hopsworks
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import joblib
import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# ============================================
# PRODUCTION HELPER FUNCTIONS
# ============================================

def create_training_dataset(feature_view, description=None):
    """
    Create a versioned training dataset for reproducibility.
    Each day creates a NEW frozen snapshot.
    Falls back to Feature View if Training Dataset creation fails.
    """
    version_date = datetime.now().strftime("%Y%m%d")
    dataset_name = f"aqi_training_data_{version_date}"
    
    try:
        # Try to create new training dataset
        print(f"   📊 Creating new training dataset: {dataset_name}")
        td = feature_view.create_training_dataset(
            name=dataset_name,
            version=1,
            description=description or f"Training data for model trained on {version_date}",
            data_format="csv",
            write_options={"wait_for_job": True}
        )
        print(f"   ✅ Created training dataset: {dataset_name}")
        return td, version_date
    except AttributeError as e:
        print(f"   ⚠️ Could not create Training Dataset, falling back to Feature View: {e}")
        # Fall back to using Feature View directly
        df = feature_view.get_batch_data()
        print(f"   ✅ Loaded {len(df)} rows from Feature View")
        return df, version_date  # Return df instead of td, but we'll handle it

def get_semantic_version(model_metrics):
    """
    Generate semantic version based on performance.
    Major = architecture change, Minor = new features, Patch = retraining.
    """
    # You can make this more sophisticated
    base_version = "2.0.0"
    return base_version

# ============================================
# ENV SETUP
# ============================================
ROOT_DIR = Path(__file__).resolve().parent.parent
env_path = ROOT_DIR / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"✅ Loaded .env from: {env_path}")

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if HOPSWORKS_API_KEY:
    HOPSWORKS_API_KEY = HOPSWORKS_API_KEY.strip()

if not HOPSWORKS_API_KEY:
    print("❌ API key missing")
    sys.exit(1)

print(f"🔐 Using API Key: {HOPSWORKS_API_KEY[:5]}*****")

print("=" * 70)
print("⚡ XGBOOST - AQI PREDICTION (PRODUCTION)")
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
# STEP 2: Get Feature View
# ============================================
print("\n🔍 2. Getting Feature View...")

try:
    feature_view = fs.get_feature_view("karachi_aqi_final_view", version=1)
    print(f"   ✅ Found Feature View: {feature_view.name}")
except Exception as e:
    print(f"   ❌ Feature View not found. Run feature_engineering.py first.")
    sys.exit(1)

# ============================================
# STEP 3: Create Training Dataset (Frozen Snapshot)
# ============================================
print("\n📊 3. Creating Training Dataset (frozen snapshot)...")

training_dataset, dataset_version = create_training_dataset(
    feature_view,
    description=f"XGBoost training data - captures data up to {datetime.now().strftime('%Y-%m-%d')}"
)

# Load data from frozen training dataset or Feature View
if hasattr(training_dataset, 'read'):
    df = training_dataset.read()
else:
    df = training_dataset
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"   ✅ Loaded {len(df)} rows from training dataset v{dataset_version}")
print(f"   📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# ============================================
# STEP 4: Prepare Features and Target
# ============================================
print("\n🎯 4. Preparing features and target...")

# Columns to exclude
exclude_cols = ['timestamp', 'city', 'us_aqi', 'european_aqi']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df['us_aqi']

print(f"   ✅ Features: {len(feature_cols)} columns")
print(f"   ✅ Target: us_aqi")

# ============================================
# STEP 5: Split Data (Time Series Split)
# ============================================
print("\n✂️ 5. Splitting data (80% train, 20% test)...")

df_sorted = df.sort_values('timestamp')
X_sorted = X.loc[df_sorted.index]
y_sorted = y.loc[df_sorted.index]

split_idx = int(len(df_sorted) * 0.8)
X_train = X_sorted[:split_idx]
X_test = X_sorted[split_idx:]
y_train = y_sorted[:split_idx]
y_test = y_sorted[split_idx:]

print(f"   📊 Training set: {len(X_train)} rows")
print(f"   📊 Test set: {len(X_test)} rows")

# ============================================
# STEP 6: Train XGBoost Model
# ============================================
print("\n⚡ 6. Training XGBoost model...")

xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)
print("   ✅ XGBoost training complete!")

# ============================================
# STEP 7: Evaluate Model
# ============================================
print("\n📊 7. Model Performance:")

# Predictions
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

# Training metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"\n   📈 Training Set:")
print(f"      RMSE: {train_rmse:.2f}")
print(f"      MAE: {train_mae:.2f}")
print(f"      R²: {train_r2:.3f}")

# Test metrics
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\n   📉 Test Set:")
print(f"      RMSE: {test_rmse:.2f}")
print(f"      MAE: {test_mae:.2f}")
print(f"      R²: {test_r2:.3f}")

# ============================================
# STEP 8: Feature Importance
# ============================================
print("\n🔍 8. Feature Importance Analysis:")

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   📊 Top 10 most important features:")
for i, row in importance_df.head(10).iterrows():
    bar = "█" * int(row['importance'] * 50)
    print(f"   {row['feature']:25s} {row['importance']:.4f} {bar}")

# ============================================
# STEP 9: Save Model
# ============================================
print("\n💾 9. Saving model...")

model_dir = ROOT_DIR / "models"
model_dir.mkdir(exist_ok=True)

# Save with version in filename
model_filename = f"xgboost_aqi_model_{dataset_version}.pkl"
model_path = model_dir / model_filename
joblib.dump(xgb_model, model_path)
print(f"   ✅ Model saved to: {model_path}")

# Also save as latest for dashboard
latest_path = model_dir / "xgboost_aqi_model_tuned.pkl"
joblib.dump(xgb_model, latest_path)
print(f"   ✅ Latest model saved to: {latest_path}")

# ============================================
# STEP 10: Save to Model Registry
# ============================================
print("\n📤 10. Saving to Hopsworks Model Registry...")

try:
    mr = project.get_model_registry()
    semantic_version = get_semantic_version({"rmse": test_rmse})
    
    model_registry_obj = mr.python.create_model(
        name="aqi_predictor_xgboost",
        version=semantic_version,
        description=f"XGBoost model trained on {dataset_version} data. RMSE: {test_rmse:.2f}",
        metrics={
            "train_rmse": float(train_rmse),
            "train_mae": float(train_mae),
            "train_r2": float(train_r2),
            "test_rmse": float(test_rmse),
            "test_mae": float(test_mae),
            "test_r2": float(test_r2),
            "training_dataset_version": dataset_version
        },
        input_example=X_test[:1]
    )
    model_registry_obj.save(str(latest_path))
    print(f"   ✅ Model saved to Hopsworks Model Registry v{semantic_version}")
    print(f"   📊 Associated with Training Dataset: {dataset_version}")
except Exception as e:
    print(f"   ⚠️ Could not save to registry: {e}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("✅ XGBOOST MODEL TRAINING COMPLETE!")
print("=" * 70)
print(f"\n📊 Final Model Performance:")
print(f"   Test RMSE: {test_rmse:.2f} AQI points")
print(f"   Test R²:   {test_r2:.3f}")
print(f"\n📁 Training Dataset: aqi_training_data_{dataset_version}")
print(f"📁 Model: {model_filename}")
print("=" * 70)