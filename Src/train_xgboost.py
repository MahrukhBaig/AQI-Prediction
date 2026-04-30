"""
TRAIN XGBOOST MODEL - AQI PREDICTION
XGBoost often outperforms Random Forest for tabular data
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


def save_model_registry(project, model_name, model_path, metrics):
    try:
        mr = project.get_model_registry()
        version = int(datetime.now().strftime("%Y%m%d"))
        mr.create_model(
            name=model_name,
            version=version,
            description=f"XGBoost model for AQI prediction trained on updated Karachi data.",
            metrics=metrics,
            model_path=str(model_path)
        )
        print(f"   ✅ Saved {model_name} to Hopsworks Model Registry v{version}")
    except Exception as e:
        print(f"   ⚠️ Could not save {model_name} to registry: {e}")

# ============================================
# ENV SETUP (SAME AS WORKING SCRIPT)
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
print("⚡ XGBOOST - AQI PREDICTION MODEL")
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
# STEP 2: Load Engineered Features
# ============================================
print("\n📂 2. Loading engineered features...")

try:
    fg = fs.get_feature_group("karachi_aqi_engineered_features", version=1)
    df = fg.read()
    print(f"   ✅ Loaded {len(df)} rows from engineered features")
except Exception as e:
    print(f"   ⚠️ Engineered features not found, trying raw features...")
    fg = fs.get_feature_group("karachi_aqi_features", version=1)
    df = fg.read()
    print(f"   ✅ Loaded {len(df)} rows from raw features")

df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"   📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# ============================================
# STEP 3: Prepare Features and Target
# ============================================
print("\n🎯 3. Preparing features and target...")

# Columns to exclude
exclude_cols = ['timestamp', 'city', 'us_aqi', 'european_aqi']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df['us_aqi']

print(f"   ✅ Features: {len(feature_cols)} columns")
print(f"   ✅ Target: us_aqi")

# ============================================
# STEP 4: Split Data (Time Series Split)
# ============================================
print("\n✂️ 4. Splitting data (80% train, 20% test)...")

# Sort by timestamp (critical for time series!)
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
# STEP 5: Train XGBoost Model
# ============================================
print("\n⚡ 5. Training XGBoost model...")

# Base XGBoost model
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
print("   ✅ Base XGBoost training complete!")

# ============================================
# STEP 6: Make Predictions
# ============================================
print("\n🔮 6. Making predictions...")

y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

# ============================================
# STEP 7: Evaluate Model
# ============================================
print("\n📊 7. Model Performance:")

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
# STEP 8: Feature Importance (XGBoost)
# ============================================
print("\n🔍 8. Feature Importance Analysis:")

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   📊 Top 15 most important features:")
print("   " + "-" * 55)
for i, row in importance_df.head(15).iterrows():
    bar = "█" * int(row['importance'] * 50)
    print(f"   {row['feature']:25s} {row['importance']:.4f} {bar}")

# ============================================
# STEP 9: Compare with Random Forest
# ============================================
print("\n📊 9. Model Comparison:")

# Load Random Forest model if exists
rf_model_path = ROOT_DIR / "models" / "random_forest_aqi_model.pkl"
if rf_model_path.exists():
    rf_model = joblib.load(rf_model_path)
    rf_test_pred = rf_model.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
    rf_r2 = r2_score(y_test, rf_test_pred)
    
    print(f"\n   {'Model':<20} {'RMSE':<10} {'R²':<10}")
    print(f"   {'-' * 40}")
    print(f"   {'Random Forest':<20} {rf_rmse:<10.2f} {rf_r2:<10.3f}")
    print(f"   {'XGBoost':<20} {test_rmse:<10.2f} {test_r2:<10.3f}")
    
    if test_rmse < rf_rmse:
        print(f"\n   ✅ XGBoost performs BETTER than Random Forest!")
    else:
        print(f"\n   ℹ️ Random Forest still performs better.")
else:
    print(f"\n   ℹ️ Random Forest model not found for comparison.")

# ============================================
# STEP 10: Save Model
# ============================================
print("\n💾 10. Saving model...")

model_dir = ROOT_DIR / "models"
model_dir.mkdir(exist_ok=True)

# Save XGBoost model
model_path = model_dir / "xgboost_aqi_model.pkl"
joblib.dump(xgb_model, model_path)
print(f"   ✅ XGBoost model saved to: {model_path}")

# ============================================
# STEP 11: Try Hyperparameter Tuning (Optional)
# ============================================
print("\n🔧 11. Hyperparameter Tuning (Optional)...")

tune = os.getenv("TUNE_XGBOOST")
if tune is None:
    if sys.stdin.isatty():
        tune = input("\n   Do you want to tune hyperparameters? (y/n): ").lower()
    else:
        tune = 'n'
else:
    tune = tune.lower()

if tune == 'y':
    print("\n   ⚙️ Running GridSearchCV (this may take 5-10 minutes)...")
    
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.15],
        'n_estimators': [50, 100, 150]
    }
    
    grid_search = GridSearchCV(
        xgb.XGBRegressor(random_state=42, n_jobs=-1),
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n   ✅ Best parameters: {grid_search.best_params_}")
    print(f"   ✅ Best CV score: {np.sqrt(-grid_search.best_score_):.2f}")
    
    # Train best model
    best_xgb = grid_search.best_estimator_
    best_pred = best_xgb.predict(X_test)
    best_rmse = np.sqrt(mean_squared_error(y_test, best_pred))
    best_mae = mean_absolute_error(y_test, best_pred)
    best_r2 = r2_score(y_test, best_pred)
    
    print(f"\n   📊 Tuned XGBoost Test RMSE: {best_rmse:.2f}")
    
    if best_rmse < test_rmse:
        print(f"   ✅ Tuning improved the model!")
        # Save tuned model
        tuned_path = model_dir / "xgboost_aqi_model_tuned.pkl"
        joblib.dump(best_xgb, tuned_path)
        print(f"   ✅ Tuned model saved to: {tuned_path}")
        xgb_model = best_xgb
        test_rmse = best_rmse
        test_mae = best_mae
        test_r2 = best_r2

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("✅ XGBOOST MODEL TRAINING COMPLETE!")
print("=" * 70)
print(f"\n📊 Final Model Performance:")
print(f"   {'─' * 40}")
print(f"   Test RMSE: {test_rmse:.2f} AQI points")
print(f"   Test MAE:  {test_mae:.2f} AQI points")
print(f"   Test R²:   {test_r2:.3f}")
final_model_path = model_path
final_metrics = {
    "test_rmse": float(test_rmse),
    "test_mae": float(test_mae),
    "test_r2": float(test_r2)
}

if tune == 'y' and 'tuned_path' in locals():
    final_model_path = tuned_path
    final_metrics = {
        "test_rmse": float(best_rmse),
        "test_mae": float(best_mae),
        "test_r2": float(best_r2)
    }

save_model_registry(
    project,
    model_name="aqi_predictor_xgboost",
    model_path=final_model_path,
    metrics=final_metrics
)

print(f"\n📁 Model saved at: {final_model_path}")
