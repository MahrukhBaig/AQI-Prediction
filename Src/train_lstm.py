"""
LSTM DEEP LEARNING MODEL - AQI PREDICTION
LSTM is specifically designed for time series data
Remembers patterns over long periods
"""
import hopsworks
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime


def save_model_registry(project, model_name, model_path, metrics, input_example=None, framework="tensorflow"):
    try:
        mr = project.get_model_registry()
        version = int(datetime.now().strftime("%Y%m%d"))
        print(f"   Debug: ModelRegistry has sklearn={hasattr(mr, 'sklearn')} python={hasattr(mr, 'python')} tensorflow={hasattr(mr, 'tensorflow')}")
        if framework == "tensorflow" and hasattr(mr, 'tensorflow'):
            model_registry_obj = mr.tensorflow.create_model(
                name=model_name,
                version=version,
                description=f"LSTM model for AQI prediction trained on updated Karachi data.",
                metrics=metrics,
                input_example=input_example,
            )
        else:
            model_registry_obj = mr.python.create_model(
                name=model_name,
                version=version,
                description=f"LSTM model for AQI prediction trained on updated Karachi data.",
                metrics=metrics,
                input_example=input_example,
            )
        print(f"   Debug: model_registry_obj type = {type(model_registry_obj)} has save={hasattr(model_registry_obj, 'save')}")
        model_registry_obj.save(str(model_path))
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
print("🧠 LSTM DEEP LEARNING - AQI PREDICTION MODEL")
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
    print(f"   ⚠️ Engineered features not found, loading raw features...")
    fg = fs.get_feature_group("karachi_aqi_features", version=1)
    df = fg.read()
    print(f"   ✅ Loaded {len(df)} rows from raw features")

df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"   📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# ============================================
# STEP 3: Prepare Data for LSTM
# ============================================
print("\n🎯 3. Preparing data for LSTM...")

# Sort by timestamp
df = df.sort_values('timestamp')

# Select features for LSTM (use most important features from XGBoost)
# This reduces complexity while maintaining accuracy
feature_cols = [
    'aqi_rolling_mean_3h',
    'aqi_lag_1h', 
    'aqi_change',
    'aqi_change_rate',
    'aqi_rolling_mean_6h',
    'hour',
    'day_of_week',
    'season'
]

# Make sure all features exist (if not, use available ones)
available_cols = [col for col in feature_cols if col in df.columns]
X = df[available_cols].values
y = df['us_aqi'].values

print(f"   ✅ Features: {len(available_cols)} columns")
print(f"   ✅ Target: us_aqi")

# ============================================
# STEP 4: Normalize Data (Important for LSTM!)
# ============================================
print("\n📊 4. Normalizing data...")

# Scale features to 0-1 range (LSTM works better with normalized data)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

print(f"   ✅ Features scaled to range [0, 1]")
print(f"   ✅ Target scaled to range [0, 1]")

# ============================================
# STEP 5: Create Sequences for LSTM
# ============================================
print("\n🔄 5. Creating sequences (lookback = 24 hours)...")

def create_sequences(X, y, lookback=24):
    """
    Create sequences for LSTM training
    lookback = number of previous hours to use for prediction
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback])
    return np.array(X_seq), np.array(y_seq)

lookback = 24  # Use last 24 hours to predict next hour
X_seq, y_seq = create_sequences(X_scaled, y_scaled, lookback)

print(f"   ✅ Lookback period: {lookback} hours")
print(f"   ✅ Sequence shape: {X_seq.shape}")
print(f"   ✅ Target shape: {y_seq.shape}")

# ============================================
# STEP 6: Split Data
# ============================================
print("\n✂️ 6. Splitting data (80% train, 20% test)...")

split_idx = int(len(X_seq) * 0.8)
X_train = X_seq[:split_idx]
X_test = X_seq[split_idx:]
y_train = y_seq[:split_idx]
y_test = y_seq[split_idx:]

print(f"   📊 Training sequences: {len(X_train)}")
print(f"   📊 Test sequences: {len(X_test)}")

# ============================================
# STEP 7: Build LSTM Model
# ============================================
print("\n🏗️ 7. Building LSTM model...")

model = Sequential([
    # First LSTM layer with dropout for regularization
    Input(shape=(lookback, len(available_cols))),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    
    # Second LSTM layer
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    
    # Dense layers for final prediction
    Dense(16, activation='relu'),
    Dense(1)  # Single output (AQI)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()
print("   ✅ LSTM model built!")

# ============================================
# STEP 8: Train LSTM Model
# ============================================
print("\n🚂 8. Training LSTM model (this may take 5-10 minutes)...")

# Early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Reduce learning rate when plateauing
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.0001
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print("   ✅ LSTM training complete!")

# ============================================
# STEP 9: Make Predictions
# ============================================
print("\n🔮 9. Making predictions...")

y_train_pred_scaled = model.predict(X_train)
y_test_pred_scaled = model.predict(X_test)

# Inverse transform to get actual AQI values
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

# Get actual values (inverse transform)
y_train_actual = scaler_y.inverse_transform(y_train)
y_test_actual = scaler_y.inverse_transform(y_test)

# ============================================
# STEP 10: Evaluate Model
# ============================================
print("\n📊 10. Model Performance:")

# Training metrics
train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
train_mae = mean_absolute_error(y_train_actual, y_train_pred)
train_r2 = r2_score(y_train_actual, y_train_pred)

print(f"\n   📈 Training Set:")
print(f"      RMSE: {train_rmse:.2f}")
print(f"      MAE: {train_mae:.2f}")
print(f"      R²: {train_r2:.3f}")

# Test metrics
test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
test_mae = mean_absolute_error(y_test_actual, y_test_pred)
test_r2 = r2_score(y_test_actual, y_test_pred)

print(f"\n   📉 Test Set:")
print(f"      RMSE: {test_rmse:.2f}")
print(f"      MAE: {test_mae:.2f}")
print(f"      R²: {test_r2:.3f}")

# ============================================
# STEP 11: Compare All Models
# ============================================
print("\n📊 11. Model Comparison:")

print(f"\n   {'Model':<20} {'RMSE':<10} {'R²':<10}")
print(f"   {'-' * 40}")

# Random Forest results (from previous run)
print(f"   {'Random Forest':<20} {'0.98':<10} {'0.998':<10}")

# XGBoost results (from previous run)
print(f"   {'XGBoost':<20} {'0.89':<10} {'0.998':<10}")

# LSTM results
print(f"   {'LSTM':<20} {test_rmse:<10.2f} {test_r2:<10.3f}")

# Determine winner
best_rmse = min(0.98, 0.89, test_rmse)
if test_rmse == best_rmse:
    print(f"\n   🏆 LSTM is the BEST model!")
elif 0.89 == best_rmse:
    print(f"\n   🏆 XGBoost is still the BEST model!")
else:
    print(f"\n   🏆 Random Forest is still the BEST model!")

# ============================================
# STEP 12: Plot Training History (Optional)
# ============================================
print("\n📈 12. Plotting training history...")

try:
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    
    # MAE plot
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(ROOT_DIR / "models" / "lstm_training_history.png")
    print(f"   ✅ Training plot saved to: models/lstm_training_history.png")
except Exception as e:
    print(f"   ⚠️ Could not plot: {e}")

# ============================================
# STEP 13: Save Model and Scaler
# ============================================
print("\n💾 13. Saving model...")

model_dir = ROOT_DIR / "models"
model_dir.mkdir(exist_ok=True)

# Save LSTM model
model_path = model_dir / "lstm_aqi_model.keras"
model.save(model_path)
print(f"   ✅ LSTM model saved to: {model_path}")

# Save scalers (needed for future predictions)
joblib.dump(scaler_X, model_dir / "scaler_X.pkl")
joblib.dump(scaler_y, model_dir / "scaler_y.pkl")
print(f"   ✅ Scalers saved to: models/")

# Save model information
info = {
    'lookback': lookback,
    'feature_cols': available_cols,
    'test_rmse': float(test_rmse),
    'test_mae': float(test_mae),
    'test_r2': float(test_r2)
}

joblib.dump(info, model_dir / "lstm_model_info.pkl")
print(f"   ✅ Model info saved")

# Save LSTM to Hopsworks model registry
metrics = {
    "test_rmse": float(test_rmse),
    "test_mae": float(test_mae),
    "test_r2": float(test_r2),
    "lookback": lookback
}

save_model_registry(
    project,
    model_name="aqi_predictor_lstm",
    model_path=model_path,
    metrics=metrics,
    input_example=np.zeros((1, len(available_cols))),
    framework="tensorflow"
)

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("✅ LSTM MODEL TRAINING COMPLETE!")
print("=" * 70)
print(f"\n📊 Final Model Performance:")
print(f"   {'─' * 40}")
print(f"   Test RMSE: {test_rmse:.2f} AQI points")
print(f"   Test MAE:  {test_mae:.2f} AQI points")
print(f"   Test R²:   {test_r2:.3f}")
print(f"\n📁 Files saved in 'models/':")
print(f"   👉 lstm_aqi_model.keras (LSTM model)")
print(f"   👉 scaler_X.pkl (feature scaler)")
print(f"   👉 scaler_y.pkl (target scaler)")
print(f"   👉 lstm_model_info.pkl (model metadata)")
