# AQI-Prediction

## Project Overview

This repository builds an Air Quality Index (AQI) prediction system for Karachi.
It collects hourly air quality and weather data, creates ML features, trains models, and shows results in a dashboard.


## Source files and their roles

### `Src/dashboard.py`
- Main Streamlit dashboard application.
- Loads data from Hopsworks if available, otherwise from `karachi_aqi_2025.csv`.
- Loads trained models from `models/`.
- Displays current AQI and prediction charts.

### `Src/hourly_fetch_hopsworks.py`
- Hourly data ingestion script.
- Fetches new hourly air quality and weather data.
- Uploads new rows into Hopsworks feature store.

### `Src/export_to_csv.py`
- Reads the Hopsworks feature group.
- Writes the data into the local CSV `karachi_aqi_2025.csv`.
- Keeps the dashboard CSV up to date.

### `Src/telegram_alerts.py`
- Checks the latest AQI.
- Sends alerts via Telegram when AQI passes thresholds.

### `Src/feature_pipeline.py`
- Fetches the full 2025 dataset from Open-Meteo.
- Merges air quality and weather data.
- Saves the combined dataset to `karachi_aqi_2025.csv`.

### `Src/feature_engineering.py`
- Creates ML features from raw data.
- Adds time features, lag features, rolling statistics, and weather interactions.
- Uploads engineered feature data back to Hopsworks.

### `Src/train_model.py`
- Trains a Random Forest AQI model.
- Saves the model locally.
- Saves a versioned model entry to Hopsworks model registry.

### `Src/train_xgboost.py`
- Trains an XGBoost AQI model.
- Saves the model locally.
- Saves a versioned model entry to Hopsworks model registry.
- Supports non-interactive runs for CI.

### `Src/train_lstm.py`
- Trains an LSTM time-series model.
- Saves the model, scalers, and metadata locally.
- Saves a versioned model entry to Hopsworks model registry.

### `Src/shap_analysis.py`
- Uses SHAP to explain model predictions.
- Generates plots showing which features mattered most.

## Project workflow

### 1. Data collection
- `Src/hourly_fetch_hopsworks.py` runs hourly.
- It checks Hopsworks for the latest timestamp.
- It fetches new hourly air quality and weather data.
- It uploads the new rows into Hopsworks.

### 2. Data sync for dashboard
- `Src/export_to_csv.py` reads the Hopsworks feature group and writes local CSV.
- This ensures `dashboard.py` can use the latest data.

### 3. Feature engineering
- `Src/feature_engineering.py` transforms raw data into ML-ready inputs.
- It creates useful features such as lag values, trends, and weather interactions.
- The new engineered data is stored in Hopsworks.

### 4. Model training
- `Src/train_xgboost.py`, `Src/train_model.py`, and `Src/train_lstm.py` train models from the engineered features.
- Models are saved locally in `models/`.
- Models are also registered in Hopsworks model registry with a version based on the current date.

### 5. Dashboard display
- `Src/dashboard.py` loads the latest data and models.
- It shows current AQI and forecasts.
- It uses the best available model.

### 6. Alerts
- `Src/telegram_alerts.py` can send notifications for poor air quality.
- It reads the latest AQI from the updated CSV.

## Automation

### Hourly workflow
- `.github/workflows/hourly_fetch.yml`
- Runs every hour at minute 0.
- Fetches new data and updates Hopsworks and the CSV.
- Sends Telegram alerts if needed.

### Daily retraining workflow
- `.github/workflows/daily_retrain.yml`
- Runs every day at `03:00 UTC`.
- Rebuilds engineered features.
- Retrains XGBoost, Random Forest, and LSTM models.
- Saves models in Hopsworks model registry.


```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the dashboard:

```bash
streamlit run Src/dashboard.py