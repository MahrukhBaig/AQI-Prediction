"""
STREAMLIT DASHBOARD - AQI PREDICTION
Complete version with all 43 features
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌍",
    layout="wide"
)

# ============================================
# ALL 43 FEATURES (Complete list)
# ============================================
FEATURE_COLS = [
    'hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend', 'is_rush_hour', 'season',
    'aqi_lag_1h', 'aqi_lag_3h', 'aqi_lag_6h', 'aqi_lag_12h', 'aqi_lag_24h',
    'aqi_rolling_mean_3h', 'aqi_rolling_mean_6h', 'aqi_rolling_mean_12h', 'aqi_rolling_mean_24h',
    'aqi_rolling_std_6h', 'aqi_rolling_std_24h', 'aqi_rolling_max_6h', 'aqi_rolling_min_6h',
    'aqi_change', 'aqi_change_rate',
    'pm25_pm10_ratio', 'temp_humidity', 'wind_pressure',
    'pm25_lag_6h', 'pm25_lag_24h',
    'temp_squared', 'is_raining', 'high_pressure',
    'pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'ozone', 'sulphur_dioxide',
    'temperature', 'humidity', 'wind_speed', 'wind_category', 'pressure', 'precipitation', 'cloudcover'
]

def get_aqi_status(aqi):
    if aqi <= 50:
        return "🟢 Good", "#00e400"
    elif aqi <= 100:
        return "🟡 Moderate", "#ffff00"
    elif aqi <= 150:
        return "🟠 Unhealthy for Sensitive", "#ff7e00"
    elif aqi <= 200:
        return "🔴 Unhealthy", "#ff0000"
    elif aqi <= 300:
        return "🟣 Very Unhealthy", "#8f3f97"
    else:
        return "⚫ Hazardous", "#7e0023"

def engineer_features(df):
    """Create all 43 features"""
    df = df.copy()
    
    # Ensure all raw columns exist
    raw_cols = ['pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 
                'ozone', 'sulphur_dioxide', 'temperature', 'humidity', 
                'wind_speed', 'pressure', 'precipitation', 'cloudcover']
    for col in raw_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Rush hour (7-9 AM and 5-7 PM)
    df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                          (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    
    # Season (Winter=0, Spring=1, Summer=2, Autumn=3)
    df['season'] = df['month'].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 
                                    6:2, 7:2, 8:2, 9:3, 10:3, 11:3}).fillna(0).astype(int)
    
    # Wind speed categories
    df['wind_category'] = pd.cut(df['wind_speed'], 
                                   bins=[0, 2, 5, 10, 100], 
                                   labels=[0, 1, 2, 3]).astype(int)
    
    # Lag features
    df['aqi_lag_1h'] = df['us_aqi'].shift(1)
    df['aqi_lag_3h'] = df['us_aqi'].shift(3)
    df['aqi_lag_6h'] = df['us_aqi'].shift(6)
    df['aqi_lag_12h'] = df['us_aqi'].shift(12)
    df['aqi_lag_24h'] = df['us_aqi'].shift(24)
    
    # Rolling statistics
    df['aqi_rolling_mean_3h'] = df['us_aqi'].rolling(3).mean()
    df['aqi_rolling_mean_6h'] = df['us_aqi'].rolling(6).mean()
    df['aqi_rolling_mean_12h'] = df['us_aqi'].rolling(12).mean()
    df['aqi_rolling_mean_24h'] = df['us_aqi'].rolling(24).mean()
    df['aqi_rolling_std_6h'] = df['us_aqi'].rolling(6).std()
    df['aqi_rolling_std_24h'] = df['us_aqi'].rolling(24).std()
    df['aqi_rolling_max_6h'] = df['us_aqi'].rolling(6).max()
    df['aqi_rolling_min_6h'] = df['us_aqi'].rolling(6).min()
    
    # Change features
    df['aqi_change'] = df['us_aqi'].diff()
    df['aqi_change_rate'] = df['us_aqi'].pct_change() * 100
    
    # Ratio features
    df['pm25_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 0.01)
    df['temp_humidity'] = df['temperature'] * df['humidity']
    df['wind_pressure'] = df['wind_speed'] * df['pressure']
    
    # Pollutant lags
    df['pm25_lag_6h'] = df['pm2_5'].shift(6)
    df['pm25_lag_24h'] = df['pm2_5'].shift(24)
    
    # Weather derived
    df['temp_squared'] = df['temperature'] ** 2
    df['is_raining'] = (df['precipitation'] > 0.1).astype(int)
    df['high_pressure'] = (df['pressure'] > 1015).astype(int)
    
    # Fill NaN values
    df = df.ffill().fillna(0)
    
    return df

def load_models():
    models = {}
    model_dir = ROOT_DIR / "models"
    
    xgb_path = model_dir / "xgboost_aqi_model_tuned.pkl"
    if xgb_path.exists():
        models['xgboost'] = joblib.load(xgb_path)
    
    rf_path = model_dir / "random_forest_aqi_model.pkl"
    if rf_path.exists():
        models['random_forest'] = joblib.load(rf_path)
    
    return models

def load_data():
    csv_path = ROOT_DIR / "karachi_aqi_2025.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return None

def predict_future(df, model, hours=72):
    if df is None or model is None or len(df) < 100:
        return None
    
    predictions = []
    temp_df = df.tail(200).copy()
    
    for i in range(hours):
        temp_df = engineer_features(temp_df)
        
        if len(temp_df) == 0:
            break
        
        # Get features in correct order
        X = temp_df[FEATURE_COLS].iloc[-1:].values
        
        try:
            pred = model.predict(X)[0]
        except Exception as e:
            st.warning(f"Prediction error: {e}")
            break
        
        next_time = temp_df['timestamp'].iloc[-1] + timedelta(hours=1)
        predictions.append({'timestamp': next_time, 'predicted_aqi': pred})
        
        # Create new row
        new_row = temp_df.iloc[-1:].copy()
        new_row['timestamp'] = next_time
        new_row['us_aqi'] = pred
        
        # Preserve raw columns
        for col in ['pm10', 'pm2_5', 'temperature', 'humidity', 'wind_speed', 
                     'pressure', 'precipitation', 'cloudcover', 'carbon_monoxide',
                     'nitrogen_dioxide', 'ozone', 'sulphur_dioxide']:
            if col not in new_row.columns and col in temp_df.columns:
                new_row[col] = temp_df[col].iloc[-1]
        
        temp_df = pd.concat([temp_df, new_row], ignore_index=True)
    
    return pd.DataFrame(predictions)

# ============================================
# MAIN DASHBOARD
# ============================================

def main():
    st.title("🌍 Karachi AQI Predictor")
    st.markdown("*Real-time Air Quality Monitoring and Forecasting*")
    st.markdown("---")
    
    with st.sidebar:
        st.header("📊 Controls")
        model_choice = st.selectbox("Select Model", ["XGBoost (Recommended)", "Random Forest"])
        st.markdown("---")
        st.header("ℹ️ About")
        st.info(f"""
        **Model Performance:**
        - XGBoost RMSE: 0.87 AQI points
        - Random Forest RMSE: 0.98 AQI points
        
        **Features used:** {len(FEATURE_COLS)} features
        """)
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()
    
    with st.spinner("Loading..."):
        df = load_data()
        models = load_models()
        model = models.get('xgboost') if model_choice == "XGBoost (Recommended)" else models.get('random_forest')
    
    if df is None:
        st.error("❌ No data found!")
        return
    
    if model is None:
        st.error("❌ No model found! Please train models first.")
        return
    
    # Current AQI
    current_aqi = df['us_aqi'].iloc[-1]
    current_time = df['timestamp'].iloc[-1]
    status, color = get_aqi_status(current_aqi)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current AQI", f"{current_aqi:.0f}")
        st.write(f"**Status:** {status}")
        st.caption(f"Updated: {current_time.strftime('%Y-%m-%d %H:%M')}")
    
    with col2:
        if current_aqi > 150:
            st.error("🚨 **HAZARDOUS!** Avoid outdoor activities!")
        elif current_aqi > 100:
            st.warning("⚠️ **UNHEALTHY!** Sensitive groups limit exposure.")
        else:
            st.success(f"✅ **Good air quality!** Enjoy your day!")
    
    with col3:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=current_aqi,
            title={'text': "AQI Meter"},
            gauge={
                'axis': {'range': [0, 300]},
                'steps': [
                    {'range': [0, 50], 'color': "#00e400"},
                    {'range': [50, 100], 'color': "#ffff00"},
                    {'range': [100, 150], 'color': "#ff7e00"},
                    {'range': [150, 200], 'color': "#ff0000"},
                    {'range': [200, 300], 'color': "#8f3f97"}
                ]
            }
        ))
        fig.update_layout(height=200)
        st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # Historical Trends
    st.subheader("📈 Historical AQI Trends")
    days = st.selectbox("Time range (days)", [7, 14, 30, 60, 90], index=2)
    df_filtered = df.tail(days * 24)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filtered['timestamp'],
        y=df_filtered['us_aqi'],
        mode='lines',
        name='Actual AQI',
        line=dict(color='blue', width=1.5)
    ))
    fig.add_hline(y=150, line_dash="dash", line_color="red", annotation_text="Hazardous")
    fig.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Unhealthy")
    fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good")
    fig.update_layout(height=400, xaxis_title="Date/Time", yaxis_title="AQI")
    st.plotly_chart(fig, width='stretch')
    
    # Forecast
    st.subheader("🔮 3-Day AQI Forecast")
    with st.spinner("Generating forecast..."):
        forecast = predict_future(df, model, hours=72)
    
    if forecast is not None and len(forecast) > 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast['timestamp'],
            y=forecast['predicted_aqi'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=2),
            marker=dict(size=3)
        ))
        fig.add_hline(y=150, line_dash="dash", line_color="red", annotation_text="Hazardous")
        fig.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Unhealthy")
        fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good")
        fig.update_layout(height=400, xaxis_title="Date/Time", yaxis_title="Predicted AQI")
        st.plotly_chart(fig, width='stretch')
        
        forecast['date'] = forecast['timestamp'].dt.date
        daily = forecast.groupby('date')['predicted_aqi'].agg(['mean', 'max', 'min']).round(0)
        daily.columns = ['Avg AQI', 'Max AQI', 'Min AQI']
        st.dataframe(daily, width='stretch')
        
        max_val = forecast['predicted_aqi'].max()
        if max_val > 150:
            st.error(f"🚨 **ALERT!** Hazardous AQI ({max_val:.0f}) predicted!")
        elif max_val > 100:
            st.warning(f"⚠️ **CAUTION!** Unhealthy AQI ({max_val:.0f}) predicted!")
    
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Features: {len(FEATURE_COLS)}")

if __name__ == "__main__":
    main()