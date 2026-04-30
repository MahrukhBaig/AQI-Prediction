"""
TELEGRAM ALERTS - Hazardous AQI Notifications
Sends alerts when AQI exceeds thresholds
"""
import requests
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
import os

# ============================================
# CONFIGURATION - REPLACE WITH YOUR VALUES
# ============================================
BOT_TOKEN = "8395636326:AAFsKvw7TIN5wBG4z6sSv0dgyMDkXWPB4fs"  # From BotFather
CHAT_ID = "7096033104"      # From userinfobot

# Thresholds
HAZARDOUS_THRESHOLD = 50
UNHEALTHY_THRESHOLD = 70

def send_telegram_message(message):
    """Send message to Telegram"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        response = requests.post(url, json=payload)
        return response.ok
    except Exception as e:
        print(f"Error sending message: {e}")
        return False

def get_latest_aqi():
    """Get latest AQI from CSV"""
    csv_path = Path(__file__).resolve().parent.parent / "karachi_aqi_2025.csv"
    
    if not csv_path.exists():
        return None, None
    
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    latest = df.iloc[-1]
    return latest['us_aqi'], latest['timestamp']

def get_forecast_summary():
    """Get forecast summary (simplified)"""
    csv_path = Path(__file__).resolve().parent.parent / "karachi_aqi_2025.csv"
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Simple forecast based on recent trend
    recent = df['us_aqi'].tail(24)
    avg = recent.mean()
    trend = "increasing" if recent.iloc[-1] > recent.iloc[-6] else "decreasing"
    
    return avg, trend

def check_and_alert():
    """Check AQI and send alerts if needed"""
    
    # Get latest AQI
    aqi, timestamp = get_latest_aqi()
    
    if aqi is None:
        print("No data available")
        return
    
    # Get forecast
    forecast_avg, trend = get_forecast_summary()
    
    # Current time
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Send alerts based on AQI level
    if aqi >= HAZARDOUS_THRESHOLD:
        message = f"""
🚨 <b>HAZARDOUS AIR QUALITY ALERT!</b> 🚨

📍 <b>Location:</b> Karachi
📊 <b>Current AQI:</b> {aqi:.0f}
⏰ <b>Time:</b> {now}
⚠️ <b>Status:</b> Hazardous - Health alert for everyone!

<b>Recommendations:</b>
• Stay indoors with windows closed
• Use air purifier if available
• Wear N95 mask if you must go outside
• Avoid outdoor exercise

🔮 <b>Forecast:</b> AQI {forecast_avg:.0f} ({trend})

<i>Stay safe! Check dashboard for updates.</i>
        """
        send_telegram_message(message)
        print(f"HAZARDOUS alert sent! AQI: {aqi:.0f}")
        
    elif aqi >= UNHEALTHY_THRESHOLD:
        message = f"""
⚠️ <b>Unhealthy Air Quality Alert</b> ⚠️

📍 <b>Location:</b> Karachi
📊 <b>Current AQI:</b> {aqi:.0f}
⏰ <b>Time:</b> {now}

<b>Recommendations:</b>
• Sensitive groups limit outdoor activities
• Keep windows closed
• Use air purifier if available

🔮 <b>Forecast:</b> AQI {forecast_avg:.0f} ({trend})
        """
        send_telegram_message(message)
        print(f"Unhealthy alert sent! AQI: {aqi:.0f}")
    else:
        print(f"Air quality is good (AQI: {aqi:.0f}). No alert needed.")

def run_alert_loop(interval_minutes=60):
    """Run continuous monitoring"""
    print(f"Starting AQI Alert System...")
    print(f"Checking every {interval_minutes} minutes")
    print(f"Hazardous threshold: {HAZARDOUS_THRESHOLD}")
    print(f"Unhealthy threshold: {UNHEALTHY_THRESHOLD}")
    print("-" * 40)
    
    while True:
        check_and_alert()
        print(f"Next check in {interval_minutes} minutes...")
        time.sleep(interval_minutes * 60)

if __name__ == "__main__":
    # Run once
    check_and_alert()
    
    # Uncomment below for continuous monitoring
    # run_alert_loop()