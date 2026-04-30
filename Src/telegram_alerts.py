"""
TELEGRAM ALERTS - Hazardous AQI Notifications
Sends alerts when AQI exceeds thresholds
"""
import argparse
import requests
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from a .env file in the repo root
load_dotenv()

# ============================================
# CONFIGURATION - REPLACE WITH YOUR VALUES
# ============================================
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # From BotFather
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")      # From userinfobot

# Thresholds
HAZARDOUS_THRESHOLD = 150
UNHEALTHY_THRESHOLD = 100

def send_telegram_message(message):
    """Send message to Telegram"""
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram bot token or chat ID is not configured.")
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.ok:
            return True
        print(f"Telegram API error {response.status_code}: {response.text}")
        return False
    except requests.RequestException as e:
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
        success = send_telegram_message(message)
        if success:
            print(f"HAZARDOUS alert sent! AQI: {aqi:.0f}")
        else:
            print(f"Failed to send hazardous alert. AQI: {aqi:.0f}")
        
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
        success = send_telegram_message(message)
        if success:
            print(f"Unhealthy alert sent! AQI: {aqi:.0f}")
        else:
            print(f"Failed to send unhealthy alert. AQI: {aqi:.0f}")
    else:
        print(f"Air quality is good (AQI: {aqi:.0f}). No alert needed.")


def test_telegram_connection():
    """Send a single test message to verify Telegram configuration."""
    print("Testing Telegram connection...")
    test_message = (
        "✅ Telegram connection verified. This is a test message from the AQI alert script."
    )
    success = send_telegram_message(test_message)
    if success:
        print("Telegram test message sent successfully.")
    else:
        print("Telegram test failed. Check the token/chat ID and network connectivity.")
    return success


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
    parser = argparse.ArgumentParser(description="AQI Telegram alert utility")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Send a one-time test message to verify Telegram connectivity",
    )
    args = parser.parse_args()

    if args.test:
        test_telegram_connection()
    else:
        check_and_alert()

    # Uncomment below for continuous monitoring
    # run_alert_loop()