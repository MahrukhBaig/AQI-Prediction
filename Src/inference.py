"""
TEST TELEGRAM ALERTS - Simplified version
Tests your bot connection and sends a test message
"""
import requests
import time

# ============================================
# REPLACE WITH YOUR ACTUAL VALUES
# ============================================
BOT_TOKEN = "8395636326:AAFsKvw7TIN5wBG4z6sSv0dgyMDkXWPB4fs"  # From BotFather
CHAT_ID = "7096033104"      # From userinfobot

def test_bot_connection():
    """Test if bot token is valid"""
    print("=" * 50)
    print("🤖 Testing Telegram Bot Connection")
    print("=" * 50)
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"
    
    try:
        response = requests.get(url, timeout=10)
        if response.ok:
            data = response.json()
            print(f"✅ Bot connected!")
            print(f"   Name: {data['result']['first_name']}")
            print(f"   Username: @{data['result']['username']}")
            return True
        else:
            print(f"❌ Bot connection failed: {response.text}")
            return False
    except requests.exceptions.Timeout:
        print("❌ Timeout - Check your internet connection")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def send_test_message():
    """Send a simple test message"""
    print("\n📤 Sending test message...")
    
    message = """
🧪 <b>AQI Alert System Test</b>

✅ Your bot is working correctly!
✅ Your chat ID is configured properly!

📊 When AQI exceeds 100, you'll receive:
⚠️ Unhealthy Air Quality Alert

📊 When AQI exceeds 150, you'll receive:
🚨 HAZARDOUS AIR QUALITY Alert!

<i>This is a test message. Your system is ready!</i>
"""
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.ok:
            print("✅ Test message sent! Check your Telegram!")
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False
    except requests.exceptions.Timeout:
        print("❌ Timeout - Check your internet/VPN connection")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def get_bot_updates():
    """Get recent updates to verify chat ID"""
    print("\n📥 Checking recent messages...")
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    
    try:
        response = requests.get(url, timeout=10)
        if response.ok:
            data = response.json()
            if data['result']:
                print(f"✅ Found {len(data['result'])} recent messages")
                for update in data['result'][:3]:
                    if 'message' in update:
                        chat = update['message']['chat']
                        print(f"   - Chat ID: {chat['id']} (Username: @{chat.get('username', 'N/A')})")
            else:
                print("⚠️ No recent messages. Send a message to your bot first!")
                print("   Open Telegram and send: /start to your bot")
        else:
            print(f"❌ Failed: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🚀 TELEGRAM ALERT SYSTEM TEST")
    print("=" * 50)
    
    # Step 1: Test bot connection
    if not test_bot_connection():
        print("\n💡 Troubleshooting:")
        print("   1. Make sure BOT_TOKEN is correct")
        print("   2. Disable VPN if connected")
        print("   3. Check your internet connection")
        exit(1)
    
    # Step 2: Get updates to verify chat ID
    get_bot_updates()
    
    # Step 3: Send test message
    print("\n" + "-" * 50)
    if send_test_message():
        print("\n✅ TEST COMPLETE! Check your Telegram app!")
    else:
        print("\n❌ Test failed. Check your configuration.")
    
    print("\n" + "=" * 50)