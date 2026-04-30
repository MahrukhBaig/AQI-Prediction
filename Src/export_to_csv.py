"""
EXPORT DATA FROM HOPSWORKS TO CSV
Reads all data from Hopsworks feature group and saves to CSV
"""
import hopsworks
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv

# Load API key from .env
ROOT_DIR = Path(__file__).resolve().parent.parent
env_path = ROOT_DIR / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if HOPSWORKS_API_KEY:
    HOPSWORKS_API_KEY = HOPSWORKS_API_KEY.strip()

if not HOPSWORKS_API_KEY:
    print("❌ API key missing")
    exit(1)

print("=" * 50)
print("EXPORTING DATA FROM HOPSWORKS TO CSV")
print("=" * 50)

try:
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        host="eu-west.cloud.hopsworks.ai"
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group("karachi_aqi_features", version=1)
    df = fg.read()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Save to CSV
    csv_path = ROOT_DIR / "karachi_aqi_2025.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Exported {len(df)} rows to {csv_path}")
    
except Exception as e:
    print(f"❌ Failed to export: {e}")
    exit(1)