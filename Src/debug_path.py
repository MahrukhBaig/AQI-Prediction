# debug_path.py
from pathlib import Path
import os

env_path = Path(__file__).parent.parent / '.env'
print(f"Looking for .env at: {env_path}")
print(f"File exists: {env_path.exists()}")

if env_path.exists():
    with open(env_path) as f:
        print(f"Content: {f.read()[:20]}...")