import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

print(f"Checking API Key: {api_key[:5]}...{api_key[-5:]}")

# Try v1 instead of v1beta
url = f"https://generativelanguage.googleapis.com/v1/models?key={api_key}"
try:
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        models = response.json().get("models", [])
        print("\nAvailable Models (v1):")
        for m in models:
            print(f" - {m.get('name')}")
    else:
        print(f"v1 failed: {response.status_code}")

    # Also try v1beta again but list ALL
    url_beta = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    response_beta = requests.get(url_beta, timeout=10)
    if response_beta.status_code == 200:
        models = response_beta.json().get("models", [])
        print("\nAvailable Models (v1beta):")
        for m in models:
            print(f" - {m.get('name')}")
except Exception as e:
    print(f"Failed: {e}")
