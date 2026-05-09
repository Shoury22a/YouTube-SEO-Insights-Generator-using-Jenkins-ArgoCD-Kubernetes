import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("GOOGLE_API_KEY not found in .env")
    exit(1)

print(f"Checking API Key: {api_key[:5]}...{api_key[-5:]}")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
try:
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        models = response.json().get("models", [])
        print("\nAvailable Models:")
        found_flash_2 = False
        for m in models:
            name = m.get("name", "")
            if "gemini-2.0-flash" in name:
                print(f" [OK] {name}")
                found_flash_2 = True
            elif "gemini-1.5" in name:
                print(f" [OK] {name}")
        
        if not found_flash_2:
            print("\n!!! gemini-2.0-flash NOT FOUND in available models for this key.")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Failed: {e}")
