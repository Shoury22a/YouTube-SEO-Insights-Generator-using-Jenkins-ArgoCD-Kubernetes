import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

models = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro"
]

for model in models:
    print(f"Testing {model}...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": "hi"}]}]}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=5)
        if response.status_code == 200:
            print(f" [SUCCESS] {model}")
        else:
            print(f" [FAILED] {model} - {response.status_code}")
            # print(response.text)
    except Exception as e:
        print(f" [ERROR] {model} - {e}")
