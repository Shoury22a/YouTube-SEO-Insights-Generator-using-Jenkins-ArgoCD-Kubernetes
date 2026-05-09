import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

model = "gemini-flash-latest"
url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
headers = {'Content-Type': 'application/json'}
data = {"contents": [{"parts": [{"text": "hi"}]}]}

try:
    response = requests.post(url, headers=headers, data=json.dumps(data), timeout=5)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
         print("SUCCESS!")
         print(response.json().get('candidates')[0].get('content').get('parts')[0].get('text'))
    else:
        print(response.text)
except Exception as e:
    print(f"Error: {e}")
