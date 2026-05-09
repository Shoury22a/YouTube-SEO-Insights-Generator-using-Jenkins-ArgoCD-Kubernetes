import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("GOOGLE_API_KEY not found in .env")
    exit(1)

print(f"Testing generation with Key: {api_key[:5]}...{api_key[-5:]}")

try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
    response = llm.invoke("Hello, are you working?")
    print("\nResponse:")
    print(response.content)
except Exception as e:
    print(f"\nGeneration FAILED:")
    print(e)
