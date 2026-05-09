import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai_model import _build_llm_with_fallback

def test_chain():
    load_dotenv()
    print("--- Multi-Provider Chain Test ---")
    
    try:
        llm = _build_llm_with_fallback()
        print("SUCCESS: Chain Built Successfully")
        
        # Access the fallbacks to see if Groq is there
        if hasattr(llm, "fallbacks"):
            print(f"Number of fallbacks: {len(llm.fallbacks)}")
            for i, f in enumerate(llm.fallbacks):
                model_name = getattr(f, "model_name", getattr(f, "model", "Unknown"))
                provider = "Groq" if "ChatGroq" in str(type(f)) else "Gemini"
                print(f"  Fallback {i+1}: {model_name} ({provider})")
        
        # Test a simple invocation
        print("\nTesting invocation (this will try primary first)...")
        # We use a very simple prompt to minimize token use
        response = llm.invoke("Hi, please respond with 'READY' if you can hear me.")
        print(f"Response: {response.content}")
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_chain()
