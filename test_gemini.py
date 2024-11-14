import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

def verify_gemini_key():
    api_key = os.getenv('GEMINI_API_KEY')
    print(f"API Key starts with: {api_key[:4]}...")
    
    genai.configure(api_key=api_key)
    
    try:
        models = genai.list_models()
        print("\nAvailable models:")
        for model in models:
            print(f"- {model.name}")
        return True
    except Exception as e:
        print(f"\nError: {str(e)}")
        return False

if __name__ == "__main__":
    verify_gemini_key()