# test_env.py
from dotenv import load_dotenv
import os

load_dotenv()

def test_env_vars():
    print("Checking environment variables...")
    vars_to_check = [
        'OPENAI_API_KEY',
        'ANTHROPIC_API_KEY',
        'PERPLEXITY_API_KEY',
        'SERPAPI_API_KEY'
    ]
    
    for var in vars_to_check:
        value = os.getenv(var)
        if value:
            print(f"{var}: Found (value hidden)")
        else:
            print(f"{var}: Not found")

if __name__ == "__main__":
    test_env_vars()git add