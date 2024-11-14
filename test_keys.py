import os
import json
import requests
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import openai
import anthropic
from datetime import datetime

load_dotenv()

class APITester:
    def __init__(self):
        self.results = {}
        self.test_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    def test_openai(self):
        print("\nTesting OpenAI API...")
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return {"status": "Failed", "error": "OPENAI_API_KEY not found in .env"}
            
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello, how are you?"}]
            )
            return {"status": "Success", "response": str(response)}
        except Exception as e:
            return {"status": "Failed", "error": str(e)}

    def test_anthropic(self):
        print("\nTesting Anthropic API...")
        try:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                return {"status": "Failed", "error": "ANTHROPIC_API_KEY not found in .env"}
            
            client = anthropic.Client(api_key=api_key)
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": "Hello, how are you?"}]
            )
            return {"status": "Success", "response": str(response)}
        except Exception as e:
            # Add more detailed error logging
            print(f"\nDetailed error: {str(e)}")
            return {"status": "Failed", "error": str(e)}
        
    def test_perplexity(self):
        print("\nTesting Perplexity API...")
        try:
            api_key = os.getenv('PERPLEXITY_API_KEY')
            if not api_key:
                return {"status": "Failed", "error": "PERPLEXITY_API_KEY not found in .env"}
            
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            data = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [
                    {
                        "role": "system",
                        "content": "Be precise and concise."
                    },
                    {
                        "role": "user",
                        "content": "Hello, how are you?"
                    }
                ],
                "temperature": 0.2,
                "top_p": 0.9
            }
            response = requests.post(
                'https://api.perplexity.ai/chat/completions',
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return {"status": "Success", "response": response.json()}
        except Exception as e:
            return {"status": "Failed", "error": str(e)}

    def test_gemini(self):
        print("\nTesting Gemini API...")
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                return {"status": "Failed", "error": "GEMINI_API_KEY not found in .env"}
            
            import google.generativeai as genai
            
            # Configure the API
            genai.configure(api_key=api_key)
            
            # Create model and generate content
            model = genai.GenerativeModel('gemini-pro')  # Using the stable model version
            response = model.generate_content(
                "Hello, how are you?",
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
            )
            
            # Check if response was successful
            if hasattr(response, 'text'):
                return {
                    "status": "Success",
                    "response": {
                        "text": response.text,
                        "model": "gemini-pro"
                    }
                }
            else:
                return {"status": "Failed", "error": "No text in response"}
                
        except Exception as e:
            return {"status": "Failed", "error": str(e)}

    def test_grok(self):
        print("\nTesting Grok API...")
        try:
            api_key = os.getenv('GROK_API_KEY')
            if not api_key:
                return {"status": "Failed", "error": "GROK_API_KEY not found in .env"}
            
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            data = {
                "model": "grok-beta",  # Updated model name
                "messages": [
                    {
                        "role": "system",
                        "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."
                    },
                    {
                        "role": "user",
                        "content": "Hello, how are you?"
                    }
                ],
                "stream": False,
                "temperature": 0.7
            }
            response = requests.post(
                'https://api.x.ai/v1/chat/completions',
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return {"status": "Success", "response": response.json()}
        except Exception as e:
            return {"status": "Failed", "error": str(e)}
 
    def test_serpapi(self):
        print("\nTesting SerpAPI...")
        try:
            api_key = os.getenv('SERPAPI_API_KEY')
            if not api_key:
                return {"status": "Failed", "error": "SERPAPI_API_KEY not found in .env"}
            
            params = {
                'api_key': api_key,
                'q': 'test query',
                'engine': 'google'
            }
            response = requests.get('https://serpapi.com/search', params=params)
            response.raise_for_status()
            return {"status": "Success", "response": response.json()}
        except Exception as e:
            return {"status": "Failed", "error": str(e)}

    def test_alphavantage(self):
        print("\nTesting Alpha Vantage API...")
        try:
            api_key = os.getenv('ALPHAVANTAGE_API_KEY')
            if not api_key:
                return {"status": "Failed", "error": "ALPHAVANTAGE_API_KEY not found in .env"}
            
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': 'IBM',
                'apikey': api_key
            }
            response = requests.get('https://www.alphavantage.co/query', params=params)
            response.raise_for_status()
            return {"status": "Success", "response": response.json()}
        except Exception as e:
            return {"status": "Failed", "error": str(e)}

    def test_google_drive(self):
        print("\nTesting Google Drive API...")
        try:
            SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']
            creds = None
            token_path = 'token.json'
            credentials_path = 'credentials.json'

            if os.path.exists(token_path):
                creds = Credentials.from_authorized_user_file(token_path, SCOPES)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not os.path.exists(credentials_path):
                        return {"status": "Failed", "error": "credentials.json not found"}
                    flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
                    creds = flow.run_local_server(port=0)
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())

            service = build('drive', 'v3', credentials=creds)
            results = service.files().list(pageSize=10, fields="nextPageToken, files(id, name)").execute()
            return {"status": "Success", "response": str(results.get('files', []))}
        except Exception as e:
            return {"status": "Failed", "error": str(e)}

    def test_ollama(self):
        print("\nTesting Local Ollama...")
        try:
            url = "http://localhost:11434/api/generate"
            data = {
                "model": "llama2",
                "prompt": "Hello, how are you?",
                "stream": False
            }
            response = requests.post(url, json=data)
            response.raise_for_status()
            
            if response.text:
                return {"status": "Success", "response": response.json()}
            return {"status": "Failed", "error": "Empty response from Ollama"}
        except requests.exceptions.ConnectionError:
            return {"status": "Failed", "error": "Cannot connect to Ollama. Is 'ollama serve' running?"}
        except json.JSONDecodeError:
            return {"status": "Failed", "error": "Invalid JSON response from Ollama"}
        except Exception as e:
            return {"status": "Failed", "error": str(e)}

    def run_all_tests(self):
        # Dictionary of test methods
        tests = {
            'OpenAI': self.test_openai,
            'Anthropic': self.test_anthropic,
            'Perplexity': self.test_perplexity,
            'Gemini': self.test_gemini,
            'Grok': self.test_grok,
            'SerpAPI': self.test_serpapi,
            'AlphaVantage': self.test_alphavantage,
            'Google Drive': self.test_google_drive,
            'Ollama': self.test_ollama
        }

        # Run each test
        for api_name, test_method in tests.items():
            self.results[api_name] = test_method()

        # Print results
        print("\n=== Test Results ===")
        for api, result in self.results.items():
            status = "✅" if result['status'] == "Success" else "❌"
            print(f"\n{status} {api}: {result['status']}")
            if result['status'] == "Failed":
                print(f"   Error: {result['error']}")

        # Save results to file
        filename = f'api_test_results_{self.test_time}.json'
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed results saved to {filename}")

def main():
    print("Starting API Tests...")
    print("\nRequired Environment Variables:")
    print("- OPENAI_API_KEY")
    print("- ANTHROPIC_API_KEY")
    print("- PERPLEXITY_API_KEY")
    print("- GEMINI_API_KEY")
    print("- GROK_API_KEY")
    print("- SERPAPI_API_KEY")
    print("- ALPHAVANTAGE_API_KEY")
    print("\nRequired Files:")
    print("- credentials.json (for Google Drive)")
    print("- token.json (will be created for Google Drive)")
    print("\nRequired Services:")
    print("- Ollama running locally ('ollama serve')")
    
    input("\nPress Enter to continue...")
    
    tester = APITester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()