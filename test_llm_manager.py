# test_llm_manager.py
import asyncio
import os
from dotenv import load_dotenv
from chat.services.llm.llm_manager import EnhancedLLMManager

async def test_llm_manager():
    # Initialize the LLM manager
    print("Initializing LLM Manager...")
    llm_manager = EnhancedLLMManager()
    
    # Test API key validation
    print("\nTesting API keys...")
    api_status = await llm_manager.validate_api_keys()
    print("API Status:", api_status)
    
    # Test simple query with each LLM
    test_message = "What is the current stock price of Apple?"
    
    for llm_type in ['gpt', 'claude', 'perplexity']:
        print(f"\nTesting {llm_type.upper()}...")
        try:
            # Test without RAG
            print(f"\nTesting {llm_type} without RAG:")
            response = await llm_manager.get_response(
                llm_type=llm_type,
                message=test_message,
                use_rag=False
            )
            print(f"{llm_type.upper()} Response:", response)
            
            # Test with RAG
            print(f"\nTesting {llm_type} with RAG:")
            response_with_rag = await llm_manager.get_response(
                llm_type=llm_type,
                message=test_message,
                use_rag=True
            )
            print(f"{llm_type.upper()} Response with RAG:", response_with_rag)
            
        except Exception as e:
            print(f"Error testing {llm_type}: {str(e)}")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run the test
    asyncio.run(test_llm_manager())