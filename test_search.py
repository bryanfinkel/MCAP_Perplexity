# tests/test_search.py
import asyncio
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from chat.services.search.web_search import WebSearchService

async def test_web_search():
    print("\nTesting Web Search Service...")
    
    # Initialize service
    search_service = WebSearchService()
    
    # Test search
    test_query = "What is the current stock price of Apple?"
    print(f"\nSearching for: {test_query}")
    
    results = await search_service.search(test_query)
    
    if results:
        print(f"\nFound {len(results)} results")
        print("\nFormatted results:")
        print(search_service.format_results(results))
    else:
        print("No results found")

if __name__ == "__main__":
    # Create directory for test files
    os.makedirs("tests", exist_ok=True)
    
    # Run the test
    asyncio.run(test_web_search())