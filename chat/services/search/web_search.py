# chat/services/search/web_search.py
from typing import List, Dict
from duckduckgo_search import AsyncDDGS
import logging

class WebSearchService:
    def __init__(self):
        self.logger = logging.getLogger('LLMApp.WebSearch')
        self.search_client = AsyncDDGS()
        self.logger.info("WebSearchService initialized")

    async def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Perform web search and return results.
        """
        self.logger.debug(f"Searching for: {query}")
        try:
            results = []
            async for r in self.search_client.text(query, max_results=max_results):
                results.append({
                    'title': r['title'],
                    'link': r['link'],
                    'snippet': r['body']
                })
            self.logger.info(f"Found {len(results)} results")
            return results
        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            return []

    def format_results(self, results: List[Dict]) -> str:
        """
        Format search results into a readable string.
        """
        formatted = []
        for r in results:
            formatted.append(f"""
            Title: {r['title']}
            Snippet: {r['snippet']}
            Source: {r['link']}
            """)
        return "\n".join(formatted)