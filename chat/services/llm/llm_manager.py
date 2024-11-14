
import os  # For accessing environment variables
from typing import Dict, List, Optional  # For type hints
import openai  # OpenAI API client
from anthropic import Anthropic  # Anthropic (Claude) API client
from groq import AsyncGroq  # Groq API client
import json  # For handling JSON data
from datetime import datetime  # For timestamps
from ...utils.logging_config import setup_logging  # Our custom logging  # Note the ... for going up three levels
from ..search.web_search import WebSearchService  # Web search functionality
from ..storage.vector_store import VectorStore  # Vector database
from ..embeddings.embedding_service import EmbeddingService  # Embedding generation
from .external_tools.perplexity import Chat, Search  # Perplexity API tools  # Note the relative import

class EnhancedLLMManager:
    def __init__(self):
        # Set up logging with timestamp banner and separator line
        self.logger = setup_logging()
        self.logger.info(f"{'='*50}")  # Print a separator line
        self.logger.info(f"Starting new LLM session at {datetime.now()}")
        self.logger.info(f"{'='*50}")

        try:
            # Initialize OpenAI
            self.openai = openai
            self.openai.api_key = os.getenv('OPENAI_API_KEY')
            if not self.openai.api_key:
                self.logger.error("OpenAI API key not found")
                raise ValueError("OpenAI API key not found")
            self.logger.info("OpenAI initialized successfully")
            
            # Initialize Anthropic
            self.anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            if not os.getenv('ANTHROPIC_API_KEY'):
                self.logger.error("Anthropic API key not found")
                raise ValueError("Anthropic API key not found")
            self.logger.info("Anthropic initialized successfully")
            
            # Initialize Groq
            self.groq = AsyncGroq(api_key=os.getenv('GROQ_API_KEY'))
            if not os.getenv('GROQ_API_KEY'):
                self.logger.error("Groq API key not found")
                raise ValueError("Groq API key not found")
            self.logger.info("Groq initialized successfully")

            # Initialize Perplexity
            self.perplexity_chat = Chat(
                api_key=os.getenv('PERPLEXITY_API_KEY'),
                model="pplx-7b-online"
            )
            self.perplexity_search = Search(
                api_key=os.getenv('PERPLEXITY_API_KEY')
            )
            self.logger.info("Perplexity initialized successfully")

            # Initialize RAG components
            self.search_service = WebSearchService()
            self.logger.info("WebSearchService initialized")
            
            self.vector_store = VectorStore()
            self.logger.info("VectorStore initialized")
            
            self.embedding_service = EmbeddingService()
            self.logger.info("EmbeddingService initialized")

        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            raise

    def _log_api_call(self, api_name: str, request_data: dict, response_data: dict = None, error: str = None):
        """Helper method to log API calls"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'api_name': api_name,
            'request': request_data,   # What we sent to the API
            'response': response_data, # What we got back (if successful)
            'error': error             # Any error message (if failed)
        }
        self.logger.debug(f"API Call Details: {json.dumps(log_data, indent=2)}")

    async def get_response(self, llm_type: str, message: str, use_rag: bool = False) -> str:
        """Get response from specified LLM"""
        self.logger.info(f"Processing request for LLM type: {llm_type}")
        self.logger.debug(f"Input message: {message[:100]}..." if len(message) > 100 else message)
        self.logger.debug(f"RAG enabled: {use_rag}")

        try:
            # Check for empty message
            if not message.strip():
                self.logger.warning("Empty message received")
                return "Error: Empty message provided"

            # Get RAG context if enabled
            enhanced_message = message
            if use_rag:
                self.logger.info("Retrieving RAG context")
                try:
                    context = await self._get_rag_context(message)
                    if context:
                        enhanced_message = self._create_enhanced_prompt(message, context)
                        self.logger.debug(f"Enhanced prompt created: {enhanced_message[:100]}...")
                    else:
                        self.logger.warning("No RAG context found, using original message")
                except Exception as e:
                    self.logger.error(f"RAG enhancement failed: {str(e)}")
                    self.logger.warning("Falling back to original message")

            if llm_type == 'gpt':
                self.logger.info("Sending request to GPT-4")
                try:
                    # Prepare request data
                    request_data = {
                        'model': 'gpt-4',
                        'messages': [{"role": "user", "content": enhanced_message}]
                    }
                    
                    # Time the API call
                    start_time = datetime.now()
                    response = await self.openai.ChatCompletion.acreate(
                        **request_data
                    )
                    end_time = datetime.now()
                    
                    # Extract response text   
                    response_text = response.choices[0].message.content
                    
                    # Log API call details
                    self._log_api_call(
                        api_name='GPT-4',
                        request_data=request_data,
                        response_data={
                            'response_text': response_text[:100] + "..." if len(response_text) > 100 else response_text,
                            'processing_time': str(end_time - start_time)
                        }
                    )
                    
                    return response_text

                except Exception as e:
                    error_msg = f"GPT-4 API error: {str(e)}"
                    self.logger.error(error_msg)
                    self._log_api_call(
                        api_name='GPT-4',
                        request_data=request_data,
                        error=error_msg
                    )
                    return f"Error with GPT: {str(e)}"

            elif llm_type == 'claude':
                self.logger.info("Sending request to Claude")
                try:
                    request_data = {
                        'model': 'claude-2',
                        'messages': [{"role": "user", "content": enhanced_message}]
                    }
                    
                    start_time = datetime.now()
                    response = await self.anthropic.messages.create(
                        **request_data
                    )
                    end_time = datetime.now()
                    
                    response_text = response.content
                    
                    self._log_api_call(
                        api_name='Claude',
                        request_data=request_data,
                        response_data={
                            'response_text': response_text[:100] + "..." if len(response_text) > 100 else response_text,
                            'processing_time': str(end_time - start_time)
                        }
                    )
                    
                    return response_text

                except Exception as e:
                    error_msg = f"Claude API error: {str(e)}"
                    self.logger.error(error_msg)
                    self._log_api_call(
                        api_name='Claude',
                        request_data=request_data,
                        error=error_msg
                    )
                    return f"Error with Claude: {str(e)}"

            elif llm_type == 'perplexity':
                self.logger.info("Sending request to Perplexity")
                try:
                    # First try search for real-time information
                    start_time = datetime.now()
                    search_response = self.perplexity_search.search(message)
                    
                    # Enhance the prompt with search results
                    enhanced_prompt = f"""
                    Original question: {message}
                    
                    Search results: {search_response}
                    
                    Please provide a comprehensive answer based on the search results.
                    """
                    
                    # Get chat response with enhanced prompt
                    response = self.perplexity_chat.send_message(enhanced_prompt)
                    end_time = datetime.now()
                    
                    self._log_api_call(
                        api_name='Perplexity',
                        request_data={'message': message},
                        response_data={
                            'search_results': str(search_response)[:100] + "...",
                            'response': str(response)[:100] + "...",
                            'processing_time': str(end_time - start_time)
                        }
                    )
                    
                    return response

                except Exception as e:
                    error_msg = f"Perplexity API error: {str(e)}"
                    self.logger.error(error_msg)
                    self._log_api_call(
                        api_name='Perplexity',
                        request_data={'message': message},
                        error=error_msg
                    )
                    return f"Error with Perplexity: {str(e)}"

            else:
                error_msg = f"Unsupported LLM type: {llm_type}"
                self.logger.error(error_msg)
                return error_msg

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    async def _get_rag_context(self, query: str) -> Optional[str]:
        """Get relevant context using RAG pipeline"""
        self.logger.info("Getting RAG context")
        self.logger.debug(f"Query: {query[:100]}...")

        try:
            # Step 1: Search the web
            self.logger.debug("Performing web search")
            search_results = await self.search_service.search(query)
            if not search_results:
                self.logger.warning("No search results found")
                return None
            
            self.logger.debug(f"Found {len(search_results)} search results")
            
            # Step 2: Generate embeddings for search results
            self.logger.debug("Generating embeddings")
            texts = [result['snippet'] for result in search_results]
            embeddings = await self.embedding_service.get_embeddings(texts)
            
            if not embeddings:
                self.logger.warning("No embeddings generated")
                return None
            
            self.logger.debug(f"Generated {len(embeddings)} embeddings")
            
            # Step 3: Store in vector database
            self.logger.debug("Storing documents in vector database")
            await self.vector_store.store_documents(search_results, embeddings)
            
            # Step 4: Find similar documents
            self.logger.debug("Finding similar documents")
            query_embedding = (await self.embedding_service.get_embeddings([query]))[0]
            similar_docs = await self.vector_store.search_similar(query_embedding)
            
            if not similar_docs:
                self.logger.warning("No similar documents found")
                return None

            # Step 5: Format and return results    
            formatted_results = self.search_service.format_results(similar_docs)
            self.logger.debug(f"Returning formatted results: {formatted_results[:100]}...")
            return formatted_results

        except Exception as e:
            self.logger.error(f"Error in RAG context retrieval: {str(e)}")
            return None

    def _create_enhanced_prompt(self, query: str, context: str) -> str:
        """Create enhanced prompt with RAG context"""
        enhanced_prompt = f"""
        Original Question: {query}

        Relevant Context:
        {context}

        Please provide a comprehensive answer based on the context above. 
        If the context doesn't fully address the question, please indicate this 
        and provide the best possible answer based on available information.
        
        Format your response in a clear, structured manner with:
        1. Direct answer to the question
        2. Supporting details from the context
        3. Any important caveats or limitations
        """
        self.logger.debug(f"Created enhanced prompt: {enhanced_prompt[:100]}...")
        return enhanced_prompt

    async def validate_api_keys(self) -> Dict[str, bool]:
        """Validate all API keys are working"""
        self.logger.info("Starting API key validation")
        status = {}
        
        for api_name, test_func in [
            ('openai', self._test_openai),
            ('anthropic', self._test_anthropic),
            ('perplexity', self._test_perplexity),
            ('groq', self._test_groq)
        ]:
            try:
                self.logger.debug(f"Testing {api_name} API key")
                await test_func()
                status[api_name] = True
                self.logger.info(f"{api_name} API key is valid")
            except Exception as e:
                status[api_name] = False
                self.logger.error(f"{api_name} API key validation failed: {str(e)}")

        self.logger.info(f"API key validation completed. Status: {status}")
        return status

    async def _test_openai(self):
        await self.openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )

    async def _test_anthropic(self):
        await self.anthropic.messages.create(
            model="claude-2",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )

    async def _test_perplexity(self):
        self.perplexity_chat.send_message("test")

    async def _test_groq(self):
        await self.groq.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )