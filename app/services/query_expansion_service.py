import asyncio
import logging
import json
from typing import List, Dict, Any, Optional
import aiohttp
from app.config import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


class QueryExpansionService:
    """Service for expanding queries using LLMs."""
    
    def __init__(self):
        self.session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def expand_query(
        self, 
        query: str, 
        llm_config: LLMConfig, 
        strategy: str = "fusion", 
        num_queries: int = 3
    ) -> List[str]:
        """
        Expand a query using the specified LLM and strategy.
        
        Args:
            query: Original query to expand
            llm_config: LLM configuration to use
            strategy: Expansion strategy ("fusion" or "multi_query")
            num_queries: Number of expanded queries to generate
            
        Returns:
            List of expanded queries including the original
        """
        queries, _ = await self.expand_query_with_metadata(query, llm_config, strategy, num_queries)
        return queries
    
    async def expand_query_with_metadata(
        self, 
        query: str, 
        llm_config: LLMConfig, 
        strategy: str = "fusion", 
        num_queries: int = 3
    ) -> tuple[List[str], Dict[str, Any]]:
        """
        Expand a query using the specified LLM and strategy, returning both queries and metadata.
        
        Args:
            query: Original query to expand
            llm_config: LLM configuration to use
            strategy: Expansion strategy ("fusion" or "multi_query")
            num_queries: Number of expanded queries to generate
            
        Returns:
            Tuple of (expanded queries list, metadata dict)
        """
        import time
        start_time = time.time()
        
        logger.debug(f"[QueryExpansion] Request: query='{query}', strategy='{strategy}', num_queries={num_queries}, llm_config.name='{getattr(llm_config, 'name', None)}', provider='{getattr(llm_config, 'provider', None)}'")
        
        metadata = {
            "original_query": query,
            "strategy": strategy,
            "llm_config_name": getattr(llm_config, 'name', None),
            "llm_provider": getattr(llm_config, 'provider', None),
            "requested_num_queries": num_queries,
            "expansion_successful": False,
            "error_message": None,
            "expanded_queries": [],
            "processing_time_seconds": 0.0
        }
        
        try:
            if strategy == "fusion":
                logger.debug("[QueryExpansion] Using fusion strategy")
                expanded_queries = await self._fusion_expansion(query, llm_config, num_queries)
            elif strategy == "multi_query":
                logger.debug("[QueryExpansion] Using multi_query strategy")
                expanded_queries = await self._multi_query_expansion(query, llm_config, num_queries)
            else:
                logger.warning(f"Unknown expansion strategy: {strategy}. Using original query only.")
                metadata["error_message"] = f"Unknown expansion strategy: {strategy}"
                metadata["processing_time_seconds"] = time.time() - start_time
                return [query], metadata
            
            # Always include the original query
            if query not in expanded_queries:
                expanded_queries.insert(0, query)
            
            # Update metadata with successful expansion
            metadata["expansion_successful"] = True
            metadata["expanded_queries"] = expanded_queries
            metadata["actual_num_queries"] = len(expanded_queries)
            metadata["processing_time_seconds"] = time.time() - start_time
            
            logger.info(f"Expanded query '{query}' into {len(expanded_queries)} queries using {strategy} strategy")
            logger.debug(f"[QueryExpansion] Expanded queries: {expanded_queries}")
            return expanded_queries, metadata
            
        except Exception as e:
            logger.error(f"Error expanding query: {str(e)}")
            logger.debug(f"[QueryExpansion] Exception details:", exc_info=True)
            
            # Update metadata with error information
            metadata["error_message"] = str(e)
            metadata["processing_time_seconds"] = time.time() - start_time
            
            # Return original query if expansion fails
            return [query], metadata
    
    async def _fusion_expansion(self, query: str, llm_config: LLMConfig, num_queries: int) -> List[str]:
        """
        Generate query variations for fusion strategy.
        Creates semantically similar queries with different phrasings.
        """
        system_prompt = llm_config.system_prompt or """You are a helpful assistant that generates query variations for information retrieval. 
Generate semantically similar queries with different phrasings, synonyms, and perspectives that would retrieve the same information.
Focus on maintaining the core intent while varying the language and structure."""
        
        user_prompt = f"""Generate {num_queries} different variations of this query that maintain the same meaning but use different words, phrasings, or perspectives:

Original query: "{query}"

Return only the variations as a JSON array of strings, without any additional text or explanations.
Example format: ["variation 1", "variation 2", "variation 3"]"""
        
        return await self._call_llm(llm_config, system_prompt, user_prompt, num_queries)
    
    async def _multi_query_expansion(self, query: str, llm_config: LLMConfig, num_queries: int) -> List[str]:
        """
        Generate related queries for multi-query strategy.
        Creates queries that explore different aspects of the topic.
        """
        system_prompt = llm_config.system_prompt or """You are a helpful assistant that generates related queries for comprehensive information retrieval.
Generate queries that explore different aspects, subtopics, or related concepts that would provide comprehensive coverage of the topic.
Each query should be distinct and explore a different angle or facet of the subject."""
        
        user_prompt = f"""Generate {num_queries} related queries that explore different aspects or subtopics of this main query:

Main query: "{query}"

Each query should be distinct and explore a different angle, subtopic, or related concept.
Return only the queries as a JSON array of strings, without any additional text or explanations.
Example format: ["related query 1", "related query 2", "related query 3"]"""
        
        return await self._call_llm(llm_config, system_prompt, user_prompt, num_queries)
    
    async def _call_llm(self, llm_config: LLMConfig, system_prompt: str, user_prompt: str, num_queries: int) -> List[str]:
        """
        Call the LLM API to generate expanded queries.
        """
        session = await self._get_session()
        
        try:
            if llm_config.provider == LLMProvider.GROQ:
                return await self._call_groq(session, llm_config, system_prompt, user_prompt, num_queries)
            elif llm_config.provider == LLMProvider.OPENAI_COMPATIBLE:
                return await self._call_openai_compatible(session, llm_config, system_prompt, user_prompt, num_queries)
            elif llm_config.provider == LLMProvider.TRITON:
                return await self._call_triton(session, llm_config, system_prompt, user_prompt, num_queries)
            else:
                raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")
                
        except Exception as e:
            logger.error(f"Error calling LLM API: {str(e)}")
            raise
    
    async def _call_groq(self, session: aiohttp.ClientSession, llm_config: LLMConfig, system_prompt: str, user_prompt: str, num_queries: int) -> List[str]:
        """Call Groq API for query expansion."""
        headers = {
            "Authorization": f"Bearer {llm_config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "model": llm_config.model,
            "temperature": llm_config.temperature,
            "max_tokens": llm_config.max_tokens,
            "top_p": llm_config.top_p
        }
        
        if llm_config.top_k:
            payload["top_k"] = llm_config.top_k
        
        timeout = aiohttp.ClientTimeout(total=llm_config.timeout)
        
        async with session.post(llm_config.endpoint, json=payload, headers=headers, timeout=timeout) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Groq API error {response.status}: {error_text}")
            
            result = await response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            return self._parse_json_response(content, num_queries)
    
    async def _call_openai_compatible(self, session: aiohttp.ClientSession, llm_config: LLMConfig, system_prompt: str, user_prompt: str, num_queries: int) -> List[str]:
        """Call OpenAI-compatible API for query expansion."""
        headers = {
            "Content-Type": "application/json"
        }
        
        if llm_config.api_key:
            headers["Authorization"] = f"Bearer {llm_config.api_key}"
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "model": llm_config.model,
            "temperature": llm_config.temperature,
            "max_tokens": llm_config.max_tokens,
            "top_p": llm_config.top_p
        }
        
        # Add top_k if specified - NVIDIA API requires it in nvext object
        if llm_config.top_k is not None:
            payload["nvext"] = {"top_k": llm_config.top_k}
        
        timeout = aiohttp.ClientTimeout(total=llm_config.timeout)
        
        async with session.post(llm_config.endpoint, json=payload, headers=headers, timeout=timeout) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"OpenAI-compatible API error {response.status}: {error_text}")
            
            result = await response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            return self._parse_json_response(content, num_queries)
    
    async def _call_triton(self, session: aiohttp.ClientSession, llm_config: LLMConfig, system_prompt: str, user_prompt: str, num_queries: int) -> List[str]:
        """Call Triton inference server for query expansion."""
        headers = {
            "Content-Type": "application/json"
        }
        
        # Triton inference server payload format
        payload = {
            "inputs": [
                {
                    "name": "text_input",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"]
                }
            ],
            "parameters": {
                "temperature": llm_config.temperature,
                "max_tokens": llm_config.max_tokens,
                "top_p": llm_config.top_p
            }
        }
        
        if llm_config.top_k:
            payload["parameters"]["top_k"] = llm_config.top_k
        
        timeout = aiohttp.ClientTimeout(total=llm_config.timeout)
        
        async with session.post(f"{llm_config.endpoint}/v2/models/{llm_config.model}/infer", json=payload, headers=headers, timeout=timeout) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Triton API error {response.status}: {error_text}")
            
            result = await response.json()
            content = result["outputs"][0]["data"][0].strip()
            
            return self._parse_json_response(content, num_queries)
    
    def _parse_json_response(self, content: str, expected_count: int) -> List[str]:
        """
        Parse JSON response from LLM and extract queries.
        """
        try:
            # Try to parse as JSON array
            queries = json.loads(content)
            if isinstance(queries, list):
                # Filter out empty strings and ensure we have strings
                valid_queries = [str(q).strip() for q in queries if str(q).strip()]
                return valid_queries[:expected_count]  # Limit to expected count
            else:
                logger.warning(f"Expected JSON array, got: {type(queries)}")
                return []
                
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract queries from text
            logger.warning("Failed to parse JSON response, attempting text extraction")
            return self._extract_queries_from_text(content, expected_count)
    
    def _extract_queries_from_text(self, content: str, expected_count: int) -> List[str]:
        """
        Extract queries from text response when JSON parsing fails.
        """
        lines = content.split('\n')
        queries = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                # Remove common prefixes like "1.", "-", "*", etc.
                cleaned = line.lstrip('0123456789.-* ')
                if cleaned and len(cleaned) > 5:  # Minimum query length
                    queries.append(cleaned)
                    if len(queries) >= expected_count:
                        break
        
        return queries[:expected_count]
