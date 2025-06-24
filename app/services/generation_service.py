import logging
from typing import List, Dict, Any
import httpx
import json

from app.config import GenerationConfig, settings

logger = logging.getLogger(__name__)

class GroqGenerationService:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.api_key = settings.GROQ_API_KEY
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError("Groq API key not provided")

    async def generate_response(
        self, 
        query: str, 
        context_documents: List[Dict[str, Any]],
        system_prompt: str = None
    ) -> str:
        """Generate a response using Groq API."""
        try:
            # Prepare context
            context = self._prepare_context(context_documents)
            
            # Prepare messages
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            else:
                messages.append({
                    "role": "system", 
                    "content": "You are a helpful AI assistant. Answer questions based on the provided context. If the context doesn't contain enough information to answer the question, say so clearly."
                })
            
            user_message = f"""Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above."""
            
            messages.append({"role": "user", "content": user_message})
            
            # Prepare request
            payload = {
                "model": self.config.model.value,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
                "stream": False
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make request
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.base_url,
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Groq API: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Error from Groq API: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error generating response with Groq: {str(e)}")
            raise

    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved documents."""
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            filename = metadata.get('filename', 'Unknown')
            
            context_part = f"Document {i} (from {filename}):\n{content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)

    def generate_summary(self, text: str) -> str:
        """Generate a summary of the given text."""
        # This would be implemented similarly to generate_response
        # but with a different prompt focused on summarization
        pass

class GenerationServiceFactory:
    @staticmethod
    def create_service(config: GenerationConfig) -> GroqGenerationService:
        """Create a generation service based on configuration."""
        if config.model.value.startswith(("llama", "mixtral", "gemma")):
            return GroqGenerationService(config)
        else:
            raise ValueError(f"Unsupported generation model: {config.model}")
