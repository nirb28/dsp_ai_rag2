import logging
from typing import List, Dict, Any, Optional
import json

from app.config import ContextInjectionConfig

logger = logging.getLogger(__name__)

class ContextService:
    """Service for managing and injecting additional context into RAG queries."""
    
    def __init__(self, config: ContextInjectionConfig):
        self.config = config
    
    def prepare_context(self, context_items: List[Dict[str, Any]] = None) -> Optional[str]:
        """Prepare context string from context items and static context.
        
        Args:
            context_items: List of context items to inject, e.g., chat history messages.
                         Each item should be a dict with at least a 'content' field.
        
        Returns:
            Formatted context string or None if context injection is disabled.
        """
        if not self.config.enabled:
            return None
            
        context_parts = []
        
        # Add static context if provided
        if self.config.static_context:
            context_parts.append(self.config.static_context)
        
        # Add dynamic context items
        if context_items:
            # Limit number of context items
            items_to_use = context_items[:self.config.max_items]
            
            for item in items_to_use:
                if isinstance(item, dict) and 'content' in item:
                    # Truncate content if needed
                    content = item['content']
                    if len(content) > self.config.max_tokens_per_item * 4:  # Approximate chars to tokens
                        content = content[:self.config.max_tokens_per_item * 4] + "..."
                    
                    # Add any additional formatting based on item type
                    if 'role' in item:
                        content = f"{item['role']}: {content}"
                        
                    context_parts.append(content)
                elif isinstance(item, str):
                    # Handle simple string items
                    if len(item) > self.config.max_tokens_per_item * 4:
                        item = item[:self.config.max_tokens_per_item * 4] + "..."
                    context_parts.append(item)
        
        if not context_parts:
            return None
            
        # Join all context parts with the specified separator
        context_text = self.config.separator.join(context_parts)
        
        # Add prefix if specified
        if self.config.context_prefix:
            context_text = f"{self.config.context_prefix}{context_text}"
            
        return context_text
    
    def inject_context(self, query: str, context_items: List[Dict[str, Any]] = None) -> str:
        """Inject context into the query based on configuration.
        
        Args:
            query: Original user query
            context_items: List of context items to inject
            
        Returns:
            Query with injected context
        """
        if not self.config.enabled:
            return query
            
        context_text = self.prepare_context(context_items)
        if not context_text:
            return query
            
        # Inject context based on configured position
        position = self.config.position.lower()
        if position == "before_query":
            return f"{context_text}\n\nQuery: {query}"
        elif position == "after_query":
            return f"{query}\n\n{context_text}"
        else:
            logger.warning(f"Unknown context position: {position}. Defaulting to before_query.")
            return f"{context_text}\n\nQuery: {query}"
