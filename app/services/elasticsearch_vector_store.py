"""
Elasticsearch Vector Store implementation for the DSP AI RAG2 project.

This module provides an Elasticsearch-based vector store that implements the BaseVectorStore interface.
It supports three types of search:
1. Fulltext search - Traditional BM25-based text search
2. Vector search - Embedding-based similarity search using custom embeddings
3. Semantic search - Elasticsearch's built-in semantic search with ELSER model

Features:
- Document storage and retrieval using Elasticsearch
- Multiple search strategies (fulltext, vector, semantic)
- MongoDB-style metadata filtering
- Support for both username/password and API key authentication
- Automatic index management with appropriate mappings
- Hybrid search combining multiple search strategies
"""

import json
import logging
import math
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import ElasticsearchStore
from langchain.embeddings.base import Embeddings
from elasticsearch import Elasticsearch

from app.config import VectorStoreConfig
from app.services.base_vector_store import BaseVectorStore
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class ElasticsearchSearchType(str, Enum):
    """Elasticsearch search strategy types."""
    FULLTEXT = "fulltext"  # Traditional BM25-based text search
    VECTOR = "vector"      # Embedding-based similarity search
    SEMANTIC = "semantic"  # Elasticsearch built-in semantic search with ELSER
    HYBRID = "hybrid"      # Combination of multiple search types
    QUERY_DSL = "query_dsl"  # Custom Elasticsearch Query DSL template


class EmbeddingServiceWrapper(Embeddings):
    """Wrapper to make EmbeddingService compatible with LangChain Embeddings interface."""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self.embedding_service.embed_texts(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embedding_service.embed_query(text)


class ElasticsearchVectorStore(BaseVectorStore):
    """Enhanced Elasticsearch vector store with support for fulltext, vector, and semantic search."""
    
    def __init__(self, config: VectorStoreConfig, embedding_service: EmbeddingService):
        """
        Initialize Elasticsearch vector store with multi-search capabilities.
        
        Args:
            config: Vector store configuration containing Elasticsearch connection parameters
            embedding_service: Service for generating embeddings
        """
        super().__init__(config)
        self.embedding_service = embedding_service
        self.config = config
        
        # Set default search type from configuration
        self.default_search_type = self._get_default_search_type()
        
        # Create Elasticsearch client with authentication
        self.es_client = self._create_elasticsearch_client()
        
        # Initialize LangChain ElasticsearchStore for vector search
        try:
            # Create LangChain-compatible embeddings wrapper
            langchain_embeddings = EmbeddingServiceWrapper(embedding_service)
            
            self.store = ElasticsearchStore(
                index_name=config.es_index_name,
                es_connection=self.es_client,
                embedding=langchain_embeddings,
                strategy=ElasticsearchStore.ApproxRetrievalStrategy()
            )
            
            # Set up index mappings for multi-search support
            self._setup_index_mappings()
            
            logger.info(f"Initialized Elasticsearch multi-search store with index: {config.es_index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch vector store: {str(e)}")
            raise
    
    def _create_elasticsearch_client(self) -> Elasticsearch:
        """
        Create Elasticsearch client with appropriate authentication.
        
        Returns:
            Configured Elasticsearch client
        """
        es_config = {
            'hosts': [self.config.es_url],
            'verify_certs': True,
            'ssl_show_warn': False,
            # Fix version compatibility issue
            'headers': {
                'Accept': 'application/vnd.elasticsearch+json; compatible-with=8'
            }
        }
        
        # Configure authentication
        if self.config.es_api_key:
            # Use API key authentication
            es_config['api_key'] = self.config.es_api_key
            logger.info("Using API key authentication for Elasticsearch")
            
        elif self.config.es_user and self.config.es_password:
            # Use username/password authentication
            es_config['basic_auth'] = (self.config.es_user, self.config.es_password)
            logger.info(f"Using username/password authentication for Elasticsearch (user: {self.config.es_user})")
            
        else:
            # No authentication
            logger.info("Using no authentication for Elasticsearch")
        
        try:
            client = Elasticsearch(**es_config)
            # Test connection
            client.info()
            logger.info(f"Successfully connected to Elasticsearch at {self.config.es_url}")
            return client
            
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {str(e)}")
            raise
    
    def _setup_index_mappings(self):
        """Set up index mappings to support fulltext, vector, and semantic search."""
        try:
            index_name = self.config.es_index_name
            
            # Check if index exists
            if not self.es_client.indices.exists(index=index_name):
                # Create index with mappings for all search types
                mapping = {
                    "mappings": {
                        "properties": {
                            self.config.es_fulltext_field: {
                                "type": "text",
                                "analyzer": "standard"
                            },
                            "vector": {
                                "type": "dense_vector",
                                "dims": self.embedding_service.get_dimension()
                            },
                            self.config.es_semantic_field: {
                                "type": "semantic_text",
                                "inference_id": self.config.es_semantic_inference_id
                            },
                            "metadata": {
                                "type": "object",
                                "dynamic": True
                            }
                        }
                    },
                    "settings": {
                        "index": {
                            "number_of_shards": 1,
                            "number_of_replicas": 0
                        }
                    }
                }
                
                self.es_client.indices.create(index=index_name, body=mapping)
                logger.info(f"Created Elasticsearch index '{index_name}' with multi-search mappings")
            else:
                logger.info(f"Elasticsearch index '{index_name}' already exists")
                
        except Exception as e:
            logger.warning(f"Could not set up index mappings: {str(e)}. Index may already exist.")
    
    def _get_default_search_type(self) -> ElasticsearchSearchType:
        """Get the default search type from configuration."""
        try:
            search_type_str = getattr(self.config, 'es_search_type', 'vector')
            
            # Map string to enum
            search_type_map = {
                'fulltext': ElasticsearchSearchType.FULLTEXT,
                'vector': ElasticsearchSearchType.VECTOR,
                'semantic': ElasticsearchSearchType.SEMANTIC,
                'hybrid': ElasticsearchSearchType.HYBRID,
                'query_dsl': ElasticsearchSearchType.QUERY_DSL
            }
            
            default_type = search_type_map.get(search_type_str.lower(), ElasticsearchSearchType.VECTOR)
            logger.info(f"Using default search type: {default_type.value}")
            return default_type
            
        except Exception as e:
            logger.warning(f"Could not determine default search type: {str(e)}. Using vector search.")
            return ElasticsearchSearchType.VECTOR
    
    def _normalize_scores(self, results: List[Tuple[LangchainDocument, float]], search_type: ElasticsearchSearchType) -> List[Tuple[LangchainDocument, float]]:
        """
        Normalize similarity scores to 0-1 range based on search type.
        
        Args:
            results: List of (document, score) tuples
            search_type: Type of search that generated these scores
            
        Returns:
            List of (document, normalized_score) tuples with scores in 0-1 range
        """
        if not results:
            return results
        
        if search_type == ElasticsearchSearchType.VECTOR:
            # Vector search typically already returns cosine similarity (0-1 range)
            # But ensure scores are clamped to 0-1 range
            return [(doc, max(0.0, min(1.0, score))) for doc, score in results]
        
        elif search_type == ElasticsearchSearchType.FULLTEXT:
            # BM25 scores can be much higher than 1, use min-max normalization
            scores = [score for _, score in results]
            if len(scores) == 1:
                # Single result, normalize to 1.0 if positive, 0.0 if zero/negative
                normalized_score = 1.0 if scores[0] > 0 else 0.0
                return [(results[0][0], normalized_score)]
            
            min_score = min(scores)
            max_score = max(scores)
            
            if max_score == min_score:
                # All scores are the same, assign 1.0 to all
                return [(doc, 1.0) for doc, _ in results]
            
            # Min-max normalization to 0-1 range
            normalized_results = []
            for doc, score in results:
                normalized_score = (score - min_score) / (max_score - min_score)
                normalized_results.append((doc, normalized_score))
            
            return normalized_results
        
        elif search_type == ElasticsearchSearchType.SEMANTIC:
            # ELSER scores can vary, use sigmoid normalization for better distribution
            normalized_results = []
            for doc, score in results:
                # Apply sigmoid function: 1 / (1 + e^(-x))
                # This maps any real number to (0, 1) range
                normalized_score = 1.0 / (1.0 + math.exp(-score))
                normalized_results.append((doc, normalized_score))
            
            return normalized_results
        
        elif search_type == ElasticsearchSearchType.HYBRID:
            # Hybrid scores are already weighted combinations, use min-max normalization
            scores = [score for _, score in results]
            if len(scores) == 1:
                normalized_score = 1.0 if scores[0] > 0 else 0.0
                return [(results[0][0], normalized_score)]
            
            min_score = min(scores)
            max_score = max(scores)
            
            if max_score == min_score:
                return [(doc, 1.0) for doc, _ in results]
            
            normalized_results = []
            for doc, score in results:
                normalized_score = (score - min_score) / (max_score - min_score)
                normalized_results.append((doc, normalized_score))
            
            return normalized_results
        
        else:
            # Default: clamp to 0-1 range
            return [(doc, max(0.0, min(1.0, score))) for doc, score in results]

    def _extract_document_content(self, hit_source: Dict[str, Any]) -> str:
        """
        Extract document content from Elasticsearch hit source with field fallback.
        
        Checks multiple possible field names to ensure consistent output regardless
        of how the document was indexed.
        
        Args:
            hit_source: The _source field from an Elasticsearch hit
            
        Returns:
            Document content string
        """
        # List of possible field names in order of preference
        content_fields = ['content', 'text', 'page_content', 'body', 'document']
        
        for field in content_fields:
            if field in hit_source and hit_source[field]:
                return hit_source[field]
        
        # If no content field found, return empty string
        return ''

    def add_documents(self, documents: List[LangchainDocument]) -> List[str]:
        """
        Add documents to the Elasticsearch vector store.
        
        Args:
            documents: List of LangChain documents to add
            
        Returns:
            List of document IDs that were added
        """
        try:
            logger.info(f"Adding {len(documents)} documents to Elasticsearch index: {self.config.es_index_name}")
            
            # Use LangChain's add_documents method which returns document IDs
            document_ids = self.store.add_documents(documents)
            
            logger.info(f"Successfully added {len(document_ids)} documents to Elasticsearch")
            return document_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to Elasticsearch: {str(e)}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        similarity_threshold: float = None, 
        filter: Optional[Dict[str, Any]] = None,
        search_type: Optional[ElasticsearchSearchType] = None
    ) -> List[Tuple[LangchainDocument, float]]:
        """
        Perform multi-type search in the Elasticsearch vector store.
        
        Args:
            query: Search query string
            k: Number of results to return
            similarity_threshold: Minimum similarity score (optional)
            filter: MongoDB-style metadata filter (optional)
            search_type: Type of search to perform (fulltext, vector, semantic, hybrid)
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        try:
            # Use configured default search type if none specified
            if search_type is None:
                search_type = self.default_search_type
                
            logger.debug(f"Performing {search_type} search for query: '{query}' with k={k}")
            
            # Route to appropriate search method based on search type
            if search_type == ElasticsearchSearchType.FULLTEXT:
                results = self._fulltext_search(query, k, filter)
            elif search_type == ElasticsearchSearchType.VECTOR:
                results = self._vector_search(query, k, filter)
            elif search_type == ElasticsearchSearchType.SEMANTIC:
                results = self._semantic_search(query, k, filter)
            elif search_type == ElasticsearchSearchType.HYBRID:
                results = self._hybrid_search(query, k, filter)
            elif search_type == ElasticsearchSearchType.QUERY_DSL:
                results = self._query_dsl_search(query, k, filter)
            else:
                # Default to vector search
                results = self._vector_search(query, k, filter)
            
            # Apply similarity threshold
            if similarity_threshold is not None:
                results = [(doc, score) for doc, score in results if score >= similarity_threshold]
            
            # Conditionally normalize scores to 0-1 range based on config
            if self.config.normalize_similarity_scores:
                results = self._normalize_scores(results, search_type)
            
            logger.debug(f"{search_type} search returned {len(results)} results. After applying similarity threshold {similarity_threshold}")
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error performing {search_type} search: {str(e)}")
            raise
    
    def _convert_filter_to_elasticsearch(self, mongo_filter: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert MongoDB-style filter to Elasticsearch query format.
        
        This is a basic conversion for simple equality filters.
        Complex MongoDB operators will be handled by post-filtering.
        
        Args:
            mongo_filter: MongoDB-style filter
            
        Returns:
            Elasticsearch filter dict or None
        """
        if not mongo_filter:
            return None
        
        # For simple equality filters, convert to Elasticsearch term queries
        es_filter = {"bool": {"must": []}}
        
        for key, value in mongo_filter.items():
            if isinstance(value, str) or isinstance(value, (int, float, bool)):
                # Simple equality
                es_filter["bool"]["must"].append({"term": {f"metadata.{key}": value}})
            elif isinstance(value, dict) and "$eq" in value:
                # MongoDB $eq operator
                es_filter["bool"]["must"].append({"term": {f"metadata.{key}": value["$eq"]}})
        
        return es_filter if es_filter["bool"]["must"] else None
    
    def _fulltext_search(self, query: str, k: int, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[LangchainDocument, float]]:
        """
        Perform fulltext search using BM25 algorithm.
        
        Args:
            query: Search query string
            k: Number of results to return
            filter: MongoDB-style metadata filter (optional)
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        try:
            # Build Elasticsearch query for fulltext search
            es_query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    self.config.es_fulltext_field: {
                                        "query": query,
                                        "operator": "or"
                                    }
                                }
                            }
                        ]
                    }
                }
            }
            
            # Add filter if provided
            if filter:
                es_filter = self._convert_filter_to_elasticsearch(filter)
                if es_filter:
                    es_query["query"]["bool"]["filter"] = [es_filter]
            print(f"***** Performing fulltext search for query: '{es_query}' with field={self.config.es_fulltext_field}")
            # Execute search
            response = self.es_client.search(
                index=self.config.es_index_name,
                body=es_query,
                size=k
            )
            
            # Process results
            results = []
            for hit in response['hits']['hits']:
                doc = LangchainDocument(
                    page_content=self._extract_document_content(hit['_source']),
                    metadata=hit['_source'].get('metadata', {})
                )
                score = hit['_score']
                results.append((doc, score))
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing fulltext search: {str(e)}")
            return []
    
    def _vector_search(self, query: str, k: int, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[LangchainDocument, float]]:
        """
        Perform vector similarity search using embeddings.
        
        Args:
            query: Search query string
            k: Number of results to return
            filter: MongoDB-style metadata filter (optional)
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        try:
            # Use LangChain's vector search for consistency
            es_filter = self._convert_filter_to_elasticsearch(filter) if filter else None
            
            results = self.store.similarity_search_with_score(
                query=query,
                k=k * 2 if filter else k,  # Get more results if filtering
                filter=es_filter
            )
            
            # Apply additional MongoDB-style filtering if needed
            filtered_results = []
            for doc, score in results:
                if filter and not self._matches_filter(doc.metadata, filter):
                    continue
                filtered_results.append((doc, score))
                if len(filtered_results) >= k:
                    break
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error performing vector search: {str(e)}")
            return []
    
    def _semantic_search(self, query: str, k: int, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[LangchainDocument, float]]:
        """
        Perform semantic search using Elasticsearch's built-in semantic search (ELSER).
        
        Args:
            query: Search query string
            k: Number of results to return
            filter: MongoDB-style metadata filter (optional)
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        try:
            # Build Elasticsearch semantic search query using modern syntax
            # Use semantic query with content_vector field
            es_query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "semantic": {
                                    "field": self.config.es_semantic_field,
                                    "query": query,
                                    "boost": 1.0
                                }
                            }
                        ]
                    }
                }
            }
            
            # Add filter if provided
            if filter:
                es_filter = self._convert_filter_to_elasticsearch(filter)
                if es_filter:
                    es_query["query"]["bool"]["filter"] = [es_filter]
            
            # Execute search
            response = self.es_client.search(
                index=self.config.es_index_name,
                body=es_query,
                size=k
            )
            
            # Process results
            results = []
            for hit in response['hits']['hits']:
                doc = LangchainDocument(
                    page_content=self._extract_document_content(hit['_source']),
                    metadata=hit['_source'].get('metadata', {})
                )
                score = hit['_score']
                results.append((doc, score))
            
            return results
            
        except Exception as e:
            logger.warning(f"Semantic search not available or configured: {str(e)}")
            # Fallback to fulltext search if semantic search is not available
            return self._fulltext_search(query, k, filter)
    
    def _query_dsl_search(self, query: str, k: int, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[LangchainDocument, float]]:
        """
        Perform search using custom Elasticsearch Query DSL template.
        
        Args:
            query: Search query string
            k: Number of results to return
            filter: MongoDB-style metadata filter (optional)
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        try:
            if not self.config.es_query_dsl_template:
                logger.error("Query DSL template not configured - falling back to fulltext search")
                return self._fulltext_search(query, k, filter)
            
            # Deep copy the template to avoid modifying the original
            import copy
            es_query = copy.deepcopy(self.config.es_query_dsl_template)
            
            # Replace $QUERY$ placeholder with actual query string
            es_query_str = json.dumps(es_query)
            es_query_str = es_query_str.replace("$QUERY$", query)
            es_query = json.loads(es_query_str)

            # Add filter if provided (this is more complex for custom DSL)
            if filter:
                logger.warning("Metadata filtering with custom Query DSL is not fully supported - filter may be ignored")
                # For custom DSL, we can't easily inject filters without knowing the structure
                # Users should include filtering in their DSL template if needed
            
            logger.debug(f"Executing custom Query DSL: {json.dumps(es_query, indent=2)}")
            print(
                f"***** Performing fulltext search for query: '{es_query}'")

            # Execute search with custom DSL
            response = self.es_client.search(
                index=self.config.es_index_name,
                body=es_query,
                size=k
            )
            
            # Process results
            results = []
            for hit in response['hits']['hits']:
                doc = LangchainDocument(
                    page_content=self._extract_document_content(hit['_source']),
                    metadata=hit['_source'].get('metadata', {})
                )
                score = hit['_score']
                results.append((doc, score))
            
            logger.debug(f"Query DSL search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error performing Query DSL search: {str(e)}")
            # Fallback to fulltext search if Query DSL fails
            logger.info("Falling back to fulltext search")
            return self._fulltext_search(query, k, filter)
    
    def _hybrid_search(self, query: str, k: int, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[LangchainDocument, float]]:
        """
        Perform hybrid search combining fulltext, vector, and semantic search.
        
        Args:
            query: Search query string
            k: Number of results to return
            filter: MongoDB-style metadata filter (optional)
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        try:
            # Get results from all search types
            fulltext_results = self._fulltext_search(query, k, filter)
            vector_results = self._vector_search(query, k, filter)
            semantic_results = self._semantic_search(query, k, filter)
            
            # Combine and deduplicate results
            combined_results = {}
            
            # Add fulltext results (weight: 0.3)
            for doc, score in fulltext_results:
                doc_id = id(doc.page_content + str(doc.metadata))
                if doc_id not in combined_results:
                    combined_results[doc_id] = (doc, 0.0)
                combined_results[doc_id] = (combined_results[doc_id][0], 
                                          combined_results[doc_id][1] + score * 0.3)
            
            # Add vector results (weight: 0.4)
            for doc, score in vector_results:
                doc_id = id(doc.page_content + str(doc.metadata))
                if doc_id not in combined_results:
                    combined_results[doc_id] = (doc, 0.0)
                combined_results[doc_id] = (combined_results[doc_id][0], 
                                          combined_results[doc_id][1] + score * 0.4)
            
            # Add semantic results (weight: 0.3)
            for doc, score in semantic_results:
                doc_id = id(doc.page_content + str(doc.metadata))
                if doc_id not in combined_results:
                    combined_results[doc_id] = (doc, 0.0)
                combined_results[doc_id] = (combined_results[doc_id][0], 
                                          combined_results[doc_id][1] + score * 0.3)
            
            # Sort by combined score and return top k
            sorted_results = sorted(combined_results.values(), key=lambda x: x[1], reverse=True)
            return sorted_results[:k]
            
        except Exception as e:
            logger.error(f"Error performing hybrid search: {str(e)}")
            # Fallback to vector search
            return self._vector_search(query, k, filter)
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents from the Elasticsearch vector store.
        
        Args:
            document_ids: List of document IDs to delete
        """
        try:
            logger.info(f"Deleting {len(document_ids)} documents from Elasticsearch")
            
            # Delete documents by IDs
            self.store.delete(document_ids)
            
            logger.info(f"Successfully deleted {len(document_ids)} documents from Elasticsearch")
            
        except Exception as e:
            logger.error(f"Error deleting documents from Elasticsearch: {str(e)}")
            raise
    
    def get_document_count(self) -> int:
        """
        Get the total number of documents in the Elasticsearch vector store.
        
        Returns:
            Total number of documents
        """
        try:
            # Use the Elasticsearch client to get document count
            client = self.store.client
            result = client.count(index=self.config.es_index_name)
            count = result['count']
            
            logger.debug(f"Document count in Elasticsearch index '{self.config.es_index_name}': {count}")
            return count
            
        except Exception as e:
            logger.error(f"Error getting document count from Elasticsearch: {str(e)}")
            # Return 0 if index doesn't exist or other error
            return 0
