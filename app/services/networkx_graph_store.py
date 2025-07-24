import os
import json
import pickle
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import uuid
from collections import defaultdict
import re

import networkx as nx
from langchain.docstore.document import Document as LangchainDocument

from app.services.base_vector_store import BaseVectorStore
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class NetworkXGraphStore(BaseVectorStore):
    """
    NetworkX-based graph store that represents documents as nodes and relationships as edges.
    
    This implementation creates a knowledge graph where:
    - Documents are represented as nodes with their content and metadata
    - Relationships between documents are created based on shared entities, keywords, or topics
    - Similarity search is performed using graph algorithms like shortest path, centrality, etc.
    """
    
    def __init__(self, config: dict, embedding_service: Optional[EmbeddingService] = None):
        super().__init__(config)
        self.embedding_service = embedding_service
        self.graph = nx.Graph()
        self.document_nodes = {}  # Maps document IDs to node IDs
        self.node_documents = {}  # Maps node IDs to document data
        self.entity_index = defaultdict(set)  # Maps entities to document IDs
        self.keyword_index = defaultdict(set)  # Maps keywords to document IDs
        self._initialize_graph()
    
    def _initialize_graph(self):
        """Initialize or load the NetworkX graph."""
        try:
            graph_path = Path(self.config.get('index_path', './storage/graph'))
            graph_path.mkdir(parents=True, exist_ok=True)
            
            graph_file = graph_path / f"{self.config.get('name', 'default')}_graph.pkl"
            metadata_file = graph_path / f"{self.config.get('name', 'default')}_metadata.json"
            
            if graph_file.exists() and metadata_file.exists():
                # Load existing graph
                with open(graph_file, 'rb') as f:
                    self.graph = pickle.load(f)
                
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.document_nodes = metadata.get('document_nodes', {})
                    self.node_documents = metadata.get('node_documents', {})
                    self.entity_index = defaultdict(set, metadata.get('entity_index', {}))
                    self.keyword_index = defaultdict(set, metadata.get('keyword_index', {}))
                
                # Convert sets back from lists (JSON serialization limitation)
                for entity, doc_ids in self.entity_index.items():
                    self.entity_index[entity] = set(doc_ids)
                for keyword, doc_ids in self.keyword_index.items():
                    self.keyword_index[keyword] = set(doc_ids)
                
                logger.info(f"Loaded existing NetworkX graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            else:
                # Create new graph
                self.graph = nx.Graph()
                logger.info("Created new NetworkX graph")
                
        except Exception as e:
            logger.error(f"Error initializing NetworkX graph: {str(e)}")
            raise
    
    def _extract_entities_and_keywords(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Extract entities and keywords from text for graph relationships.
        This is a simple implementation - in production, you'd use NER and more sophisticated methods.
        """
        # Simple entity extraction (capitalized words, could be improved with NER)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Simple keyword extraction (remove common words, get significant terms)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [word for word in words if word not in stop_words]
        
        # Get most frequent keywords (simple approach)
        word_freq = defaultdict(int)
        for word in keywords:
            word_freq[word] += 1
        
        # Return top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        keywords = [word for word, freq in top_keywords if freq > 1]
        
        return entities[:5], keywords  # Limit entities to top 5
    
    def _create_document_relationships(self, doc_id: str, entities: List[str], keywords: List[str]):
        """Create relationships between documents based on shared entities and keywords."""
        
        # Find documents that share entities
        for entity in entities:
            related_docs = self.entity_index[entity]
            for related_doc_id in related_docs:
                if related_doc_id != doc_id and related_doc_id in self.document_nodes:
                    # Create edge between documents
                    node1 = self.document_nodes[doc_id]
                    node2 = self.document_nodes[related_doc_id]
                    
                    if not self.graph.has_edge(node1, node2):
                        self.graph.add_edge(node1, node2, 
                                          relationship_type='shared_entity',
                                          shared_entities=[entity],
                                          weight=1.0)
                    else:
                        # Strengthen existing relationship
                        edge_data = self.graph[node1][node2]
                        edge_data['shared_entities'] = list(set(edge_data.get('shared_entities', []) + [entity]))
                        edge_data['weight'] = edge_data.get('weight', 0) + 0.5
            
            self.entity_index[entity].add(doc_id)
        
        # Find documents that share keywords
        for keyword in keywords:
            related_docs = self.keyword_index[keyword]
            for related_doc_id in related_docs:
                if related_doc_id != doc_id and related_doc_id in self.document_nodes:
                    node1 = self.document_nodes[doc_id]
                    node2 = self.document_nodes[related_doc_id]
                    
                    if not self.graph.has_edge(node1, node2):
                        self.graph.add_edge(node1, node2,
                                          relationship_type='shared_keyword',
                                          shared_keywords=[keyword],
                                          weight=0.5)
                    else:
                        # Strengthen existing relationship
                        edge_data = self.graph[node1][node2]
                        edge_data['shared_keywords'] = list(set(edge_data.get('shared_keywords', []) + [keyword]))
                        edge_data['weight'] = edge_data.get('weight', 0) + 0.3
            
            self.keyword_index[keyword].add(doc_id)
    
    def add_documents(self, documents: List[LangchainDocument]) -> List[str]:
        """Add documents to the graph store."""
        try:
            if not documents:
                return []
            
            doc_ids = []
            
            for doc in documents:
                # Generate unique document ID
                doc_id = str(uuid.uuid4())
                doc_ids.append(doc_id)
                
                # Create node ID
                node_id = f"doc_{doc_id}"
                
                # Extract entities and keywords
                entities, keywords = self._extract_entities_and_keywords(doc.page_content)
                
                # Add document node to graph
                self.graph.add_node(node_id,
                                  node_type='document',
                                  content=doc.page_content,
                                  metadata=doc.metadata,
                                  entities=entities,
                                  keywords=keywords,
                                  doc_id=doc_id)
                
                # Store mappings
                self.document_nodes[doc_id] = node_id
                self.node_documents[node_id] = {
                    'doc_id': doc_id,
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'entities': entities,
                    'keywords': keywords
                }
                
                # Create relationships with existing documents
                self._create_document_relationships(doc_id, entities, keywords)
            
            # Save graph
            self._save_graph()
            
            logger.info(f"Added {len(documents)} documents to NetworkX graph. Graph now has {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to NetworkX graph: {str(e)}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        similarity_threshold: float = 0.3,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[LangchainDocument, float]]:
        """
        Search for similar documents using graph algorithms.
        
        This implementation uses multiple strategies:
        1. Direct entity/keyword matching
        2. Graph centrality measures
        3. Shortest path analysis
        4. Community detection
        """
        try:
            if self.graph.number_of_nodes() == 0:
                return []
            
            # Extract entities and keywords from query
            query_entities, query_keywords = self._extract_entities_and_keywords(query)
            
            # Strategy 1: Direct matching
            candidate_docs = set()
            
            # Find documents with matching entities
            for entity in query_entities:
                if entity in self.entity_index:
                    candidate_docs.update(self.entity_index[entity])
            
            # Find documents with matching keywords
            for keyword in query_keywords:
                if keyword in self.keyword_index:
                    candidate_docs.update(self.keyword_index[keyword])
            
            # Strategy 2: Graph-based scoring
            doc_scores = {}
            
            for doc_id in candidate_docs:
                if doc_id not in self.document_nodes:
                    continue
                    
                node_id = self.document_nodes[doc_id]
                node_data = self.node_documents[node_id]
                
                # Calculate similarity score based on multiple factors
                score = 0.0
                
                # Entity matching score
                entity_matches = len(set(query_entities) & set(node_data['entities']))
                if query_entities:
                    score += (entity_matches / len(query_entities)) * 0.4
                
                # Keyword matching score
                keyword_matches = len(set(query_keywords) & set(node_data['keywords']))
                if query_keywords:
                    score += (keyword_matches / len(query_keywords)) * 0.3
                
                # Graph centrality score (documents connected to many others are more important)
                try:
                    centrality = nx.degree_centrality(self.graph)[node_id]
                    score += centrality * 0.2
                except:
                    pass
                
                # Content length normalization (prefer substantial content)
                content_length = len(node_data['content'])
                if content_length > 100:
                    score += min(content_length / 1000, 0.1)
                
                doc_scores[doc_id] = score
            
            # Strategy 3: Expand search using graph traversal
            if len(candidate_docs) < k and candidate_docs:
                # Find neighbors of candidate documents
                neighbor_docs = set()
                for doc_id in list(candidate_docs)[:3]:  # Limit to top 3 to avoid explosion
                    if doc_id in self.document_nodes:
                        node_id = self.document_nodes[doc_id]
                        neighbors = list(self.graph.neighbors(node_id))
                        for neighbor_id in neighbors[:5]:  # Limit neighbors
                            if neighbor_id in self.node_documents:
                                neighbor_doc_id = self.node_documents[neighbor_id]['doc_id']
                                if neighbor_doc_id not in candidate_docs:
                                    neighbor_docs.add(neighbor_doc_id)
                                    # Score neighbors lower than direct matches
                                    doc_scores[neighbor_doc_id] = doc_scores.get(neighbor_doc_id, 0) + 0.1
                
                candidate_docs.update(neighbor_docs)
            
            # Apply metadata filtering if provided
            if filter_metadata:
                filtered_candidates = set()
                for doc_id in candidate_docs:
                    if doc_id in self.node_documents:
                        doc_metadata = self.node_documents[self.document_nodes[doc_id]]['metadata']
                        match = True
                        for key, value in filter_metadata.items():
                            if key not in doc_metadata or doc_metadata[key] != value:
                                match = False
                                break
                        if match:
                            filtered_candidates.add(doc_id)
                candidate_docs = filtered_candidates
            
            # Sort by score and return top k
            scored_docs = [(doc_id, doc_scores.get(doc_id, 0)) for doc_id in candidate_docs]
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Filter by similarity threshold
            scored_docs = [(doc_id, score) for doc_id, score in scored_docs if score >= similarity_threshold]
            
            # Convert to required format
            results = []
            for doc_id, score in scored_docs[:k]:
                if doc_id in self.document_nodes:
                    node_id = self.document_nodes[doc_id]
                    doc_data = self.node_documents[node_id]
                    
                    langchain_doc = LangchainDocument(
                        page_content=doc_data['content'],
                        metadata=doc_data['metadata']
                    )
                    results.append((langchain_doc, score))
            
            logger.info(f"NetworkX graph search returned {len(results)} results for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching NetworkX graph: {str(e)}")
            return []
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from the graph store."""
        try:
            for doc_id in document_ids:
                if doc_id in self.document_nodes:
                    node_id = self.document_nodes[doc_id]
                    
                    # Remove from indices
                    if node_id in self.node_documents:
                        doc_data = self.node_documents[node_id]
                        
                        # Remove from entity index
                        for entity in doc_data.get('entities', []):
                            if entity in self.entity_index:
                                self.entity_index[entity].discard(doc_id)
                                if not self.entity_index[entity]:
                                    del self.entity_index[entity]
                        
                        # Remove from keyword index
                        for keyword in doc_data.get('keywords', []):
                            if keyword in self.keyword_index:
                                self.keyword_index[keyword].discard(doc_id)
                                if not self.keyword_index[keyword]:
                                    del self.keyword_index[keyword]
                    
                    # Remove node from graph
                    if self.graph.has_node(node_id):
                        self.graph.remove_node(node_id)
                    
                    # Clean up mappings
                    del self.document_nodes[doc_id]
                    if node_id in self.node_documents:
                        del self.node_documents[node_id]
            
            # Save updated graph
            self._save_graph()
            
            logger.info(f"Deleted {len(document_ids)} documents from NetworkX graph")
            
        except Exception as e:
            logger.error(f"Error deleting documents from NetworkX graph: {str(e)}")
            raise
    
    def get_document_count(self) -> int:
        """Get the number of documents in the store."""
        return len(self.document_nodes)
    
    def _save_graph(self):
        """Save the NetworkX graph and metadata."""
        try:
            graph_path = Path(self.config.get('index_path', './storage/graph'))
            graph_path.mkdir(parents=True, exist_ok=True)
            
            graph_file = graph_path / f"{self.config.get('name', 'default')}_graph.pkl"
            metadata_file = graph_path / f"{self.config.get('name', 'default')}_metadata.json"
            
            # Save graph
            with open(graph_file, 'wb') as f:
                pickle.dump(self.graph, f)
            
            # Save metadata (convert sets to lists for JSON serialization)
            metadata = {
                'document_nodes': self.document_nodes,
                'node_documents': self.node_documents,
                'entity_index': {k: list(v) for k, v in self.entity_index.items()},
                'keyword_index': {k: list(v) for k, v in self.keyword_index.items()}
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.debug("Saved NetworkX graph and metadata")
            
        except Exception as e:
            logger.error(f"Error saving NetworkX graph: {str(e)}")
            raise
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph structure."""
        try:
            stats = {
                'num_nodes': self.graph.number_of_nodes(),
                'num_edges': self.graph.number_of_edges(),
                'num_documents': len(self.document_nodes),
                'num_entities': len(self.entity_index),
                'num_keywords': len(self.keyword_index),
                'density': nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0,
                'is_connected': nx.is_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
            }
            
            if self.graph.number_of_nodes() > 0:
                try:
                    stats['average_clustering'] = nx.average_clustering(self.graph)
                except:
                    stats['average_clustering'] = 0
                
                try:
                    if nx.is_connected(self.graph):
                        stats['diameter'] = nx.diameter(self.graph)
                    else:
                        stats['diameter'] = 'N/A (disconnected)'
                except:
                    stats['diameter'] = 'N/A'
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting graph stats: {str(e)}")
            return {}
