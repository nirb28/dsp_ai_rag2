# DSP AI RAG2 - Release Notes

## Version 2.4.0 (July 17, 2025)

### New Features

#### Chatbot UI
- Introduced a chatbot user interface with configurable Retrieval-Augmented Generation (RAG) parameters
- Supports chat history and dynamic parameter adjustment
- Integrates directly with backend RAG API endpoints

#### Configuration Management UI
- Added a configuration management interface with full CRUD (Create, Read, Update, Delete) operations
- Includes a JSON editor for advanced configuration editing

#### Core RAG Functionality
- Implemented core RAG features with robust API endpoints
- Enhanced configuration loading and management

#### Environment Configuration
- Added support for model server URLs and LLM configuration via environment variables

### Other Changes
- Updated Postman collections for new endpoints and features

## Version 2.3.0 (July 15, 2025)

### New Features

#### BM25 Search Support
- Added BM25-based keyword search as a new vector store type
- Implemented `BM25VectorStore` class that follows the same interface as other vector stores
- BM25 provides better performance for exact keyword matching and technical terminology
- Keyword search works without requiring embeddings or downloading any models
- Zero-dependency approach that relies purely on statistical algorithms
- Perfect for endpoint-only configurations where no models should be downloaded locally
- Compatible with multi-vector store retrieval for hybrid search approaches

## Version 2.2.0 (July 13, 2025)

### New Features

#### Multi-Vector Store Retrieval with Fusion Methods
- Added support for retrieving from multiple vector stores simultaneously through the `/retrieve` endpoint
- Implemented Reciprocal Rank Fusion (RRF) algorithm for combining results from multiple vector stores
- Added simple fusion method using score normalization and averaging
- Enhanced `RetrieveRequest` model with new fields:
  - `configuration_names`: List of configurations to query
  - `fusion_method`: Method to use for combining results ("rrf" or "simple")
  - `rrf_k_constant`: Parameter to tune the RRF algorithm
- Updated `RetrieveResponse` model to include source tracking information
- Added detailed Swagger documentation with examples for multi-vector store queries
- Each document now includes information about which configuration it originated from

#### System Prompt Customization
- Added ability to override the default system prompt for LLM interactions
- Implemented per-request system prompt customization via the configuration override
- Added flexible fallback mechanism that prioritizes:
  1. Request-specific system prompt (highest priority)
  2. Configuration-defined system prompt
  3. Default system prompt (lowest priority)

#### Configuration Management Improvements
- Added endpoint to retrieve only configuration names without full details
- Added `names_only` query parameter to `/configurations` endpoint
- Created new `ConfigurationNamesResponse` model for streamlined responses
- Implemented `get_configuration_names()` method in RAGService class

#### Enhanced Model Flexibility
- Updated configuration validation to support custom model names beyond predefined enums
- Modified `EmbeddingConfig` and `RerankerConfig` to accept custom model strings
- This enables the add_configuration endpoint to work with custom model names

### Bug Fixes
- Fixed issue with boolean parameter handling in the `/configurations` endpoint
- Enhanced error handling in the retrieve endpoint for multi-configuration queries
- Improved validation for fusion method selection

### Performance Improvements
- Optimized retrieval process for multi-vector store queries
- Ensured efficient reranking on combined result sets

### API Documentation
- Updated Swagger documentation with new examples showing:
  - Basic single-configuration retrieval
  - Multi-configuration retrieval with RRF fusion
  - Result format with source tracking
