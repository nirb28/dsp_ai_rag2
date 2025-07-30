# RAG as a Service Platform

A comprehensive Retrieval-Augmented Generation (RAG) platform built with FastAPI, offering configurable chunking strategies, vector stores, embedding models, and generation models.

## Features

- **Configurable RAG Pipeline**: Choose from multiple chunking strategies, embedding models, and generation models
- **Multiple Vector Store Options**:
  - **FAISS**: High-performance similarity search with FAISS
  - **Redis**: Redis-based vector storage with filtering capabilities
  - **BM25**: Keyword-based search without requiring embeddings or model downloads
- **Multi-Vector Store Retrieval**: Retrieve from multiple vector stores with fusion methods
- **Security & Authentication**: Optional JWT Bearer token authentication with metadata-based document filtering (see `docs/security.md`)
- **Endpoint-Only Mode**: Option to run without downloading models locally (see `docs/ENDPOINT_ONLY_SETUP.md`)
- **Groq Integration**: Fast inference with Groq's LLM API
- **System Prompt Customization**: Override system prompts per request
- **Multiple File Formats**: Support for PDF, TXT, DOCX, and PPTX files
- **RESTful API**: Complete REST API with automatic documentation
- **Configuration Presets**: Pre-configured setups for different use cases
- **Collection Management**: Organize documents into separate collections
- **Comprehensive Testing**: Full test suite with unit and integration tests

## Quick Start

### Prerequisites

- Python 3.8+
- Groq API key (sign up at [console.groq.com](https://console.groq.com))

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd dsp_ai_rag2
```

2. **Create and activate virtual environment**:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

5. **Run the application**:

You can run individual services:
```bash
# Run the API server
python -m app.main

# Run the model server
python -m app.model_server
```

Or use the unified startup script to run all services together:
```bash
python scripts/start_all.py
```

For more information about the startup script options, see `scripts/README_START_SERVICES.md`.

The API will be available at `http://localhost:9000` with documentation at `http://localhost:9000/docs`.

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional, for OpenAI embeddings
STORAGE_PATH=./storage
MAX_FILE_SIZE=10485760  # 10MB
```

### RAG Configuration

The platform supports flexible configuration through JSON objects.

#### Standard Configuration (with FAISS):

```json
{
  "collection_name": "my_collection",
  "chunking": {
    "strategy": "recursive_text",
    "chunk_size": 1000,
    "chunk_overlap": 200
  },
  "vector_store": {
    "type": "faiss",
    "index_path": "./storage/faiss_index",
    "dimension": 384
  },
  "embedding": {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 32
  },
  "generation": {
    "model": "llama3-8b-8192",
    "temperature": 0.7,
    "max_tokens": 1024,
    "top_p": 0.9
  },
  "retrieval_k": 5,
  "similarity_threshold": 0.7
}
```

#### Endpoint-Only Configuration (with BM25):

```json
{
  "configuration_name": "endpoint_only_config",
  "config": {
    "embedding": {
      "enabled": false,
      "model": "none"
    },
    "vector_store": {
      "type": "bm25",
      "index_path": "./storage/bm25_index"
    },
    "generation": {
      "enabled": true,
      "provider": "groq",
      "model": "llama3-70b-8192",
      "temperature": 0.1,
      "system_prompt": "You are a helpful assistant that provides accurate information based on the given context."
    },
    "retrieval": {
      "k": 5
    }
  }
}
```

### Available Options

#### Chunking Strategies
- `fixed_size`: Fixed character-based chunking
- `recursive_text`: Recursive text splitting (recommended)
- `semantic`: Semantic-aware chunking
- `sentence`: Sentence-based chunking

#### Embedding Models
- `sentence-transformers/all-MiniLM-L6-v2`: Fast, lightweight (384 dim)
- `sentence-transformers/all-mpnet-base-v2`: High quality (768 dim)
- `text-embedding-ada-002`: OpenAI embeddings (1536 dim)

#### Generation Models (Groq)
- `llama3-8b-8192`: Fast, efficient
- `llama3-70b-8192`: High quality
- `mixtral-8x7b-32768`: Large context window
- `gemma-7b-it`: Google's Gemma model

## API Endpoints

### Document Management

**Upload Document**
```bash
POST /api/v1/upload
```

**Query Documents**
```bash
POST /api/v1/query
```

### Configuration Management

**Set Configuration**
```bash
POST /api/v1/configure
```

**Get Configuration**
```bash
GET /api/v1/configure/{collection_name}
```

**Apply Preset**
```bash
POST /api/v1/configure/preset/{preset_name}
```

### Collection Management

**List Collections**
```bash
GET /api/v1/collections
```

**Delete Collection**
```bash
DELETE /api/v1/collections/{collection_name}
```

## Usage Examples

### 1. Upload a Document

```python
import requests

# Upload a document
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/upload",
        files={"file": f},
        data={
            "collection_name": "my_docs",
            "process_immediately": True
        }
    )

print(response.json())
```

### 2. Query Documents

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={
        "query": "What is machine learning?",
        "collection_name": "my_docs",
        "k": 5
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])}")
```

### 3. Configure Collection

```python
import requests

config = {
    "collection_name": "high_quality",
    "config": {
        "chunking": {
            "strategy": "recursive_text",
            "chunk_size": 1500,
            "chunk_overlap": 300
        },
        "embedding": {
            "model": "sentence-transformers/all-mpnet-base-v2"
        },
        "generation": {
            "model": "llama3-70b-8192",
            "temperature": 0.3
        }
    }
}

response = requests.post(
    "http://localhost:8000/api/v1/configure",
    json=config
)
```

## Configuration Presets

The platform includes three built-in presets:

### Fast Processing
- Optimized for speed
- Smaller chunks (500 chars)
- Lightweight embedding model
- Fast generation model

### High Quality
- Optimized for accuracy
- Larger chunks (1500 chars)
- High-quality embedding model
- Advanced generation model

### Balanced
- Balance between speed and quality
- Medium chunks (1000 chars)
- Good embedding model
- Versatile generation model

Apply a preset:
```bash
curl -X POST "http://localhost:8000/api/v1/configure/preset/balanced?collection_name=my_collection"
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_api_endpoints.py

# Run with verbose output
pytest -v
```

## Development

### Project Structure

```
dsp_ai_rag2/
├── app/
│   ├── api/
│   │   └── endpoints.py      # API route definitions
│   ├── services/
│   │   ├── document_processor.py  # Document processing
│   │   ├── embedding_service.py   # Embedding generation
│   │   ├── vector_store.py        # FAISS vector store
│   │   ├── generation_service.py  # LLM generation
│   │   └── rag_service.py         # Main RAG orchestration
│   ├── config.py            # Configuration models
│   ├── models.py            # Pydantic models
│   └── main.py              # FastAPI application
├── tests/                   # Test suite
├── storage/                 # Data storage (created automatically)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

### Adding New Features

1. **New Chunking Strategy**: Add to `ChunkingStrategy` enum and implement in `DocumentProcessor`
2. **New Embedding Model**: Add to `EmbeddingModel` enum and implement in `EmbeddingService`
3. **New Vector Store**: Implement new store class following `FAISSVectorStore` pattern
4. **New Generation Model**: Add to `GenerationModel` enum and implement service

## Performance Considerations

- **Batch Processing**: Embeddings are processed in configurable batches
- **Index Persistence**: FAISS indices are saved to disk for persistence
- **Memory Management**: Large documents are processed in chunks
- **Async Operations**: API endpoints use async/await for better concurrency

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated and dependencies installed
2. **API Key Issues**: Verify Groq API key is set in `.env` file
3. **File Upload Errors**: Check file size limits and supported formats
4. **Memory Issues**: Reduce batch sizes or chunk sizes for large documents

### Logs

The application logs to stdout. Increase verbosity by setting log level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- **[Security & Authentication](docs/security.md)** - JWT Bearer token authentication and metadata filtering
- **[Query Expansion](docs/QUERY_EXPANSION.md)** - Multi-query generation and result fusion
- **[NetworkX Graph Store](docs/NETWORKX_GRAPH_STORE.md)** - Graph-based document storage and retrieval
- **[Neo4j Integration](docs/neo4j_integration.md)** - Neo4j graph database integration
- **[Endpoint-Only Setup](docs/ENDPOINT_ONLY_SETUP.md)** - Running without local model downloads
- **[Model Server Setup](docs/model_server_readme.md)** - Setting up local model servers
- **[Release Notes](docs/RELEASE_NOTES.md)** - Version history and changes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- Create an issue on GitHub
- Check the API documentation at `/docs`
- Review the test examples in the `tests/` directory