from fastapi import APIRouter

# Create a documentation router
router = APIRouter(
    tags=["Documentation"]
)

@router.get("/chunking-strategies")
async def chunking_strategies_documentation():
    """
    Get detailed information about available chunking strategies.
    
    This endpoint provides comprehensive documentation about the different text chunking
    strategies available in the system, including their parameters, use cases, and examples.
    """
    return {
        "strategies": {
            "fixed_size": {
                "description": "Splits text into chunks of a fixed character length",
                "implementation": "Uses CharacterTextSplitter from LangChain",
                "parameters": {
                    "chunk_size": "Number of characters per chunk (default: 1000, range: 100-4000)",
                    "chunk_overlap": "Number of overlapping characters between chunks (default: 200, range: 0-1000)"
                },
                "example": {
                    "strategy": "fixed_size",
                    "chunk_size": 1000,
                    "chunk_overlap": 200
                },
                "best_for": "Simple documents where semantic structure is less important",
                "drawbacks": "May break sentences or paragraphs at arbitrary points"
            },
            "recursive_text": {
                "description": "Recursively splits text using a hierarchy of separators",
                "implementation": "Uses RecursiveCharacterTextSplitter from LangChain",
                "parameters": {
                    "chunk_size": "Target chunk size in characters (default: 1000, range: 100-4000)",
                    "chunk_overlap": "Number of overlapping characters (default: 200, range: 0-1000)",
                    "separators": "List of separators to try in order (default: [\"\\n\\n\", \"\\n\", \" \", \"\"])"
                },
                "example": {
                    "strategy": "recursive_text",
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "separators": ["\n\n", "\n", " ", ""]
                },
                "best_for": "General purpose chunking that better preserves semantic structure",
                "default_strategy": True
            },
            "sentence": {
                "description": "Splits text based on sentence boundaries while respecting token limits",
                "implementation": "Uses SentenceTransformersTokenTextSplitter from LangChain",
                "parameters": {
                    "tokens_per_chunk": "Number of tokens per chunk (uses chunk_size value, default: 1000)",
                    "chunk_overlap": "Number of overlapping tokens between chunks (default: 200)"
                },
                "example": {
                    "strategy": "sentence",
                    "chunk_size": 1000,  # Used as tokens_per_chunk
                    "chunk_overlap": 200
                },
                "best_for": "Text where preserving sentence boundaries is important",
                "notes": "This strategy uses token counts rather than character counts"
            },
            "semantic": {
                "description": "Enhanced recursive splitting with finer separators for better semantic chunking",
                "implementation": "Currently uses RecursiveCharacterTextSplitter with enhanced separators",
                "parameters": {
                    "chunk_size": "Target chunk size in characters (default: 1000, range: 100-4000)",
                    "chunk_overlap": "Number of overlapping characters (default: 200, range: 0-1000)"
                },
                "example": {
                    "strategy": "semantic",
                    "chunk_size": 1000,
                    "chunk_overlap": 200
                },
                "best_for": "Text where preserving semantic units (sentences, clauses) is critical",
                "notes": "Currently implemented as enhanced recursive chunking, future versions may use more sophisticated semantic analysis"
            }
        }
    }

@router.get("/embedding-models")
async def embedding_models_documentation():
    """
    Get detailed information about available embedding models.
    
    This endpoint provides comprehensive documentation about the different embedding
    models available in the system, including their parameters, use cases, and examples.
    """
    return {
        "models": {
            "sentence-transformers/all-MiniLM-L6-v2": {
                "description": "Lightweight sentence transformer model for generating embeddings",
                "dimensions": 384,
                "provider": "Sentence Transformers (HuggingFace)",
                "speed": "Fast",
                "performance": "Good balance between speed and quality",
                "use_case": "General purpose text embeddings",
                "notes": "Default model due to good balance of speed and quality",
                "default_model": True
            },
            "sentence-transformers/all-mpnet-base-v2": {
                "description": "High-quality sentence transformer model with better performance",
                "dimensions": 768,
                "provider": "Sentence Transformers (HuggingFace)",
                "speed": "Moderate",
                "performance": "Better quality than MiniLM models",
                "use_case": "When embedding quality is more important than speed"
            },
            "text-embedding-ada-002": {
                "description": "OpenAI's embedding model",
                "dimensions": 1536,
                "provider": "OpenAI",
                "speed": "Fast (API call)",
                "performance": "High quality",
                "use_case": "Production systems where quality is critical",
                "requirements": "Requires OpenAI API key"
            },
            "triton-embedding": {
                "description": "Embedding model served via Triton inference server",
                "dimensions": "Varies based on model",
                "provider": "Local Triton server",
                "speed": "Varies based on hardware",
                "performance": "Depends on model deployed on Triton",
                "use_case": "Self-hosted embedding services",
                "requirements": "Requires running Triton server"
            },
            "local-model-server": {
                "description": "Embedding model from local model server",
                "dimensions": "Varies based on model",
                "provider": "Local server",
                "speed": "Varies based on hardware",
                "performance": "Depends on deployed model",
                "use_case": "Self-hosted embedding services",
                "requirements": "Requires running local model server"
            }
        }
    }

@router.get("/vector-stores")
async def vector_stores_documentation():
    """
    Get detailed information about available vector store options.
    
    This endpoint provides comprehensive documentation about the different vector
    store options available in the system, including their parameters, use cases, and examples.
    """
    return {
        "vector_stores": {
            "faiss": {
                "description": "Facebook AI Similarity Search (FAISS) is a library for efficient similarity search",
                "implementation": "Uses FAISS via LangChain integration",
                "parameters": {
                    "index_path": "Path for FAISS index files (default: ./storage/faiss_index)",
                    "dimension": "Embedding dimension (default: 384, must match embedding model)"
                },
                "features": [
                    "Fast similarity search",
                    "Local storage (no external services required)",
                    "Efficient memory usage",
                    "Support for various distance metrics"
                ],
                "best_for": "Local development and smaller to medium datasets",
                "limitations": "Limited scaling for very large datasets compared to distributed options",
                "default_store": True
            },
            "redis": {
                "description": "Redis as a vector database with Redis Stack",
                "implementation": "Uses Redis via LangChain integration",
                "parameters": {
                    "redis_host": "Redis server hostname (default: localhost)",
                    "redis_port": "Redis server port (default: 6379)",
                    "redis_password": "Redis password if authentication is required",
                    "redis_index_name": "Redis search index name (default: document-index)",
                    "dimension": "Embedding dimension (default: 384, must match embedding model)"
                },
                "features": [
                    "Scalable vector search",
                    "Persistence options",
                    "Support for high-throughput operations",
                    "Hybrid search capabilities"
                ],
                "best_for": "Production deployments and larger datasets",
                "requirements": "Requires Redis Stack server running"
            }
        }
    }

@router.get("/reranker-models")
async def reranker_models_documentation():
    """
    Get detailed information about available reranking models.
    
    This endpoint provides comprehensive documentation about the different reranking
    models available in the system, including their parameters, use cases, and examples.
    """
    return {
        "models": {
            "none": {
                "description": "No reranking is applied",
                "provider": "N/A",
                "parameters": {},
                "use_case": "When initial vector similarity is sufficient",
                "default": True
            },
            "cohere-rerank": {
                "description": "Cohere's reranking service for improving retrieval quality",
                "provider": "Cohere API",
                "parameters": {
                    "top_n": "Number of initial results to rerank (default: 10, range: 1-50)",
                    "score_threshold": "Minimum relevance score to include results (default: 0.1, range: 0-1)"
                },
                "features": [
                    "High quality reranking",
                    "Optimized for relevance to query",
                    "Easy to integrate"
                ],
                "use_case": "Production systems where retrieval quality is critical",
                "requirements": "Requires Cohere API key"
            },
            "bge-reranker-large": {
                "description": "BGE Reranker model for improved retrieval quality",
                "provider": "Local model or API",
                "parameters": {
                    "top_n": "Number of initial results to rerank (default: 10, range: 1-50)",
                    "score_threshold": "Minimum relevance score to include results (default: 0.1, range: 0-1)"
                },
                "features": [
                    "High quality open-source reranker",
                    "Can be deployed locally"
                ],
                "use_case": "Local or self-hosted deployments requiring quality reranking",
                "requirements": "Requires model to be accessible"
            },
            "cross-encoder/ms-marco-MiniLM-L-6-v2": {
                "description": "Sentence Transformers cross-encoder model for reranking",
                "provider": "Sentence Transformers (HuggingFace)",
                "parameters": {
                    "top_n": "Number of initial results to rerank (default: 10, range: 1-50)",
                    "score_threshold": "Minimum relevance score to include results (default: 0.1, range: 0-1)"
                },
                "features": [
                    "Lightweight cross-encoder model",
                    "Good balance of speed and quality",
                    "Can be deployed locally"
                ],
                "use_case": "Local deployments needing moderate reranking quality with good speed"
            }
        }
    }

@router.get("/mcp-integration")
async def mcp_integration_documentation():
    """
    Get detailed information about MCP (Model Control Plane) integration.
    
    This endpoint provides comprehensive documentation about the MCP integration
    in the RAG service, including configuration options, API endpoints, and usage examples.
    """
    return {
        "overview": "The MCP (Model Control Plane) integration allows the RAG service to fetch and import documents from an MCP server.",
        "configuration": {
            "environment_variables": {
                "MCP_ENABLED": "Boolean flag to enable/disable MCP integration (default: false)",
                "MCP_SERVER_URL": "URL of the MCP server (default: http://localhost:9002)"
            },
            "config_model": "MCPConfig in app/config.py defines the configuration structure"
        },
        "services": {
            "MCPClientService": "Handles communication with the MCP server API",
            "MCPDocumentService": "Processes MCP resources into RAG documents"
        },
        "api_endpoints": {
            "import_document": {
                "path": "/api/v1/mcp/import/{uri}",
                "method": "POST",
                "description": "Import a single document from MCP by its URI",
                "parameters": {
                    "uri": "URI of the document to import"
                },
                "returns": "Status of the import operation"
            },
            "import_all_documents": {
                "path": "/api/v1/mcp/import_all",
                "method": "POST",
                "description": "Import all documents from MCP",
                "parameters": {
                    "title_filter": "Optional filter to import only documents with titles containing this string"
                },
                "returns": "Status of the import operation and number of documents imported"
            },
            "server_info": {
                "path": "/api/v1/mcp/server_info",
                "method": "GET",
                "description": "Get information about the connected MCP server",
                "returns": "Server information including name, version, and status"
            }
        },
        "usage_examples": {
            "enable_mcp": "Set MCP_ENABLED=true and MCP_SERVER_URL in environment variables",
            "import_document": "POST /api/v1/mcp/import/my-document-uri",
            "import_all": "POST /api/v1/mcp/import_all",
            "filtered_import": "POST /api/v1/mcp/import_all?title_filter=research"
        }
    }

@router.get("/security")
async def security_documentation():
    """
    Get detailed information about security and authentication features.
    
    This endpoint provides comprehensive documentation about the security
    configuration options, JWT authentication, and metadata filtering capabilities.
    """
    return {
        "overview": "The RAG system supports optional security authentication to control access to query and retrieve endpoints with JWT Bearer token authentication and metadata-based document filtering.",
        "security_types": {
            "jwt_bearer": {
                "description": "JSON Web Token based authentication with configurable validation",
                "status": "Available",
                "features": [
                    "Configurable secret keys and algorithms",
                    "Issuer and audience validation",
                    "Expiration and issued-at validation",
                    "Metadata filtering via JWT claims",
                    "Secure token validation with proper error handling"
                ]
            },
            "api_key": {
                "description": "Simple API key authentication",
                "status": "Future implementation",
                "features": ["Custom header configuration", "Multiple API keys support"]
            },
            "oauth2": {
                "description": "OAuth2 flow authentication",
                "status": "Future implementation",
                "features": ["Standard OAuth2 flows", "Token introspection"]
            }
        },
        "configuration": {
            "security_block": {
                "enabled": "Boolean flag to enable/disable security (default: false)",
                "type": "Authentication type (jwt_bearer, api_key, oauth2)",
                "jwt_secret_key": "Secret key for JWT validation (required for JWT)",
                "jwt_algorithm": "Algorithm for JWT validation (default: HS256)",
                "jwt_issuer": "Expected issuer of JWT tokens (optional)",
                "jwt_audience": "Expected audience of JWT tokens (optional)",
                "jwt_require_exp": "Whether to require expiration claim (default: true)",
                "jwt_require_iat": "Whether to require issued-at claim (default: true)",
                "jwt_leeway": "Leeway in seconds for token expiration (default: 0)"
            },
            "example": {
                "security": {
                    "enabled": True,
                    "type": "jwt_bearer",
                    "jwt_secret_key": "your-secret-key",
                    "jwt_algorithm": "HS256",
                    "jwt_issuer": "your-auth-service",
                    "jwt_audience": "rag-api"
                }
            }
        },
        "jwt_claims": {
            "standard_claims": {
                "sub": "Subject (user identifier)",
                "iat": "Issued at timestamp",
                "exp": "Expiration timestamp",
                "iss": "Issuer",
                "aud": "Audience"
            },
            "custom_claims": {
                "metadata_filter": {
                    "description": "Document filtering criteria applied automatically",
                    "type": "Object",
                    "example": {"department": "engineering", "level": "public"}
                }
            }
        },
        "metadata_filtering": {
            "description": "JWT tokens can include metadata_filter claim for automatic document filtering",
            "operators": {
                "equality": {"department": "engineering"},
                "comparison": {"score": {"$gte": 0.8}},
                "array": {"tags": {"$in": ["ai", "ml"]}},
                "logical": {"$and": [{"dept": "eng"}, {"level": "public"}]}
            },
            "filter_merging": "Request filters and JWT filters are combined using $and operator"
        },
        "api_usage": {
            "authentication_header": "Authorization: Bearer <jwt_token>",
            "query_endpoint": "POST /api/v1/query with Authorization header",
            "retrieve_endpoint": "POST /api/v1/retrieve with Authorization header",
            "error_responses": {
                "401": "Unauthorized - Missing/invalid/expired token",
                "400": "Bad Request - Invalid configuration",
                "500": "Internal Server Error - Server configuration issues"
            }
        },
        "testing": {
            "test_script": "test_security_feature.py provides comprehensive security testing",
            "test_coverage": [
                "Configuration creation with security enabled",
                "Authentication without token (should fail)",
                "Authentication with valid JWT token",
                "Metadata filter extraction and application",
                "Invalid and expired token handling"
            ]
        },
        "best_practices": [
            "Use strong, cryptographically secure secret keys",
            "Set appropriate token expiration times",
            "Configure issuer and audience validation for production",
            "Always use HTTPS in production",
            "Implement key rotation for long-term security",
            "Use metadata filters for principle of least privilege",
            "Monitor and log authentication attempts"
        ]
    }
