import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
import numpy as np

from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import FAISSVectorStore
from app.config import ChunkingConfig, ChunkingStrategy, EmbeddingConfig, EmbeddingModel, VectorStoreConfig

class TestDocumentProcessor:
    def test_extract_text_from_txt(self):
        """Test extracting text from a TXT file."""
        processor = DocumentProcessor()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.")
            temp_file = f.name
        
        try:
            text = processor.extract_text(temp_file)
            assert text == "This is a test document."
        finally:
            os.unlink(temp_file)
    
    def test_extract_text_unsupported_format(self):
        """Test extracting text from unsupported file format."""
        processor = DocumentProcessor()
        
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                processor.extract_text(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_chunk_text_fixed_size(self):
        """Test chunking text with fixed size strategy."""
        processor = DocumentProcessor()
        config = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=50,
            chunk_overlap=10
        )
        
        text = "This is a long text that should be chunked into smaller pieces for testing purposes."
        chunks = processor.chunk_text(text, config)
        
        assert len(chunks) > 1
        assert all(len(chunk.page_content) <= 60 for chunk in chunks)  # Allow some flexibility
    
    def test_chunk_text_recursive(self):
        """Test chunking text with recursive strategy."""
        processor = DocumentProcessor()
        config = ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE_TEXT,
            chunk_size=100,
            chunk_overlap=20
        )
        
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3 with more content that should be split."
        chunks = processor.chunk_text(text, config)
        
        assert len(chunks) >= 1
        assert all(isinstance(chunk.page_content, str) for chunk in chunks)
    
    def test_process_document(self):
        """Test processing a complete document."""
        processor = DocumentProcessor()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document for processing.")
            temp_file = f.name
        
        try:
            config = ChunkingConfig()
            document = processor.process_document(
                temp_file, 
                "test.txt", 
                "test_configuration", 
                config,
                {"author": "test"}
            )
            
            assert document.filename == "test.txt"
            assert document.configuration_name == "test_configuration"
            assert document.content == "This is a test document for processing."
            assert document.metadata["author"] == "test"
            assert document.file_type == "txt"
        finally:
            os.unlink(temp_file)

class TestEmbeddingService:
    @patch('sentence_transformers.SentenceTransformer')
    def test_sentence_transformers_initialization(self, mock_st):
        """Test initialization with SentenceTransformers model."""
        config = EmbeddingConfig(model=EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MINILM)
        
        service = EmbeddingService(config)
        
        mock_st.assert_called_once_with("all-MiniLM-L6-v2")
        assert service.config == config
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_embed_texts_sentence_transformers(self, mock_st):
        """Test embedding texts with SentenceTransformers."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_st.return_value = mock_model
        
        config = EmbeddingConfig(model=EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MINILM)
        service = EmbeddingService(config)
        
        texts = ["text1", "text2"]
        embeddings = service.embed_texts(texts)
        
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_embed_query(self, mock_st):
        """Test embedding a single query."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st.return_value = mock_model
        
        config = EmbeddingConfig(model=EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MINILM)
        service = EmbeddingService(config)
        
        embedding = service.embed_query("test query")
        
        assert embedding == [0.1, 0.2, 0.3]
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_dimension(self, mock_st):
        """Test getting embedding dimension."""
        mock_st.return_value = MagicMock()
        
        config = EmbeddingConfig(model=EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MINILM)
        service = EmbeddingService(config)
        
        dimension = service.get_dimension()
        assert dimension == 384

class TestFAISSVectorStore:
    @patch('faiss.IndexFlatIP')
    @patch('sentence_transformers.SentenceTransformer')
    def test_initialization(self, mock_st, mock_faiss_index):
        """Test FAISS vector store initialization."""
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        config = VectorStoreConfig(index_path="./test_index")
        embedding_config = EmbeddingConfig(model=EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MINILM)
        embedding_service = EmbeddingService(embedding_config)
        
        store = FAISSVectorStore(config, embedding_service)
        
        mock_faiss_index.assert_called_once_with(384)  # Dimension for all-MiniLM
        assert store.dimension == 384
    
    @patch('faiss.IndexFlatIP')
    @patch('faiss.normalize_L2')
    @patch('sentence_transformers.SentenceTransformer')
    def test_add_documents(self, mock_st, mock_normalize, mock_faiss_index):
        """Test adding documents to FAISS store."""
        from langchain.docstore.document import Document as LangchainDocument
        
        # Setup mocks
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_st.return_value = mock_model
        
        mock_index = MagicMock()
        mock_faiss_index.return_value = mock_index
        
        # Create services
        config = VectorStoreConfig(index_path="./test_index")
        embedding_config = EmbeddingConfig(model=EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MINILM)
        embedding_service = EmbeddingService(embedding_config)
        
        with patch.object(FAISSVectorStore, '_save_index'):
            store = FAISSVectorStore(config, embedding_service)
            
            # Create test documents
            docs = [
                LangchainDocument(page_content="doc1", metadata={"id": 1}),
                LangchainDocument(page_content="doc2", metadata={"id": 2})
            ]
            
            doc_ids = store.add_documents(docs)
            
            assert len(doc_ids) == 2
            assert doc_ids[0] == "doc_0"
            assert doc_ids[1] == "doc_1"
            mock_index.add.assert_called_once()
    
    @patch('faiss.IndexFlatIP')
    @patch('faiss.normalize_L2')
    @patch('sentence_transformers.SentenceTransformer')
    def test_similarity_search(self, mock_st, mock_normalize, mock_faiss_index):
        """Test similarity search in FAISS store."""
        from langchain.docstore.document import Document as LangchainDocument
        
        # Setup mocks
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st.return_value = mock_model
        
        mock_index = MagicMock()
        mock_index.search.return_value = (np.array([[0.9, 0.8]]), np.array([[0, 1]]))
        mock_faiss_index.return_value = mock_index
        
        # Create services
        config = VectorStoreConfig(index_path="./test_index")
        embedding_config = EmbeddingConfig(model=EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MINILM)
        embedding_service = EmbeddingService(embedding_config)
        
        with patch.object(FAISSVectorStore, '_save_index'):
            store = FAISSVectorStore(config, embedding_service)
            
            # Add some test documents
            store.documents = [
                LangchainDocument(page_content="doc1", metadata={"id": 1}),
                LangchainDocument(page_content="doc2", metadata={"id": 2})
            ]
            store.metadata = [{"id": 1}, {"id": 2}]
            
            results = store.similarity_search("test query", k=2, similarity_threshold=0.7)
            
            assert len(results) == 2
            assert results[0][1] == 0.9  # similarity score
            assert results[1][1] == 0.8  # similarity score
    
    @patch('faiss.IndexFlatIP')
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_document_count(self, mock_st, mock_faiss_index):
        """Test getting document count."""
        mock_st.return_value = MagicMock()
        mock_faiss_index.return_value = MagicMock()
        
        config = VectorStoreConfig(index_path="./test_index")
        embedding_config = EmbeddingConfig(model=EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MINILM)
        embedding_service = EmbeddingService(embedding_config)
        
        with patch.object(FAISSVectorStore, '_save_index'):
            store = FAISSVectorStore(config, embedding_service)
            
            assert store.get_document_count() == 0
            
            # Simulate adding documents
            store.documents = ["doc1", "doc2", "doc3"]
            assert store.get_document_count() == 3
