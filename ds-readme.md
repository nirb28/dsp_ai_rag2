python tests/test_batch_document_processing.py --folder your_folder_name --query "Your first query" --query "Your second query"

& .\.venv\Scripts\Activate.ps1; uvicorn app.main:app --reload

Cached models in: ~\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2