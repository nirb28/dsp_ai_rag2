python tests/test_batch_document_processing.py --folder your_folder_name --query "Your first query" --query "Your second query"

& .\.venv\Scripts\Activate.ps1; 
uvicorn app.main:app --port 9000 --reload --host 0.0.0.0
python -m app.model_server
