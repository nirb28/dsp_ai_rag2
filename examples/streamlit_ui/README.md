# RAG Chatbot Streamlit UI

This folder contains a Streamlit-based user interface for interacting with the RAG system.

## Features
- Chat with the RAG system
- Select different configurations via dropdown
- View sources used in responses
- Upload documents to specific configurations

## Running the Application

Make sure the RAG API server is running first, then run one of the following:

### Default Port (8501)

```
cd examples/streamlit_ui
streamlit run app.py
```

### Custom Port (9501)

```
cd examples/streamlit_ui
python run_app.py
```

Using `run_app.py` will start the Streamlit app on port 9501.
