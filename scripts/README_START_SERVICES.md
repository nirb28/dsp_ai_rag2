# Starting All RAG System Services

This document describes how to use the `start_all.py` script to launch all components of the RAG system together.

## Overview

The `start_all.py` script starts the following services:
- FastAPI Server (API) on port 9000
- Model Server on port 9001
- Streamlit UI on port 9501

## Requirements

- Python 3.8+
- All dependencies for both the API server and Streamlit UI must be installed
- python-dotenv package for environment variable handling

## Usage

### Basic Usage

To start all services with default settings:

```bash
python start_all.py
```

This will:
1. Start the FastAPI server on port 9000
2. Start the Streamlit UI on port 9501
3. Open browser tabs for the API documentation and Streamlit UI
4. Monitor and display output from both services with color coding

### Command Line Options

The script supports several command-line arguments:

- `--no-browser`: Start services without automatically opening browser tabs
  ```bash
  python start_all.py --no-browser
  ```

- `--api-only`: Start only the API server
  ```bash
  python start_all.py --api-only
  ```

- `--model-only`: Start only the Model server
  ```bash
  python start_all.py --model-only
  ```

- `--ui-only`: Start only the Streamlit UI
  ```bash
  python start_all.py --ui-only
  ```

## Stopping Services

To stop all services, press `Ctrl+C` in the terminal where the script is running. The script will gracefully terminate all running processes. Shutdown events are also recorded in the log file.

## Troubleshooting

### Port Configuration

The script uses port values from the `.env` file. Here are the default ports:

```
API_PORT=9000
MODEL_SERVER_PORT=9001
STREAMLIT_PORT=9502
```

To change these ports, simply modify the values in your `.env` file. No changes to the script are needed.

### Process Monitoring and Logging

The script monitors all processes and will exit if any of them terminate unexpectedly. Output from each service is color-coded for easier identification:
- API server: Blue
- Model server: Purple
- Streamlit UI: Green

### Logging

All output and events are logged to both the console and a log file. The log file location is determined by the `LOG_PATH` variable in the `.env` file. By default, logs are stored in a `logs` directory in the project root.

Log files are named with a timestamp format: `rag_services_YYYYMMDD_HHMMSS.log`

To change the log directory, modify the `.env` file:

```
LOG_PATH=your/custom/path
```
