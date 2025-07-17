#!/usr/bin/env python
"""
Start all RAG system services:
1. Main API server (FastAPI app)
2. Streamlit UI

This script launches all components of the RAG system and handles clean shutdown.
"""
import os
import sys
import time
import signal
import subprocess
import argparse
import logging
from pathlib import Path
import webbrowser
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define service ports from environment variables or use defaults
API_PORT = int(os.getenv("API_PORT", 9000))
MODEL_SERVER_PORT = int(os.getenv("MODEL_SERVER_PORT", 9001))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 9501))

# Define service colors for output
API_COLOR = "\033[94m"  # Blue
MODEL_SERVER_COLOR = "\033[95m"  # Purple
STREAMLIT_COLOR = "\033[92m"  # Green
RESET_COLOR = "\033[0m"

# Load environment variables
load_dotenv()

# Define paths
REPO_ROOT = Path(__file__).parent.parent  # Go up one level from scripts folder
API_PATH = REPO_ROOT / "app" / "main.py"
MODEL_SERVER_PATH = REPO_ROOT / "app" / "model_server.py"
STREAMLIT_PATH = REPO_ROOT / "examples" / "ui" / "app.py"

# Define log directory and file
LOG_DIR = os.getenv("LOG_PATH", "logs")
LOG_DIR_PATH = REPO_ROOT / LOG_DIR
if not LOG_DIR_PATH.exists():
    os.makedirs(LOG_DIR_PATH, exist_ok=True)

# Create a timestamped log file name
LOG_FILE = LOG_DIR_PATH / f"rag_services_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Track running processes
processes = []


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Start all RAG system services")
    parser.add_argument("--no-browser", action="store_true", help="Don't automatically open browser windows")
    parser.add_argument("--api-only", action="store_true", help="Start only the API server")
    parser.add_argument("--model-only", action="store_true", help="Start only the Model server")
    parser.add_argument("--ui-only", action="store_true", help="Start only the Streamlit UI")
    return parser.parse_args()


def start_api_server():
    """Start the FastAPI server"""
    message = f"Starting API server on port {API_PORT}..."
    print(f"{API_COLOR}{message}{RESET_COLOR}")
    logging.info(message)
    
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", 
         "--port", str(API_PORT), "--log-level", "info"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    processes.append(("API", api_process))
    message = f"API server started with PID {api_process.pid}"
    print(f"{API_COLOR}{message}{RESET_COLOR}")
    logging.info(message)
    return api_process


def start_model_server():
    """Start the Model server"""
    message = f"Starting Model server on port {MODEL_SERVER_PORT}..."
    print(f"{MODEL_SERVER_COLOR}{message}{RESET_COLOR}")
    logging.info(message)
    
    model_server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.model_server:app", "--host", "0.0.0.0", 
         "--port", str(MODEL_SERVER_PORT), "--log-level", "info"], #, "--reload"
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    processes.append(("ModelServer", model_server_process))
    message = f"Model server started with PID {model_server_process.pid}"
    print(f"{MODEL_SERVER_COLOR}{message}{RESET_COLOR}")
    logging.info(message)
    return model_server_process


def start_streamlit():
    """Start the Streamlit UI"""
    message = f"Starting Streamlit UI on port {STREAMLIT_PORT}..."
    print(f"{STREAMLIT_COLOR}{message}{RESET_COLOR}")
    logging.info(message)
    
    streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(STREAMLIT_PATH), "--server.port", str(STREAMLIT_PORT), "--server.runOnSave=false"],
        cwd=REPO_ROOT / "examples" / "ui",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    processes.append(("Streamlit", streamlit_process))
    message = f"Streamlit UI started with PID {streamlit_process.pid}"
    print(f"{STREAMLIT_COLOR}{message}{RESET_COLOR}")
    logging.info(message)
    return streamlit_process


def setup_logging():
    """Setup logging to both console and file"""
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers (to prevent duplicates)
    for handler in logger.handlers[:]: 
        logger.removeHandler(handler)
    
    # Create formatters
    console_formatter = logging.Formatter("%(message)s")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # Setup file handler (only log to file, not to console)
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    print(f"Logging to file: {LOG_FILE}")
    logging.info(f"Logging started")
    
    return logger


def monitor_output(process, prefix):
    """Monitor and print output from a process with color coding and log to file"""
    if process.poll() is not None:
        return False
    
    while True:
        output = process.stdout.readline()
        if output:
            output_text = output.rstrip()
            # Determine color based on service type
            if prefix == "API":
                color = API_COLOR
            elif prefix == "ModelServer":
                color = MODEL_SERVER_COLOR
            else:  # Streamlit
                color = STREAMLIT_COLOR
                
            # Print colored output to console
            print(f"{color}[{prefix}] {output_text}{RESET_COLOR}")
            
            # Log to file without color codes
            logging.info(f"[{prefix}] {output_text}")
        else:
            break
    
    return True


def cleanup():
    """Terminate all running processes"""
    logging.info("\nShutting down all services...")
    
    for name, process in processes:
        if process.poll() is None:  # If process is still running
            logging.info(f"Terminating {name} process (PID {process.pid})...")
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logging.info(f"Killing {name} process (PID {process.pid})...")
                process.kill()
    
    logging.info("All services have been shut down.")


def open_browser_tabs(args):
    """Open browser tabs for the services"""
    if args.no_browser:
        return

def main():
    """Main function to start all services"""
    # Setup logging
    logger = setup_logging()
    
    args = parse_args()
    
    # Setup signal handlers for clean shutdown
    signal.signal(signal.SIGINT, lambda sig, frame: cleanup())
    signal.signal(signal.SIGTERM, lambda sig, frame: cleanup())
    
    try:
        if not args.ui_only and not args.model_only:
            api_process = start_api_server()
            
        if not args.ui_only and not args.api_only:
            model_server_process = start_model_server()
        
        if not args.api_only and not args.model_only:
            streamlit_process = start_streamlit()
        
        # Open browser tabs if requested
        open_browser_tabs(args)
        
        message = "\nAll services are running. Press Ctrl+C to stop all services.\n"
        print(message)
        logging.info(message)
        
        # Monitor process output
        while True:
            # Check if any process has terminated unexpectedly
            for name, process in processes:
                if process.poll() is not None:
                    message = f"\n{name} process exited with code {process.returncode}"
                    print(message)
                    logging.info(message)
                    cleanup()
                    return
            
            # Print output from processes
            for name, process in processes:
                monitor_output(process, name)
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        message = "\nKeyboard interrupt received."
        print(message)
        logging.info(message)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
