#!/usr/bin/env python
"""
Run script to launch the Streamlit app on port 9501
python run_app.py
"""
import os
import sys
import subprocess

if __name__ == "__main__":
    # Set the port for Streamlit
    port = 9501
    
    # Get the absolute path to app.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "app.py")
    
    # Run Streamlit on the specified port
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.port", str(port)]
    
    print(f"Starting Streamlit app on port {port}...")
    subprocess.run(cmd)
