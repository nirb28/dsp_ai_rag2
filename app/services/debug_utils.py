import os
import json
import time
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def write_debug_log(
    category: str,
    request_payload: Dict[str, Any],
    response_payload: Dict[str, Any],
    configuration_name: Optional[str] = None
) -> str:
    """
    Write request and response payloads to a debug log file for troubleshooting.
    
    Args:
        category: Category of the request (e.g., 'query', 'retrieve')
        request_payload: The request payload to log
        response_payload: The response payload to log
        configuration_name: Optional configuration name to include in the filename
    
    Returns:
        Path to the created log file
    """
    try:
        # Create debug_logs directory inside storage if it doesn't exist
        debug_dir = os.path.join("storage", "debug_logs")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Generate a unique filename with timestamp and UUID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        # Include configuration_name in filename if provided
        if configuration_name:
            filename = f"{timestamp}_{category}_{configuration_name}_{unique_id}.json"
        else:
            filename = f"{timestamp}_{category}_{unique_id}.json"
        
        filepath = os.path.join(debug_dir, filename)
        
        # Create a structured log with request and response sections
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "configuration_name": configuration_name,
            "request": request_payload,
            "response": response_payload
        }
        
        # Write to file with indentation for readability
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, default=str)
            
        logger.info(f"Debug log written to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to write debug log: {str(e)}")
        return ""
