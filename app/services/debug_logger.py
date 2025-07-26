"""
Debug logging service for detailed request/response logging.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import uuid


class DebugLogger:
    """Service for logging detailed debug information to files."""
    
    def __init__(self, storage_path: str = "./storage"):
        self.storage_path = Path(storage_path)
        self.debug_logs_path = self.storage_path / "debug_logs"
        self.debug_logs_path.mkdir(parents=True, exist_ok=True)
    
    def log_request_response(
        self,
        category: str,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any],
        processing_time: float,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log request and response data to a debug file.
        
        Args:
            category: Category name (e.g., 'query', 'retrieve')
            request_data: The request payload
            response_data: The response payload
            processing_time: Processing time in seconds
            additional_info: Additional debug information
            
        Returns:
            The filename of the created debug log
        """
        timestamp = datetime.now()
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{category}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{unique_id}.log"
        filepath = self.debug_logs_path / filename
        
        # Prepare debug log content
        log_content = {
            "timestamp": timestamp.isoformat(),
            "category": category,
            "unique_id": unique_id,
            "processing_time_seconds": processing_time,
            "request": self._sanitize_data(request_data),
            "response": self._sanitize_data(response_data),
            "additional_info": additional_info or {}
        }
        
        # Write to file with proper formatting
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"DEBUG LOG - {category.upper()}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Timestamp: {timestamp.isoformat()}\n")
                f.write(f"Unique ID: {unique_id}\n")
                f.write(f"Processing Time: {processing_time:.4f} seconds\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("REQUEST PAYLOAD:\n")
                f.write("-" * 40 + "\n")
                f.write(json.dumps(log_content["request"], indent=2, ensure_ascii=False))
                f.write("\n\n")
                
                f.write("RESPONSE PAYLOAD:\n")
                f.write("-" * 40 + "\n")
                f.write(json.dumps(log_content["response"], indent=2, ensure_ascii=False))
                f.write("\n\n")
                
                if additional_info:
                    f.write("ADDITIONAL DEBUG INFO:\n")
                    f.write("-" * 40 + "\n")
                    f.write(json.dumps(additional_info, indent=2, ensure_ascii=False))
                    f.write("\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("END OF DEBUG LOG\n")
                f.write("=" * 80 + "\n")
                
            return filename
            
        except Exception as e:
            # If file writing fails, at least log the error
            print(f"Failed to write debug log: {str(e)}")
            return f"error_{unique_id}.log"
    
    def _sanitize_data(self, data: Any) -> Any:
        """
        Sanitize data for logging by removing sensitive information.
        
        Args:
            data: Data to sanitize
            
        Returns:
            Sanitized data
        """
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                # Redact sensitive fields
                if any(sensitive in key.lower() for sensitive in ['api_key', 'password', 'token', 'secret']):
                    sanitized[key] = "[REDACTED]"
                else:
                    sanitized[key] = self._sanitize_data(value)
            return sanitized
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        else:
            return data
    
    def cleanup_old_logs(self, days_to_keep: int = 7):
        """
        Clean up debug logs older than specified days.
        
        Args:
            days_to_keep: Number of days to keep logs
        """
        try:
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
            
            for log_file in self.debug_logs_path.glob("*.log"):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    
        except Exception as e:
            print(f"Failed to cleanup old debug logs: {str(e)}")
    
    def get_debug_logs_info(self) -> Dict[str, Any]:
        """
        Get information about existing debug logs.
        
        Returns:
            Dictionary with debug logs information
        """
        try:
            log_files = list(self.debug_logs_path.glob("*.log"))
            
            return {
                "total_files": len(log_files),
                "debug_logs_path": str(self.debug_logs_path),
                "recent_files": [
                    {
                        "filename": f.name,
                        "size_bytes": f.stat().st_size,
                        "created_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                    }
                    for f in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]
                ]
            }
        except Exception as e:
            return {
                "error": f"Failed to get debug logs info: {str(e)}",
                "total_files": 0,
                "debug_logs_path": str(self.debug_logs_path),
                "recent_files": []
            }


# Global debug logger instance
debug_logger = DebugLogger()
