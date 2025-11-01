import os
import sys
import json
import time
import requests
import argparse
from typing import Dict, Any, List

# Setup for importing from app directory
sys.path.insert(0, os.path.abspath('.'))

# Base URL for API calls
BASE_URL = "http://localhost:9000/api/v1"

def test_query_debug_logging(configuration_name: str = "default") -> None:
    """
    Test debug logging for the query endpoint
    
    Args:
        configuration_name: RAG configuration name to use
    """
    print(f"\n=== Testing Query Endpoint Debug Logging with configuration: {configuration_name} ===")
    
    # Prepare query request with debug=True
    query_payload = {
        "query": "What is machine learning?",
        "configuration_name": configuration_name,
        "k": 3,
        "similarity_threshold": 0.1,
        "include_metadata": True,
        "filter_after_reranking": True,
        "debug": True  # Enable debug logging
    }
    
    # Send query request
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/query", json=query_payload)
    processing_time = time.time() - start_time
    
    # Check response
    if response.status_code == 200:
        print(f"Query successful in {processing_time:.2f}s")
        data = response.json()
        print(f"Answer: {data['answer'][:100]}...")
        print(f"Sources: {len(data['sources'])}")
        
        # Check storage/debug_logs directory for new log file
        debug_dir = os.path.join("storage", "debug_logs")
        if os.path.exists(debug_dir):
            # Get most recent log file in the directory
            log_files = [f for f in os.listdir(debug_dir) if f.startswith(time.strftime("%Y%m%d_")) and "query" in f]
            if log_files:
                most_recent = max(log_files, key=lambda f: os.path.getmtime(os.path.join(debug_dir, f)))
                log_path = os.path.join(debug_dir, most_recent)
                print(f"Debug log written to: {log_path}")
                
                # Display log structure
                with open(log_path, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                    print("\nLog Structure:")
                    print(f"- timestamp: {log_data.get('timestamp')}")
                    print(f"- category: {log_data.get('category')}")
                    print(f"- configuration_name: {log_data.get('configuration_name')}")
                    print("- request: {")
                    print(f"    query: {log_data.get('request', {}).get('query')}")
                    print("    ... (additional request details)")
                    print("  }")
                    print("- response: {")
                    print(f"    query: {log_data.get('response', {}).get('query')}")
                    print(f"    answer: {log_data.get('response', {}).get('answer', '')[:50]}...")
                    print(f"    sources: {len(log_data.get('response', {}).get('sources', []))} items")
                    print("    ... (additional response details)")
                    print("  }")
            else:
                print("No debug logs found!")
        else:
            print(f"Debug directory not found: {debug_dir}")
    else:
        print(f"Query failed with status code: {response.status_code}")
        print(response.text)

def test_retrieve_debug_logging(configuration_name: str = "default", use_query_expansion: bool = False) -> None:
    """
    Test debug logging for the retrieve endpoint
    
    Args:
        configuration_name: RAG configuration name to use
        use_query_expansion: Whether to include query expansion in the request
    """
    print(f"\n=== Testing Retrieve Endpoint Debug Logging with configuration: {configuration_name} ===")
    
    # Prepare retrieve request with debug=True
    retrieve_payload = {
        "query": "What is machine learning?",
        "configuration_name": configuration_name,
        "k": 5,
        "similarity_threshold": 0.1,
        "include_metadata": True,
        "use_reranking": True,
        "filter_after_reranking": True,
        "debug": True  # Enable debug logging
    }
    
    # Add query expansion if requested
    if use_query_expansion:
        retrieve_payload["query_expansion"] = {
            "enabled": True,
            "strategy": "multi_query",
            "llm_config_name": "nvidia-llama3-8b",
            "num_queries": 3,
            "include_metadata": True
        }
        print("Using query expansion with the retrieve request")
    
    # Send retrieve request
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/retrieve", json=retrieve_payload)
    processing_time = time.time() - start_time
    
    # Check response
    if response.status_code == 200:
        print(f"Retrieve successful in {processing_time:.2f}s")
        data = response.json()
        print(f"Documents retrieved: {len(data['documents'])}")
        
        # Check storage/debug_logs directory for new log file
        debug_dir = os.path.join("storage", "debug_logs")
        if os.path.exists(debug_dir):
            # Get most recent log file in the directory
            log_files = [f for f in os.listdir(debug_dir) if f.startswith(time.strftime("%Y%m%d_")) and "retrieve" in f]
            if log_files:
                most_recent = max(log_files, key=lambda f: os.path.getmtime(os.path.join(debug_dir, f)))
                log_path = os.path.join(debug_dir, most_recent)
                print(f"Debug log written to: {log_path}")
                
                # Display log structure
                with open(log_path, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                    print("\nLog Structure:")
                    print(f"- timestamp: {log_data.get('timestamp')}")
                    print(f"- category: {log_data.get('category')}")
                    print(f"- configuration_name: {log_data.get('configuration_name')}")
                    print("- request: {")
                    print(f"    query: {log_data.get('request', {}).get('query')}")
                    print("    ... (additional request details)")
                    print("  }")
                    print("- response: {")
                    print(f"    documents: {len(log_data.get('response', {}).get('documents', []))} items")
                    print(f"    total_found: {log_data.get('response', {}).get('total_found')}")
                    print("    ... (additional response details)")
                    print("  }")
                    
                    # Check for query expansion metadata
                    if use_query_expansion and log_data.get('response', {}).get('query_expansion_metadata'):
                        print("\nQuery Expansion Metadata:")
                        metadata = log_data['response']['query_expansion_metadata']
                        print(f"- strategy: {metadata.get('strategy')}")
                        print(f"- expanded_queries: {len(metadata.get('expanded_queries', []))} queries")
            else:
                print("No debug logs found!")
        else:
            print(f"Debug directory not found: {debug_dir}")
    else:
        print(f"Retrieve failed with status code: {response.status_code}")
        print(response.text)

def test_multi_config_retrieve_debug_logging(config_names: List[str], fusion_method: str = "rrf") -> None:
    """
    Test debug logging for the retrieve endpoint with multiple configurations
    
    Args:
        config_names: List of configuration names to use
        fusion_method: Fusion method to use (rrf or simple)
    """
    print(f"\n=== Testing Multi-Configuration Retrieve Debug Logging with configurations: {config_names} ===")
    
    # Prepare retrieve request with debug=True and multiple configurations
    retrieve_payload = {
        "query": "What is machine learning?",
        "configuration_names": config_names,
        "k": 5,
        "similarity_threshold": 0.1,
        "include_metadata": True,
        "use_reranking": True,
        "filter_after_reranking": True,
        "fusion_method": fusion_method,
        "debug": True  # Enable debug logging
    }
    
    # Send retrieve request
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/retrieve", json=retrieve_payload)
    processing_time = time.time() - start_time
    
    # Check response
    if response.status_code == 200:
        print(f"Multi-config retrieve successful in {processing_time:.2f}s")
        data = response.json()
        print(f"Documents retrieved: {len(data['documents'])}")
        print(f"Fusion method: {data.get('fusion_method')}")
        print(f"Configurations used: {data.get('configuration_names')}")
        
        # Check storage/debug_logs directory for new log file
        debug_dir = os.path.join("storage", "debug_logs")
        if os.path.exists(debug_dir):
            # Get most recent log file in the directory
            log_files = [f for f in os.listdir(debug_dir) if f.startswith(time.strftime("%Y%m%d_")) and "retrieve" in f and "multi_" in f]
            if log_files:
                most_recent = max(log_files, key=lambda f: os.path.getmtime(os.path.join(debug_dir, f)))
                log_path = os.path.join(debug_dir, most_recent)
                print(f"Debug log written to: {log_path}")
                
                # Display log structure
                with open(log_path, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                    print("\nLog Structure:")
                    print(f"- timestamp: {log_data.get('timestamp')}")
                    print(f"- category: {log_data.get('category')}")
                    print(f"- configuration_name: {log_data.get('configuration_name')}")
                    print("- request: {")
                    print(f"    configuration_names: {log_data.get('request', {}).get('configuration_names')}")
                    print("    ... (additional request details)")
                    print("  }")
                    print("- response: {")
                    print(f"    documents: {len(log_data.get('response', {}).get('documents', []))} items")
                    print(f"    configuration_names: {log_data.get('response', {}).get('configuration_names')}")
                    print(f"    fusion_method: {log_data.get('response', {}).get('fusion_method')}")
                    print("    ... (additional response details)")
                    print("  }")
            else:
                print("No debug logs found!")
        else:
            print(f"Debug directory not found: {debug_dir}")
    else:
        print(f"Multi-config retrieve failed with status code: {response.status_code}")
        print(response.text)

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test debug logging for RAG endpoints")
    parser.add_argument("--config", type=str, default="batch_ml_ai_basics_test", help="RAG configuration name to use (default: batch_ml_ai_basics_test)")
    parser.add_argument("--create-new", action="store_true", help="Force creation of a new test configuration (not implemented, will warn)")
    parser.add_argument("--multi-config", action="store_true", help="Test with multiple configurations")
    parser.add_argument("--query-expansion", action="store_true", help="Test with query expansion")
    args = parser.parse_args()
    
    # Warn if --create-new is set (not implemented)
    if args.create_new:
        print("⚠️  --create-new is not implemented in this script. Only existing configs will be used.")
    print(f"\nℹ️ Using configuration: {args.config}")
    
    # Create the debug_logs directory if it doesn't exist
    os.makedirs(os.path.join("storage", "debug_logs"), exist_ok=True)
    
    # Run query endpoint test
    #test_query_debug_logging(configuration_name=args.config)
    
    # Run retrieve endpoint test
    test_retrieve_debug_logging(configuration_name=args.config, use_query_expansion=args.query_expansion)
    #args.multi_config = True
    # Run multi-config retrieve test if requested
    if args.multi_config:
        # Use two test configurations - adjust as needed based on your available configs
        configs = [args.config, "batch_rl-docs_test"]
        test_multi_config_retrieve_debug_logging(configs)

if __name__ == "__main__":
    main()
