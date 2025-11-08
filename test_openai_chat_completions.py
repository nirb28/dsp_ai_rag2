"""
Test script for OpenAI-compatible chat completions endpoints.

This script tests the OpenAI-compatible API endpoints for RAG configurations.

Usage:
    python test_openai_chat_completions.py
    
Note: This is a standalone script, not a pytest test suite.
To avoid pytest auto-discovery, the file is named with test_ prefix but
should be run directly with Python, not pytest.
"""

import requests
import json
import time
import sys

# Prevent pytest from collecting these as tests
__test__ = False

BASE_URL = "http://localhost:9000"


def print_section(title):
    """Print a section header."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)


def print_result(success, message):
    """Print test result."""
    status = "✓" if success else "✗"
    print(f"{status} {message}")


def test_health_check():
    """Test that the service is running."""
    print_section("Test 1: Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        print_result(True, f"Service is running (status: {data['status']})")
        return True
    except Exception as e:
        print_result(False, f"Service not reachable: {str(e)}")
        return False


def test_list_configurations():
    """Test listing available configurations."""
    print_section("Test 2: List Available Configurations")
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        endpoints = data.get("openai_compatible_endpoints", {})
        
        if endpoints:
            print_result(True, f"Found {len(endpoints)} configuration(s)")
            for config_name, endpoint in endpoints.items():
                print(f"  - {config_name}: {BASE_URL}{endpoint}")
            return list(endpoints.keys())
        else:
            print_result(False, "No configurations found")
            return []
            
    except Exception as e:
        print_result(False, f"Error listing configurations: {str(e)}")
        return []


def test_simple_chat_completion(config_name):
    """Test simple chat completion."""
    print_section(f"Test 3: Simple Chat Completion ({config_name})")
    
    url = f"{BASE_URL}/{config_name}/v1/chat/completions"
    
    payload = {
        "model": config_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is machine learning? Please provide a brief answer."
            }
        ],
        "temperature": 0.7,
        "max_tokens": 200,
        "k": 3,
        "include_sources": True
    }
    
    try:
        print(f"Sending request to: {url}")
        start_time = time.time()
        
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        elapsed = time.time() - start_time
        result = response.json()
        
        # Validate response structure
        assert "id" in result, "Missing 'id' field"
        assert "object" in result, "Missing 'object' field"
        assert "choices" in result, "Missing 'choices' field"
        assert "usage" in result, "Missing 'usage' field"
        assert len(result["choices"]) > 0, "No choices in response"
        
        choice = result["choices"][0]
        assert "message" in choice, "Missing 'message' in choice"
        assert "content" in choice["message"], "Missing 'content' in message"
        
        answer = choice["message"]["content"]
        usage = result["usage"]
        
        print_result(True, "Chat completion successful")
        print(f"\n  Answer: {answer[:200]}...")
        print(f"  Tokens: {usage['total_tokens']} (prompt: {usage['prompt_tokens']}, completion: {usage['completion_tokens']})")
        print(f"  Time: {elapsed:.2f}s")
        
        if "sources" in result:
            print(f"  Sources: {len(result['sources'])} documents")
        
        return True
        
    except Exception as e:
        print_result(False, f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_conversation_history(config_name):
    """Test chat completion with conversation history."""
    print_section(f"Test 4: Conversation History ({config_name})")
    
    url = f"{BASE_URL}/{config_name}/v1/chat/completions"
    
    payload = {
        "model": config_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is deep learning?"
            },
            {
                "role": "assistant",
                "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers."
            },
            {
                "role": "user",
                "content": "Can you give me an example?"
            }
        ],
        "temperature": 0.7,
        "k": 3
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        
        print_result(True, "Conversation history handled correctly")
        print(f"\n  Answer: {answer[:200]}...")
        
        return True
        
    except Exception as e:
        print_result(False, f"Error: {str(e)}")
        return False


def test_streaming(config_name):
    """Test streaming chat completion."""
    print_section(f"Test 5: Streaming Response ({config_name})")
    
    url = f"{BASE_URL}/{config_name}/v1/chat/completions"
    
    payload = {
        "model": config_name,
        "messages": [
            {
                "role": "user",
                "content": "Explain neural networks briefly."
            }
        ],
        "stream": True,
        "k": 3
    }
    
    try:
        response = requests.post(url, json=payload, stream=True, timeout=30)
        response.raise_for_status()
        
        chunks_received = 0
        content_received = ""
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data_str)
                        chunks_received += 1
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            content_received += content
                    except json.JSONDecodeError:
                        pass
        
        print_result(True, f"Streaming successful ({chunks_received} chunks)")
        print(f"\n  Content: {content_received[:200]}...")
        
        return True
        
    except Exception as e:
        print_result(False, f"Error: {str(e)}")
        return False


def test_list_models(config_name):
    """Test listing models endpoint."""
    print_section(f"Test 6: List Models ({config_name})")
    
    url = f"{BASE_URL}/{config_name}/v1/models"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        result = response.json()
        
        assert "object" in result, "Missing 'object' field"
        assert result["object"] == "list", "Object should be 'list'"
        assert "data" in result, "Missing 'data' field"
        assert len(result["data"]) > 0, "No models in response"
        
        model = result["data"][0]
        assert model["id"] == config_name, f"Model ID should be '{config_name}'"
        
        print_result(True, f"Models endpoint working (found model: {model['id']})")
        
        return True
        
    except Exception as e:
        print_result(False, f"Error: {str(e)}")
        return False


def test_retrieve_model(config_name):
    """Test retrieve model endpoint."""
    print_section(f"Test 7: Retrieve Model ({config_name})")
    
    url = f"{BASE_URL}/{config_name}/v1/models/{config_name}"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        result = response.json()
        
        assert "id" in result, "Missing 'id' field"
        assert result["id"] == config_name, f"Model ID should be '{config_name}'"
        assert "object" in result, "Missing 'object' field"
        assert result["object"] == "model", "Object should be 'model'"
        
        print_result(True, f"Retrieve model endpoint working")
        
        return True
        
    except Exception as e:
        print_result(False, f"Error: {str(e)}")
        return False


def test_invalid_configuration():
    """Test error handling for invalid configuration."""
    print_section("Test 8: Error Handling (Invalid Configuration)")
    
    url = f"{BASE_URL}/invalid_config_name/v1/chat/completions"
    
    payload = {
        "model": "invalid_config_name",
        "messages": [
            {
                "role": "user",
                "content": "Test"
            }
        ]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=5)
        
        if response.status_code == 404:
            print_result(True, "404 error returned for invalid configuration")
            return True
        else:
            print_result(False, f"Expected 404, got {response.status_code}")
            return False
            
    except Exception as e:
        print_result(False, f"Error: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print(" OpenAI-Compatible Chat Completions - Test Suite")
    print("="*80)
    print("\nTesting OpenAI-compatible endpoints for RAG configurations")
    print(f"Base URL: {BASE_URL}")
    
    # Track results
    results = []
    
    # Test 1: Health check
    if not test_health_check():
        print("\n✗ Service is not running. Please start the RAG service first.")
        sys.exit(1)
    results.append(True)
    
    # Test 2: List configurations
    config_names = test_list_configurations()
    if not config_names:
        print("\n✗ No configurations found. Please create at least one configuration.")
        print("  Use: POST /api/v1/configurations")
        sys.exit(1)
    results.append(True)
    
    # Use the first configuration for testing
    test_config = "batch_rl-docs_test" #config_names[0]
    print(f"\nUsing configuration '{test_config}' for tests...")
    
    # Test 3: Simple chat completion
    results.append(test_simple_chat_completion(test_config))
    
    # Test 4: Conversation history
    results.append(test_conversation_history(test_config))
    
    # Test 5: Streaming
    results.append(test_streaming(test_config))
    
    # Test 6: List models
    results.append(test_list_models(test_config))
    
    # Test 7: Retrieve model
    results.append(test_retrieve_model(test_config))
    
    # Test 8: Error handling
    results.append(test_invalid_configuration())
    
    # Summary
    print_section("Test Summary")
    passed = sum(results)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
