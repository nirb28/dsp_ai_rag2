"""
Example script demonstrating the unified OpenAI-compatible endpoint.

This shows how to use /v1/chat/completions with the 'model' parameter
to specify which RAG configuration to use, similar to how LiteLLM works.

Usage:
    python examples/unified_openai_example.py
"""

import requests
import json
from openai import OpenAI


BASE_URL = "http://localhost:9000"


def example_1_unified_endpoint():
    """Example 1: Using the unified /v1/chat/completions endpoint."""
    print("\n" + "="*80)
    print("Example 1: Unified Endpoint with Model Parameter")
    print("="*80)
    
    url = f"{BASE_URL}/v1/chat/completions"
    
    # First configuration
    payload1 = {
        "model": "batch_rl-docs_test",  # Model parameter specifies configuration
        "messages": [
            {
                "role": "user",
                "content": "What is reinforcement learning?"
            }
        ],
        "temperature": 0.7,
        "k": 5
    }
    
    print(f"\nRequest to: {url}")
    print(f"Model (Configuration): {payload1['model']}")
    
    response = requests.post(url, json=payload1)
    response.raise_for_status()
    
    result = response.json()
    print(f"\nAnswer: {result['choices'][0]['message']['content'][:200]}...")
    print(f"Processing time: {result.get('processing_time', 'N/A'):.2f}s")
    
    # Different configuration, same endpoint
    payload2 = {
        "model": "default",  # Different model/configuration
        "messages": [
            {
                "role": "user",
                "content": "What is machine learning?"
            }
        ],
        "temperature": 0.7,
        "k": 5
    }
    
    print(f"\n\nSwitching to different model: {payload2['model']}")
    response = requests.post(url, json=payload2)
    response.raise_for_status()
    
    result = response.json()
    print(f"Answer: {result['choices'][0]['message']['content'][:200]}...")


def example_2_list_models():
    """Example 2: List all available models using /v1/models."""
    print("\n" + "="*80)
    print("Example 2: List Available Models")
    print("="*80)
    
    url = f"{BASE_URL}/v1/models"
    
    print(f"\nRequest to: {url}")
    
    response = requests.get(url)
    response.raise_for_status()
    
    result = response.json()
    
    print(f"\nAvailable models ({len(result['data'])}):")
    for model in result['data']:
        print(f"  - {model['id']}")


def example_3_openai_library():
    """Example 3: Using the official OpenAI Python library."""
    print("\n" + "="*80)
    print("Example 3: Using OpenAI Python Library")
    print("="*80)
    
    try:
        # Create client pointing to unified endpoint
        client = OpenAI(
            base_url=f"{BASE_URL}/v1",
            api_key="dummy-key"  # Not used if security is disabled
        )
        
        print(f"\nUsing OpenAI library with base_url: {BASE_URL}/v1")
        
        # List models
        models = client.models.list()
        print(f"\nAvailable models: {[model.id for model in models.data]}")
        
        # Create completion with first model
        if models.data:
            model_name = models.data[0].id
            print(f"\nCreating completion with model: {model_name}")
            
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": "Explain neural networks briefly."
                    }
                ],
                temperature=0.7
            )
            
            print(f"\nAnswer: {completion.choices[0].message.content[:200]}...")
            print(f"Tokens used: {completion.usage.total_tokens}")
        
    except ImportError:
        print("\nOpenAI library not installed. Install with: pip install openai")


def example_4_streaming():
    """Example 4: Streaming response with unified endpoint."""
    print("\n" + "="*80)
    print("Example 4: Streaming Response")
    print("="*80)
    
    url = f"{BASE_URL}/v1/chat/completions"
    
    payload = {
        "model": "batch_rl-docs_test",
        "messages": [
            {
                "role": "user",
                "content": "What are the key concepts in deep learning?"
            }
        ],
        "stream": True,
        "temperature": 0.7
    }
    
    print(f"\nStreaming from model: {payload['model']}")
    print("Response:")
    print("-" * 80)
    
    response = requests.post(url, json=payload, stream=True)
    response.raise_for_status()
    
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                data_str = line_str[6:]
                if data_str == '[DONE]':
                    print("\n" + "-" * 80)
                    break
                try:
                    chunk = json.loads(data_str)
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        content = delta.get('content', '')
                        if content:
                            print(content, end='', flush=True)
                except json.JSONDecodeError:
                    pass


def example_5_multiple_models():
    """Example 5: Using multiple models with the same endpoint."""
    print("\n" + "="*80)
    print("Example 5: Switching Between Models")
    print("="*80)
    
    url = f"{BASE_URL}/v1/chat/completions"
    
    models = ["batch_rl-docs_test", "default"]
    
    print(f"\nUsing the same endpoint: {url}")
    print("Switching models by changing the 'model' parameter\n")
    
    for model_name in models:
        print(f"Model: {model_name}")
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "What is machine learning?"}],
            "k": 3
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            print(f"  Response: {result['choices'][0]['message']['content'][:100]}...")
        except Exception as e:
            print(f"  Error: {str(e)}")
        print()


def example_6_langchain_integration():
    """Example 6: Using with LangChain."""
    print("\n" + "="*80)
    print("Example 6: LangChain Integration")
    print("="*80)
    
    try:
        from langchain.chat_models import ChatOpenAI
        
        # Using unified endpoint
        llm = ChatOpenAI(
            base_url=f"{BASE_URL}/v1",
            api_key="dummy",
            model="batch_rl-docs_test"
        )
        
        print(f"\nUsing LangChain with unified endpoint")
        print(f"Model: batch_rl-docs_test")
        
        response = llm.invoke("What is policy gradient?")
        print(f"\nResponse: {response.content[:200]}...")
        
    except ImportError:
        print("\nLangChain not installed. Install with: pip install langchain")


def check_service():
    """Check if the service is running."""
    try:
        response = requests.get(f"{BASE_URL}/")
        response.raise_for_status()
        data = response.json()
        
        print("\n" + "="*80)
        print("RAG Service Information")
        print("="*80)
        print(f"Version: {data.get('version', 'N/A')}")
        print(f"\nOpenAI-Compatible Endpoints:")
        print(f"  Chat Completions: {data['openai_compatible_endpoints']['chat_completions']}")
        print(f"  Models List: {data['openai_compatible_endpoints']['models']}")
        print(f"\nAvailable Models: {', '.join(data.get('available_models', []))}")
        
        return True
    except Exception as e:
        print(f"\n✗ Error: Cannot connect to RAG service: {str(e)}")
        print("Please start the service first: python -m app.main")
        return False


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("Unified OpenAI-Compatible Endpoint Examples")
    print("="*80)
    print("\nThis demonstrates the /v1/chat/completions endpoint where the")
    print("'model' parameter specifies which RAG configuration to use.")
    
    if not check_service():
        return
    
    try:
        example_1_unified_endpoint()
        example_2_list_models()
        example_3_openai_library()
        example_4_streaming()
        example_5_multiple_models()
        example_6_langchain_integration()
        
        print("\n" + "="*80)
        print("All examples completed!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
