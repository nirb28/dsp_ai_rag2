"""
Example script demonstrating OpenAI-compatible chat completions with RAG.

This example shows how to use the OpenAI-compatible API endpoints for each
RAG configuration. These endpoints can be used as drop-in replacements for
OpenAI's chat completions API.

Usage:
    python examples/openai_chat_completions_example.py
"""

import requests
import json
import sys
from typing import Optional


BASE_URL = "http://localhost:9000"


def create_chat_completion(
    configuration_name: str,
    messages: list,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    k: int = 5,
    include_sources: bool = True,
    authorization: Optional[str] = None
):
    """
    Create a chat completion using the OpenAI-compatible endpoint.
    
    Args:
        configuration_name: Name of the RAG configuration to use
        messages: List of message dictionaries with 'role' and 'content'
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens to generate
        stream: Whether to stream the response
        k: Number of documents to retrieve
        include_sources: Whether to include source documents
        authorization: Optional Bearer token for authentication
    
    Returns:
        Response dictionary
    """
    url = f"{BASE_URL}/{configuration_name}/v1/chat/completions"
    
    payload = {
        "model": configuration_name,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
        "k": k,
        "include_sources": include_sources
    }
    
    if max_tokens:
        payload["max_tokens"] = max_tokens
    
    headers = {
        "Content-Type": "application/json"
    }
    
    if authorization:
        headers["Authorization"] = f"Bearer {authorization}"
    
    print(f"\n{'='*80}")
    print(f"Sending request to: {url}")
    print(f"Configuration: {configuration_name}")
    print(f"Messages: {json.dumps(messages, indent=2)}")
    print(f"{'='*80}\n")
    
    if stream:
        # Handle streaming response
        response = requests.post(url, json=payload, headers=headers, stream=True)
        response.raise_for_status()
        
        print("Streaming response:")
        print("-" * 80)
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]  # Remove 'data: ' prefix
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
        
        return None
    else:
        # Handle non-streaming response
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        
        print("Response:")
        print("-" * 80)
        print(f"Answer: {result['choices'][0]['message']['content']}")
        print(f"\nProcessing Time: {result.get('processing_time', 'N/A'):.2f}s")
        print(f"Tokens Used: {result['usage']['total_tokens']} "
              f"(prompt: {result['usage']['prompt_tokens']}, "
              f"completion: {result['usage']['completion_tokens']})")
        
        if include_sources and 'sources' in result:
            print(f"\nSources ({len(result['sources'])} documents):")
            for i, source in enumerate(result['sources'][:3], 1):  # Show first 3
                metadata = source.get('metadata', {})
                print(f"  {i}. {metadata.get('filename', 'Unknown')} "
                      f"(score: {source.get('score', 'N/A'):.3f})")
        
        print("-" * 80)
        
        return result


def example_simple_query():
    """Example 1: Simple query without conversation history."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Simple Query")
    print("="*80)
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based on provided context."
        },
        {
            "role": "user",
            "content": "What is machine learning?"
        }
    ]
    
    create_chat_completion(
        configuration_name="default",
        messages=messages,
        temperature=0.7,
        k=5,
        include_sources=True
    )


def example_conversation_history():
    """Example 2: Query with conversation history."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Query with Conversation History")
    print("="*80)
    
    messages = [
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
            "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers..."
        },
        {
            "role": "user",
            "content": "Can you give me some examples of its applications?"
        }
    ]
    
    create_chat_completion(
        configuration_name="default",
        messages=messages,
        temperature=0.7,
        k=5,
        include_sources=True
    )


def example_streaming():
    """Example 3: Streaming response."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Streaming Response")
    print("="*80)
    
    messages = [
        {
            "role": "user",
            "content": "Explain neural networks in simple terms."
        }
    ]
    
    create_chat_completion(
        configuration_name="default",
        messages=messages,
        temperature=0.7,
        stream=True,
        k=5,
        include_sources=False
    )


def example_with_metadata_filter():
    """Example 4: Query with metadata filtering."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Query with Metadata Filter")
    print("="*80)
    
    url = f"{BASE_URL}/default/v1/chat/completions"
    
    payload = {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": "What are the key concepts?"
            }
        ],
        "temperature": 0.7,
        "k": 5,
        "filter": {
            "category": "tutorial"
        },
        "include_sources": True
    }
    
    headers = {"Content-Type": "application/json"}
    
    print(f"\nSending request with metadata filter: {payload['filter']}")
    
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    
    result = response.json()
    print(f"\nAnswer: {result['choices'][0]['message']['content']}")
    print(f"Sources: {len(result.get('sources', []))} documents")


def example_using_openai_library():
    """Example 5: Using the official OpenAI Python library."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Using OpenAI Python Library")
    print("="*80)
    
    try:
        from openai import OpenAI
        
        # Create client pointing to your RAG service
        client = OpenAI(
            base_url=f"{BASE_URL}/default/v1",
            api_key="dummy-key"  # Not used if security is disabled
        )
        
        print("\nUsing OpenAI library to call RAG service...")
        
        completion = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "What is reinforcement learning?"
                }
            ],
            temperature=0.7
        )
        
        print(f"\nAnswer: {completion.choices[0].message.content}")
        print(f"Tokens: {completion.usage.total_tokens}")
        
    except ImportError:
        print("\nOpenAI library not installed. Install with: pip install openai")
        print("This example shows how you can use the official OpenAI library")
        print("with your RAG service as a drop-in replacement.")


def list_available_configurations():
    """List all available configurations with OpenAI endpoints."""
    print("\n" + "="*80)
    print("Available Configurations with OpenAI-Compatible Endpoints")
    print("="*80)
    
    try:
        response = requests.get(f"{BASE_URL}/")
        response.raise_for_status()
        data = response.json()
        
        endpoints = data.get("openai_compatible_endpoints", {})
        
        if endpoints:
            print("\nConfiguration Name → Endpoint URL")
            print("-" * 80)
            for config_name, endpoint in endpoints.items():
                full_url = f"{BASE_URL}{endpoint}"
                print(f"  {config_name:20} → {full_url}")
        else:
            print("\nNo configurations found.")
            
    except Exception as e:
        print(f"\nError listing configurations: {str(e)}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("OpenAI-Compatible Chat Completions with RAG - Examples")
    print("="*80)
    print("\nMake sure the RAG service is running on http://localhost:9000")
    print("and you have at least one configuration set up.")
    
    # List available configurations
    list_available_configurations()
    
    # Check if service is running
    try:
        response = requests.get(f"{BASE_URL}/api/v1/health")
        response.raise_for_status()
        print("\n✓ RAG service is running")
    except Exception as e:
        print(f"\n✗ Error: Cannot connect to RAG service: {str(e)}")
        print("Please start the service first.")
        sys.exit(1)
    
    # Run examples
    try:
        example_simple_query()
        example_conversation_history()
        example_streaming()
        example_with_metadata_filter()
        example_using_openai_library()
        
        print("\n" + "="*80)
        print("All examples completed!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
