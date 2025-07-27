#!/usr/bin/env python
"""
Example script to demonstrate LoRA fine-tuning with the RAG API
This script shows how to:
1. Create a fine-tuning job
2. Monitor its progress
3. Generate text with the fine-tuned model
"""
import requests
import json
import time
import sys
from pprint import pprint

# Base URL for the API
BASE_URL = "http://localhost:9001/api/v1/lora"

def print_section(title):
    """Print a section title"""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def check_response(response, exit_on_error=True):
    """Check if response is successful, print details if not"""
    try:
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        print(f"Response: {response.text}")
        if exit_on_error:
            sys.exit(1)
        return None

def create_lora_job():
    """Create a LoRA fine-tuning job"""
    print_section("Creating LoRA Fine-Tuning Job")
    
    # Sample training data
    training_data = [
        {"prompt": "What is machine learning?", 
         "completion": "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data."},
        {"prompt": "Explain deep learning", 
         "completion": "Deep learning is a subset of machine learning that uses neural networks with many layers (hence 'deep') to analyze various factors of data."},
        {"prompt": "What is natural language processing?", 
         "completion": "Natural language processing (NLP) is a field of AI that gives machines the ability to read, understand, and derive meaning from human languages."},
        {"prompt": "Define computer vision", 
         "completion": "Computer vision is an AI field that trains computers to interpret and understand visual information from the world, such as images and videos."},
        {"prompt": "What is reinforcement learning?", 
         "completion": "Reinforcement learning is a machine learning training method based on rewarding desired behaviors and punishing undesired ones."}
    ]
    
    # LoRA job request
    job_request = {
        "name": "AI-Concepts-LoRA",
        "config": {
            "base_model": "meta-llama/Llama-2-7b-hf",  # This is a placeholder, use a model you have access to
            "adapter_name": "ai_concepts_adapter",
            "training_data": training_data,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "batch_size": 4,
            "learning_rate": 3e-4,
            "num_epochs": 3,
            "sequence_length": 512,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "use_fp16": False,
            "use_bf16": False,
            "use_int8": True
        }
    }
    
    response = requests.post(f"{BASE_URL}/jobs", json=job_request)
    job_data = check_response(response)
    
    print("Job created successfully!")
    pprint(job_data)
    return job_data["job_id"]

def monitor_job_progress(job_id):
    """Monitor the progress of a job until it's completed or failed"""
    print_section(f"Monitoring Job Progress (ID: {job_id})")
    
    job_status = None
    poll_interval = 10  # seconds
    max_polls = 60  # 10 minutes max
    polls = 0
    
    print("Polling job status (this may take several minutes)...")
    
    while polls < max_polls:
        response = requests.get(f"{BASE_URL}/jobs/{job_id}")
        job_data = check_response(response, exit_on_error=False)
        
        if not job_data:
            print("Failed to get job status, retrying...")
            time.sleep(poll_interval)
            polls += 1
            continue
        
        status = job_data["status"]
        progress = job_data["progress"]
        
        if job_status != status or progress["current_step"] != progress.get("last_reported_step", -1):
            job_status = status
            progress["last_reported_step"] = progress.get("current_step", 0)
            print(f"\nStatus: {status}")
            print(f"Progress: {progress.get('current_step', 0)}/{progress.get('total_steps', '?')} steps")
            print(f"Loss: {progress.get('loss', 'N/A')}")
            print(f"Learning Rate: {progress.get('learning_rate', 'N/A')}")
            if progress.get("message"):
                print(f"Message: {progress['message']}")
                
        if status in ["COMPLETED", "FAILED", "CANCELED"]:
            print(f"\nJob {status.lower()}!")
            if status == "COMPLETED":
                print("Fine-tuning complete! The adapter is now available for generation.")
            else:
                print(f"Job did not complete successfully. Status: {status}")
            return job_data
            
        # Simple spinning indicator
        sys.stdout.write(".")
        sys.stdout.flush()
        
        time.sleep(poll_interval)
        polls += 1
    
    print("\nTimeout reached while monitoring job.")
    return None

def list_jobs():
    """List all jobs"""
    print_section("Listing All Jobs")
    
    response = requests.get(f"{BASE_URL}/jobs")
    jobs_data = check_response(response)
    
    print(f"Total jobs: {jobs_data['total_count']}")
    for job in jobs_data["jobs"]:
        print(f"- {job['id']}: {job['name']} (Status: {job['status']})")
    
    return jobs_data

def list_adapters():
    """List all available adapters"""
    print_section("Listing Available Adapters")
    
    response = requests.get(f"{BASE_URL}/adapters")
    adapters = check_response(response)
    
    print(f"Available adapters: {len(adapters)}")
    for adapter in adapters:
        print(f"- {adapter}")
    
    return adapters

def generate_text(adapter_name):
    """Generate text using a fine-tuned model"""
    print_section(f"Generating Text with Adapter: {adapter_name}")
    
    generation_request = {
        "adapter_name": adapter_name,
        "prompt": "What is the difference between machine learning and deep learning?",
        "max_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50
    }
    
    response = requests.post(f"{BASE_URL}/generate", json=generation_request)
    result = check_response(response)
    
    print("\nPrompt:")
    print(generation_request["prompt"])
    print("\nGenerated Text:")
    print(result["generated_text"])
    
    return result

def cancel_job(job_id):
    """Cancel a running job"""
    print_section(f"Canceling Job (ID: {job_id})")
    
    response = requests.post(f"{BASE_URL}/jobs/{job_id}/cancel")
    result = check_response(response, exit_on_error=False)
    
    if result:
        print("Job canceled successfully!")
        pprint(result)
    
    return result

def delete_job(job_id):
    """Delete a job"""
    print_section(f"Deleting Job (ID: {job_id})")
    
    response = requests.delete(f"{BASE_URL}/jobs/{job_id}")
    result = check_response(response, exit_on_error=False)
    
    if result:
        print("Job deleted successfully!")
        pprint(result)
    
    return result

def delete_adapter(adapter_name):
    """Delete an adapter"""
    print_section(f"Deleting Adapter: {adapter_name}")
    
    response = requests.delete(f"{BASE_URL}/adapters/{adapter_name}")
    result = check_response(response, exit_on_error=False)
    
    if result:
        print("Adapter deleted successfully!")
        pprint(result)
    
    return result

if __name__ == "__main__":
    try:
        # Step 1: Create a fine-tuning job
        job_id = create_lora_job()
        
        # Step 2: Monitor its progress
        job_data = monitor_job_progress(job_id)
        
        # Only proceed with generation if job completed successfully
        if job_data and job_data["status"] == "COMPLETED":
            # Step 3: List all available adapters
            adapters = list_adapters()
            
            # Step 4: Generate text with the fine-tuned model
            if adapters and "ai_concepts_adapter" in adapters:
                generate_text("ai_concepts_adapter")
            else:
                print("Adapter not found. Cannot generate text.")
        
        # List all jobs for reference
        list_jobs()
        
        # Uncomment these lines to test deletion functionality
        # delete_job(job_id)
        # delete_adapter("ai_concepts_adapter")
        
    except KeyboardInterrupt:
        print("\nOperation canceled by user.")
        sys.exit(0)
