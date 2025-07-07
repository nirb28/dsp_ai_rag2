#!/usr/bin/env python
"""
Script to download a GitHub project and extract it to a specified destination.
"""

import argparse
import os
import shutil
import tempfile
import logging
import sys
import requests
from zipfile import ZipFile
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_github_url(url):
    """
    Parse GitHub URL to extract owner and repo name.
    Supports formats:
    - https://github.com/owner/repo
    - https://github.com/owner/repo.git
    - git@github.com:owner/repo.git
    """
    if url.startswith('git@'):
        # Handle SSH format
        parts = url.split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid GitHub SSH URL: {url}")
        path = parts[1]
        # Remove .git suffix if present
        if path.endswith('.git'):
            path = path[:-4]
        # Split into owner and repo
        owner_repo = path.split('/')
        if len(owner_repo) != 2:
            raise ValueError(f"Cannot extract owner and repo from: {path}")
        return owner_repo[0], owner_repo[1]
    else:
        # Handle HTTPS format
        parsed_url = urlparse(url)
        if parsed_url.netloc != 'github.com':
            raise ValueError(f"URL is not a GitHub URL: {url}")
        
        path = parsed_url.path.lstrip('/')
        # Remove .git suffix if present
        if path.endswith('.git'):
            path = path[:-4]
        
        # Split into owner and repo
        owner_repo = path.split('/')
        if len(owner_repo) < 2:
            raise ValueError(f"Cannot extract owner and repo from: {path}")
        
        return owner_repo[0], owner_repo[1]

def download_project(github_url, destination, branch='main', proxies=None):
    """
    Download a GitHub project and extract it to the specified destination.
    
    Args:
        github_url: URL of the GitHub repository
        destination: Directory where the project should be extracted
        branch: Branch to download (default: main)
        proxies: Optional dictionary mapping protocol to proxy URL
    """
    try:
        # Parse the GitHub URL
        owner, repo = parse_github_url(github_url)
        logger.info(f"Downloading repository {owner}/{repo} (branch: {branch})")
        
        # Create the download URL for the zip archive
        zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
        
        # Create a temporary directory to store the downloaded zip
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "repo.zip")
            
            # Download the zip file
            logger.info(f"Downloading from {zip_url}")
            if proxies:
                logger.info(f"Using proxies: {proxies}")
            response = requests.get(zip_url, stream=True, proxies=proxies)
            response.raise_for_status()
            
            # Save the zip file
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info("Download complete. Extracting files...")
            
            # Extract the zip file
            with ZipFile(zip_path, 'r') as zip_ref:
                # The zip file contains a top-level directory with the name format: {repo}-{branch}
                zip_root_dir = zip_ref.namelist()[0].split('/')[0]
                
                # Extract to temp directory first
                zip_ref.extractall(temp_dir)
                
                # Create destination directory if it doesn't exist
                os.makedirs(destination, exist_ok=True)
                
                # Move contents from the zip's root directory to the destination
                extracted_dir = os.path.join(temp_dir, zip_root_dir)
                
                # List all files and directories in the extracted directory
                for item in os.listdir(extracted_dir):
                    source_path = os.path.join(extracted_dir, item)
                    dest_path = os.path.join(destination, item)
                    
                    # Handle existing files/directories at the destination
                    if os.path.exists(dest_path):
                        if os.path.isdir(dest_path):
                            shutil.rmtree(dest_path)
                        else:
                            os.remove(dest_path)
                    
                    # Move the item to the destination
                    shutil.move(source_path, dest_path)
            
            logger.info(f"Successfully extracted project to {destination}")
            
    except Exception as e:
        logger.error(f"Error downloading project: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Download a GitHub project")
    parser.add_argument("github_url", help="GitHub repository URL (HTTPS or SSH)")
    parser.add_argument("destination", help="Destination directory to extract the project")
    parser.add_argument("--branch", default="main", help="Branch to download (default: main)")
    parser.add_argument("--http-proxy", help="HTTP proxy server (e.g., http://proxy.example.com:8080)")
    parser.add_argument("--https-proxy", help="HTTPS proxy server (e.g., https://proxy.example.com:8080)")
    
    args = parser.parse_args()
    
    try:
        # Set up proxies if provided
        proxies = {}
        if args.http_proxy:
            proxies['http'] = args.http_proxy
        if args.https_proxy:
            proxies['https'] = args.https_proxy
        
        # Use empty dict if no proxies were provided
        proxies = proxies if proxies else None
        
        download_project(args.github_url, args.destination, args.branch, proxies)
        logger.info("Project download and extraction completed successfully")
    except Exception as e:
        logger.error(f"Failed to download and extract project: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
