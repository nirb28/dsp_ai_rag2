#!/usr/bin/env python
"""
Script to fetch the specified number of recent commits from a GitHub repository
and update local files with their contents.
"""

import argparse
import os
import logging
import sys
import subprocess
import tempfile
import shutil
import requests
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_github_url(url):
    """Parse GitHub URL to extract owner and repo name."""
    # Handle SSH format
    if url.startswith('git@github.com:'):
        path = url.split(':')[1]
        if path.endswith('.git'):
            path = path[:-4]
        parts = path.split('/')
        return parts[0], parts[1]
    
    # Handle HTTPS format
    url = url.rstrip('/')
    if url.endswith('.git'):
        url = url[:-4]
    parts = url.split('/')
    return parts[-2], parts[-1]

def get_recent_commits(owner, repo, num_commits, branch='main', github_token=None, proxies=None):
    """
    Get the specified number of recent commits from a GitHub repository.
    
    Args:
        owner: GitHub repository owner
        repo: GitHub repository name
        num_commits: Number of commits to fetch
        branch: Branch to get commits from (default: main)
        github_token: Optional GitHub API token for authenticated requests
        proxies: Optional dictionary mapping protocol to proxy URL
    
    Returns:
        List of commit data dictionaries
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    params = {'sha': branch, 'per_page': num_commits}
    headers = {}
    
    if github_token:
        headers['Authorization'] = f"token {github_token}"
    
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    
    commits = response.json()
    logger.info(f"Retrieved {len(commits)} commits from {owner}/{repo}")
    return commits

def get_file_content_from_commit(owner, repo, commit_sha, file_path, github_token=None, proxies=None):
    """
    Get the content of a file at a specific commit.
    
    Args:
        owner: GitHub repository owner
        repo: GitHub repository name
        commit_sha: Commit SHA
        file_path: Path to the file in the repository
        github_token: Optional GitHub API token for authenticated requests
        proxies: Optional dictionary mapping protocol to proxy URL
    
    Returns:
        File content as bytes
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    params = {'ref': commit_sha}
    headers = {'Accept': 'application/vnd.github.v3.raw'}
    
    if github_token:
        headers['Authorization'] = f"token {github_token}"
    
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    
    return response.content

def get_files_from_commit(owner, repo, commit_sha, github_token=None, proxies=None):
    """
    Get list of files modified in a specific commit.
    
    Args:
        owner: GitHub repository owner
        repo: GitHub repository name
        commit_sha: Commit SHA
        github_token: Optional GitHub API token for authenticated requests
        proxies: Optional dictionary mapping protocol to proxy URL
    
    Returns:
        List of file paths modified in the commit
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}"
    headers = {}
    
    if github_token:
        headers['Authorization'] = f"token {github_token}"
    
    response = requests.get(url, headers=headers, proxies=proxies)
    response.raise_for_status()
    
    commit_data = response.json()
    files = []
    
    for file_info in commit_data.get('files', []):
        # Skip deleted files and renamed files
        if file_info.get('status') in ['removed', 'renamed']:
            continue
        files.append(file_info['filename'])
    
    return files

def update_local_file(local_repo_path, file_path, content):
    """
    Update a local file with the content from GitHub.
    
    Args:
        local_repo_path: Path to the local repository
        file_path: Path of the file within the repository
        content: Content to write to the file
    """
    full_path = os.path.join(local_repo_path, file_path)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    # Write content to file
    with open(full_path, 'wb') as f:
        f.write(content)
    
    logger.info(f"Updated file: {file_path}")

def update_from_commits(github_url, local_repo_path, num_commits, branch='main', github_token=None, proxies=None):
    """
    Update local files based on recent commits from GitHub.
    
    Args:
        github_url: GitHub repository URL
        local_repo_path: Path to the local repository
        num_commits: Number of recent commits to process
        branch: Branch to get commits from (default: main)
        github_token: Optional GitHub API token for authenticated requests
        proxies: Optional dictionary mapping protocol to proxy URL
    """
    try:
        # Parse GitHub URL
        owner, repo = parse_github_url(github_url)
        logger.info(f"Updating files from {owner}/{repo} ({num_commits} recent commits)")
        
        # Get recent commits
        commits = get_recent_commits(owner, repo, num_commits, branch, github_token, proxies)
        
        # Process each commit
        for commit in commits:
            commit_sha = commit['sha']
            commit_message = commit['commit']['message'].split('\n')[0]  # Get first line of commit message
            commit_date = commit['commit']['author']['date']
            
            logger.info(f"Processing commit: {commit_sha[:8]} - {commit_date} - {commit_message}")
            
            # Get files modified in this commit
            modified_files = get_files_from_commit(owner, repo, commit_sha, github_token, proxies)
            logger.info(f"Found {len(modified_files)} modified files")
            
            # Update each file
            for file_path in modified_files:
                try:
                    # Get file content from this commit
                    content = get_file_content_from_commit(owner, repo, commit_sha, file_path, github_token, proxies)
                    
                    # Update local file
                    update_local_file(local_repo_path, file_path, content)
                except Exception as e:
                    logger.warning(f"Failed to update file {file_path}: {str(e)}")
        
        logger.info(f"Successfully updated local files from {num_commits} recent commits")
        
    except Exception as e:
        logger.error(f"Error updating files: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Update local files from recent GitHub commits")
    parser.add_argument("github_url", help="GitHub repository URL (HTTPS or SSH)")
    parser.add_argument("local_repo_path", help="Path to the local repository")
    parser.add_argument("--num-commits", type=int, default=5, help="Number of recent commits to process (default: 5)")
    parser.add_argument("--branch", default="main", help="Branch to get commits from (default: main)")
    parser.add_argument("--github-token", help="GitHub API token for authenticated requests")
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
        
        update_from_commits(
            args.github_url, 
            args.local_repo_path, 
            args.num_commits, 
            args.branch, 
            args.github_token,
            proxies
        )
        logger.info("File update completed successfully")
    except Exception as e:
        logger.error(f"Failed to update files: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
