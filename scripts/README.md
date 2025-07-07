# RAG Project Scripts

This folder contains utility scripts for managing and syncing the RAG project.

## Script Descriptions

### 1. `download_project.py`

Downloads a GitHub project and extracts it to a specified destination directory.

**Usage:**
```bash
python download_project.py <github_url> <destination_directory> [--branch <branch_name>] [--http-proxy <http_proxy>] [--https-proxy <https_proxy>]
```

**Arguments:**
- `github_url`: The GitHub repository URL (HTTPS or SSH format)
- `destination_directory`: Directory where the project should be extracted
- `--branch`: Branch to download (default: main)
- `--http-proxy`: HTTP proxy server (e.g., http://proxy.example.com:8080)
- `--https-proxy`: HTTPS proxy server (e.g., https://proxy.example.com:8080)

**Example:**
```bash
python download_project.py https://github.com/username/dsp_ai_rag2.git D:\projects\rag_project
python download_project.py git@github.com:username/dsp_ai_rag2.git D:\projects\rag_project --branch develop
```

### 2. `update_from_commits.py`

Updates local files based on recent commits from GitHub. Fetches a configurable number of recent commits and applies their changes to your local repository.

**Usage:**
```bash
python update_from_commits.py <github_url> <local_repo_path> [--num-commits <number>] [--branch <branch_name>] [--github-token <token>] [--http-proxy <http_proxy>] [--https-proxy <https_proxy>]
```

**Arguments:**
- `github_url`: The GitHub repository URL (HTTPS or SSH format)
- `local_repo_path`: Path to the local repository that should be updated
- `--num-commits`: Number of recent commits to process (default: 5)
- `--branch`: Branch to get commits from (default: main)
- `--github-token`: Optional GitHub API token for authenticated requests (helps avoid rate limits)
- `--http-proxy`: HTTP proxy server (e.g., http://proxy.example.com:8080)
- `--https-proxy`: HTTPS proxy server (e.g., https://proxy.example.com:8080)

**Example:**
```bash
python update_from_commits.py https://github.com/username/dsp_ai_rag2.git D:\projects\rag_project --num-commits 10
```

## Requirements

Both scripts require the following Python packages:
- requests
- shutil
- zipfile
- tempfile
- logging
- argparse

You can install them with:
```bash
pip install requests
```
(The other packages are part of the Python standard library)

## Notes

- These scripts use the GitHub API, which has rate limits. For heavy usage, provide a GitHub API token using the `--github-token` argument.
- The scripts handle both HTTPS and SSH GitHub URL formats.
- When downloading a project, existing files at the destination will be overwritten.
- When updating from commits, only files modified in those commits will be affected.
- If you're behind a corporate firewall or need to use a proxy, both scripts support HTTP and HTTPS proxies via command line arguments.

## Using with Proxies

Both scripts support proxy configuration for environments where direct internet access is restricted:

```bash
# Example with HTTP proxy
python download_project.py https://github.com/username/repo.git ./my-project --http-proxy http://proxy.company.com:8080

# Example with HTTPS proxy
python update_from_commits.py https://github.com/username/repo.git ./my-project --https-proxy https://proxy.company.com:8080
```

You can also set environment variables for proxies that will be respected by the Python requests library:

```bash
# On Windows
set HTTP_PROXY=http://proxy.company.com:8080
set HTTPS_PROXY=https://proxy.company.com:8080

# On Linux/Mac
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=https://proxy.company.com:8080
```
