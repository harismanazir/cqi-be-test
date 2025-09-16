"""
GitHub Repository Integration for Code Quality Intelligence
Handles cloning and analyzing GitHub repositories
"""

import os
import tempfile
import shutil
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import re

try:
    from git import Repo, GitCommandError
    from github import Github, GithubException
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    print("Warning: GitPython not installed. Install with: pip install GitPython PyGithub")


class GitHubRepoAnalyzer:
    """GitHub repository analyzer with intelligent cloning and analysis"""
    
    def __init__(self):
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.github = Github(self.github_token) if self.github_token and GIT_AVAILABLE else None
        
        if not GIT_AVAILABLE:
            raise ImportError("GitPython and PyGithub are required. Install with: pip install GitPython PyGithub")
        
        print(f"[GITHUB] GitHub integration initialized")
        if self.github_token:
            print(f"[GITHUB] Authenticated with GitHub API token")
        else:
            print(f"[GITHUB] Using unauthenticated GitHub API (rate limited)")
    
    async def download_repo(self, repo_url: str, branch: str = "main") -> str:
        """Download GitHub repository to temporary directory"""
        
        print(f"[GITHUB] Downloading repository: {repo_url}")
        print(f"[GITHUB] Branch: {branch}")
        
        # Parse repository information
        owner, repo_name = self._parse_github_url(repo_url)
        print(f"[GITHUB] Repository: {owner}/{repo_name}")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix=f"cqi_github_{repo_name}_")
        print(f"[GITHUB] Temp directory: {temp_dir}")
        
        try:
            # Clone repository
            print(f"[GITHUB] Cloning repository...")
            
            # Try different branch names if main doesn't exist
            branches_to_try = [branch]
            if branch == "main":
                branches_to_try.extend(["master", "develop"])
            
            repo = None
            for branch_name in branches_to_try:
                try:
                    repo = Repo.clone_from(repo_url, temp_dir, branch=branch_name, depth=1)
                    print(f"[GITHUB] Successfully cloned branch: {branch_name}")
                    break
                except GitCommandError as e:
                    if "does not exist" in str(e) and branch_name != branches_to_try[-1]:
                        print(f"[GITHUB] Branch '{branch_name}' not found, trying next...")
                        continue
                    else:
                        raise e
            
            if not repo:
                raise Exception(f"Could not clone repository with any of the branches: {branches_to_try}")
            
            # Get repository statistics
            code_files = self._count_code_files(temp_dir)
            print(f"[GITHUB] Repository cloned successfully")
            print(f"[GITHUB] Found {len(code_files)} code files")
            
            return temp_dir
            
        except Exception as e:
            # Cleanup on failure
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"[GITHUB] Failed to clone repository: {str(e)}")
            raise Exception(f"Failed to clone repository: {str(e)}")
    
    def _parse_github_url(self, url: str) -> Tuple[str, str]:
        """Extract owner/repo from GitHub URL"""
        
        # Clean up URL
        url = url.strip().rstrip('/')
        
        # Handle different URL formats
        patterns = [
            r'https://github\.com/([^/]+)/([^/]+)',
            r'git@github\.com:([^/]+)/([^/]+)\.git',
            r'github\.com/([^/]+)/([^/]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                owner, repo_name = match.groups()
                # Remove .git suffix if present
                repo_name = repo_name.replace('.git', '')
                return owner, repo_name
        
        raise ValueError(f"Invalid GitHub URL format: {url}")
    
    def _count_code_files(self, directory: str) -> List[str]:
        """Count code files in directory using same logic as main analysis"""
        
        # Use same extensions as your existing system
        code_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.cs', 
            '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.clj'
        }
        
        # Use same ignore patterns as your existing system
        ignore_dirs = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'build', 'dist', 'target'}
        ignore_files = {'.gitignore', '.env', '.env.local', 'package-lock.json', 'yarn.lock'}
        
        code_files = []
        
        for root, dirs, files in os.walk(directory):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                if file in ignore_files:
                    continue
                    
                file_path = Path(file)
                if file_path.suffix.lower() in code_extensions:
                    full_path = os.path.join(root, file)
                    code_files.append(full_path)
        
        return sorted(code_files)
    
    async def validate_repository(self, repo_url: str) -> Dict[str, Any]:
        """Validate GitHub repository and get metadata"""
        
        try:
            owner, repo_name = self._parse_github_url(repo_url)
            
            result = {
                "valid": True,
                "owner": owner,
                "repo_name": repo_name,
                "full_name": f"{owner}/{repo_name}"
            }
            
            # Get additional info if GitHub API is available
            if self.github:
                try:
                    repo = self.github.get_repo(f"{owner}/{repo_name}")
                    
                    result.update({
                        "description": repo.description,
                        "language": repo.language,
                        "size_kb": repo.size,
                        "stars": repo.stargazers_count,
                        "forks": repo.forks_count,
                        "open_issues": repo.open_issues_count,
                        "default_branch": repo.default_branch,
                        "last_update": repo.updated_at.isoformat() if repo.updated_at else None,
                        "is_private": repo.private,
                        "is_fork": repo.fork
                    })
                    
                    # Get available branches (limit to first 20)
                    try:
                        branches = [branch.name for branch in repo.get_branches()[:20]]
                        result["branches"] = branches
                    except Exception:
                        result["branches"] = [repo.default_branch]
                        
                except GithubException as e:
                    if e.status == 404:
                        result["valid"] = False
                        result["error"] = "Repository not found or is private"
                    else:
                        result["valid"] = False
                        result["error"] = f"GitHub API error: {str(e)}"
            else:
                # Basic validation without API
                result.update({
                    "description": "GitHub API not available",
                    "branches": ["main", "master"],
                    "default_branch": "main"
                })
            
            return result
            
        except ValueError as e:
            return {
                "valid": False,
                "error": str(e)
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation failed: {str(e)}"
            }
    
    async def get_repository_branches(self, owner: str, repo_name: str) -> List[str]:
        """Get available branches for a repository"""
        
        if not self.github:
            return ["main", "master", "develop"]
        
        try:
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            branches = [branch.name for branch in repo.get_branches()]
            return branches
        except Exception as e:
            print(f"[GITHUB] Could not fetch branches: {str(e)}")
            return ["main", "master", "develop"]
    
    def cleanup_temp_directory(self, temp_dir: str):
        """Clean up temporary directory"""
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"[GITHUB] Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            print(f"[GITHUB] Warning: Could not clean up temp directory: {str(e)}")
    
    def get_repository_stats(self, temp_dir: str) -> Dict[str, Any]:
        """Get basic statistics about the downloaded repository"""
        
        code_files = self._count_code_files(temp_dir)
        
        # Count lines and calculate size
        total_lines = 0
        total_size = 0
        language_stats = {}
        
        for file_path in code_files:
            try:
                file_size = os.path.getsize(file_path)
                total_size += file_size
                
                # Count lines
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = len(f.readlines())
                    total_lines += lines
                
                # Language statistics
                ext = Path(file_path).suffix.lower()
                if ext not in language_stats:
                    language_stats[ext] = {'files': 0, 'lines': 0}
                language_stats[ext]['files'] += 1
                language_stats[ext]['lines'] += lines
                
            except Exception:
                continue
        
        return {
            "total_files": len(code_files),
            "total_lines": total_lines,
            "total_size_bytes": total_size,
            "language_breakdown": language_stats
        }