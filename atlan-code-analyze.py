#!/usr/bin/env python3
"""
Atlan Code Analysis CLI Tool
A comprehensive code quality intelligence system with LangGraph multi-agent analysis

Usage:
    atlan-code-analyze --path="/path/to/code" [options]
    atlan-code-analyze --repo="https://github.com/user/repo" [options]
    atlan-code-analyze --qa --path="/path/to/code"
    atlan-code-analyze --qa --repo="https://github.com/user/repo"
"""

import os
import sys
import asyncio
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing modules
try:
    from main import LangGraphCQI
    from interactive_qa import InteractiveQASession
    from extensions.github_integration import GitHubRepoAnalyzer
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Make sure all required files are in the same directory")
    sys.exit(1)


class AtlanCodeAnalyzer:
    """Main CLI class for Atlan Code Analysis"""
    
    def __init__(self):
        self.temp_dirs = []  # Track temp directories for cleanup
        
    async def analyze_path(self, path: str, detailed: bool = False, agents: Optional[List[str]] = None,
                          enable_rag: bool = True, enable_cache: bool = True, max_files: Optional[int] = None):
        """Analyze local file or directory path using main.py logic"""

        if not os.path.exists(path):
            print(f"[ERROR] Path does not exist: {path}")
            return False

        print(f"[ANALYZING] {path}")
        print("=" * 60)

        # Initialize LangGraph CQI system exactly like main.py
        cqi = LangGraphCQI(
            enable_rag=enable_rag,
            enable_cache=enable_cache,
            use_langgraph=True  # Always use LangGraph for consistency
        )

        try:
            if os.path.isfile(path):
                # Use main.py's exact logic for file analysis
                result = await cqi.analyze_file(path, agents, detailed)
                # Don't print custom summary, let main.py's _print_results handle it
                return True
            else:
                # Use main.py's exact logic for directory analysis
                result = await cqi.analyze_directory(path, agents, detailed, max_parallel=4, max_files=max_files)
                # Don't print custom summary, let main.py's _print_results handle it
                return True

        except KeyboardInterrupt:
            print(f"\n[INTERRUPT] Analysis interrupted by user")
            return False
        except Exception as e:
            print(f"\n[ERROR] Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    async def analyze_github_repo(self, repo_url: str, branch: str = "main", 
                                 detailed: bool = False, agents: Optional[List[str]] = None,
                                 enable_rag: bool = True, enable_cache: bool = True,
                                 max_files: Optional[int] = None):
        """Analyze GitHub repository"""
        
        print(f"🐙 Analyzing GitHub repository: {repo_url}")
        print(f"📝 Branch: {branch}")
        print("=" * 60)
        
        temp_dir = None
        
        try:
            # Initialize GitHub analyzer
            github_analyzer = GitHubRepoAnalyzer()
            
            # Validate repository first
            print("🔍 Validating repository...")
            validation = await github_analyzer.validate_repository(repo_url)
            
            if not validation.get('valid', False):
                print(f"❌ Repository validation failed: {validation.get('error', 'Unknown error')}")
                return False
            
            print(f"✅ Repository validated: {validation['owner']}/{validation['repo_name']}")
            if validation.get('description'):
                print(f"📄 Description: {validation['description']}")
            if validation.get('language'):
                print(f"💻 Primary language: {validation['language']}")
            
            # Download repository
            print("\n📥 Downloading repository...")
            temp_dir = await github_analyzer.download_repo(repo_url, branch)
            self.temp_dirs.append(temp_dir)
            
            # Get repository stats
            repo_stats = github_analyzer.get_repository_stats(temp_dir)
            print(f"📊 Repository stats: {repo_stats['total_files']} files, {repo_stats['total_lines']:,} lines")
            
            # Analyze the downloaded code
            print("\n🧠 Starting LangGraph analysis...")
            cqi = LangGraphCQI(
                enable_rag=enable_rag,
                enable_cache=enable_cache,
                use_langgraph=True
            )
            
            result = await cqi.analyze_directory(
                temp_dir, agents, detailed, max_parallel=4, max_files=max_files
            )
            
            # Let main.py handle the printing
            return True
            
        except Exception as e:
            print(f"[ERROR] GitHub analysis failed: {str(e)}")
            return False
        
        finally:
            # Cleanup temp directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"🧹 Cleaned up temporary directory")
                except Exception as cleanup_error:
                    print(f"[WARNING] Could not clean up temp directory: {cleanup_error}")
    
    async def start_qa_session(self, path: Optional[str] = None, repo_url: Optional[str] = None, 
                              branch: str = "main", enable_rag: bool = True, run_analysis: bool = True):
        """Start interactive Q&A session"""
        
        codebase_path = None
        temp_dir = None
        
        try:
            if repo_url:
                # Handle GitHub repository
                print(f"🐙 Setting up Q&A for GitHub repository: {repo_url}")
                print("=" * 60)
                
                github_analyzer = GitHubRepoAnalyzer()
                
                # Validate and download
                validation = await github_analyzer.validate_repository(repo_url)
                if not validation.get('valid', False):
                    print(f"❌ Repository validation failed: {validation.get('error', 'Unknown error')}")
                    return False
                
                print(f"✅ Repository: {validation['owner']}/{validation['repo_name']}")
                
                print("📥 Downloading repository for Q&A...")
                temp_dir = await github_analyzer.download_repo(repo_url, branch)
                self.temp_dirs.append(temp_dir)
                codebase_path = temp_dir
                
                print(f"📁 Repository downloaded to: {temp_dir}")
                
            elif path:
                # Handle local path
                if not os.path.exists(path):
                    print(f"❌ Error: Path does not exist: {path}")
                    return False
                
                print(f"📁 Setting up Q&A for local path: {path}")
                print("=" * 60)
                codebase_path = path
            
            else:
                print("❌ Error: Either --path or --repo must be specified for Q&A")
                return False
            
            # Initialize Q&A session
            print("🧠 Initializing interactive Q&A system...")
            session = InteractiveQASession(
                codebase_path=codebase_path,
                enable_rag=enable_rag,
                run_analysis=run_analysis
            )
            
            await session.initialize()
            await session.start_session()
            
            return True
            
        except KeyboardInterrupt:
            print("\n👋 Q&A session ended by user")
            return True
        except Exception as e:
            print(f"❌ Q&A session failed: {str(e)}")
            return False
        
        finally:
            # Cleanup temp directory for GitHub repos
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"\n🧹 Cleaned up temporary directory")
                except Exception as cleanup_error:
                    print(f"[WARNING] Could not clean up temp directory: {cleanup_error}")
    
    
    def cleanup(self):
        """Cleanup temporary directories"""
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception:
                pass


def create_parser():
    """Create argument parser"""
    
    parser = argparse.ArgumentParser(
        prog='atlan-code-analyze',
        description='Atlan Code Analysis - AI-powered code quality intelligence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze local directory
  atlan-code-analyze --path="/path/to/code" --detailed
  
  # Analyze GitHub repository
  atlan-code-analyze --repo="https://github.com/user/repo" --branch="main"
  
  # Start Q&A session for local code
  atlan-code-analyze --qa --path="/path/to/code"
  
  # Start Q&A session for GitHub repository
  atlan-code-analyze --qa --repo="https://github.com/user/repo"
  
  # Advanced analysis with specific agents
  atlan-code-analyze --path="." --agents="security,performance" --rag --detailed
        """
    )
    
    # Source specification (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--path', type=str, help='Local file or directory path to analyze')
    source_group.add_argument('--repo', type=str, help='GitHub repository URL to analyze')
    
    # Mode selection
    parser.add_argument('--qa', action='store_true', help='Start interactive Q&A session')
    
    # GitHub options
    parser.add_argument('--branch', type=str, default='main', help='GitHub branch to analyze (default: main)')
    
    # Analysis options
    parser.add_argument('--detailed', action='store_true', help='Show detailed issues report')
    parser.add_argument('--agents', type=str, help='Comma-separated list of agents (security,performance,complexity,documentation)')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to analyze')
    
    # System options
    parser.add_argument('--rag', action='store_true', default=True, help='Enable RAG system (default: True)')
    parser.add_argument('--no-rag', action='store_true', help='Disable RAG system')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    
    # Utility commands
    parser.add_argument('--version', action='version', version='atlan-code-analyze 1.0.0')
    
    return parser


async def main():
    """Main CLI entry point"""
    
    # ASCII Art Banner
    print("""
================================================================
                    ATLAN CODE ANALYZER
           AI-Powered Code Quality Intelligence
                  Powered by LangGraph
================================================================
    """)
    
    parser = create_parser()
    args = parser.parse_args()
    
    # Parse agents - but use None to match main.py behavior
    # main.py defaults to all agents when selected_agents is None
    selected_agents = None
    if args.agents:
        requested_agents = [agent.strip() for agent in args.agents.split(',')]
        valid_agents = {'security', 'performance', 'complexity', 'documentation'}
        invalid_agents = [a for a in requested_agents if a not in valid_agents]
        if invalid_agents:
            print(f"[ERROR] Invalid agents: {', '.join(invalid_agents)}")
            print(f"[INFO] Valid agents: {', '.join(sorted(valid_agents))}")
            return
        # Always use None to let main.py's logic handle agent selection
        selected_agents = None
    
    # Handle RAG settings
    enable_rag = args.rag and not args.no_rag
    enable_cache = not args.no_cache
    
    # Initialize analyzer
    analyzer = AtlanCodeAnalyzer()
    
    try:
        success = False
        
        if args.qa:
            # Q&A mode
            success = await analyzer.start_qa_session(
                path=args.path,
                repo_url=args.repo,
                branch=args.branch,
                enable_rag=enable_rag,
                run_analysis=True
            )
        
        elif args.path:
            # Local path analysis
            success = await analyzer.analyze_path(
                path=args.path,
                detailed=args.detailed,
                agents=selected_agents,
                enable_rag=enable_rag,
                enable_cache=enable_cache,
                max_files=args.max_files
            )
        
        elif args.repo:
            # GitHub repository analysis
            success = await analyzer.analyze_github_repo(
                repo_url=args.repo,
                branch=args.branch,
                detailed=args.detailed,
                agents=selected_agents,
                enable_rag=enable_rag,
                enable_cache=enable_cache,
                max_files=args.max_files
            )
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
        sys.exit(1)
    finally:
        analyzer.cleanup()


if __name__ == '__main__':
    asyncio.run(main())