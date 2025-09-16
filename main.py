#!/usr/bin/env python3
"""
LangGraph Multi-Agent Code Quality Intelligence (CQI) - Main Entry Point
Advanced AI-powered code analysis with intelligent workflow orchestration
"""

import os
import sys
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import LangGraph workflow
from workflow.graph import LangGraphMultiAgentAnalyzer
from agents.base_agent import LanguageDetector
from rate_limit_manager import rate_limit_manager

# Import legacy analyzer for fallback
from llm_multi_agent_analyzer import LLMMultiAgentAnalyzer

class LangGraphCQI:
    """Main LangGraph Code Quality Intelligence System"""
    
    def __init__(self, enable_rag: bool = True, enable_cache: bool = True, use_langgraph: bool = True):
        self.enable_rag = enable_rag
        self.enable_cache = enable_cache
        self.use_langgraph = use_langgraph
        
        print("LangGraph Multi-Agent Code Quality Intelligence")
        print("AI-powered code analysis with intelligent workflow orchestration")
        print("Powered by LangGraph + Groq API")
        print("=" * 70)
        
        if use_langgraph:
            print("[LANGGRAPH] Initializing LangGraph workflow engine...")
            self.analyzer = LangGraphMultiAgentAnalyzer(
                enable_rag=enable_rag,
                enable_cache=enable_cache
            )
            print("[LANGGRAPH] LangGraph multi-agent system ready!")
        else:
            print("[LEGACY] Initializing legacy async analyzer...")
            self.analyzer = LLMMultiAgentAnalyzer(
                enable_rag=enable_rag,
                enable_cache=enable_cache
            )
            print("[LEGACY] Legacy multi-agent system ready!")
    
    async def analyze_file_async(self, file_path: str, detailed: bool = True, rag: bool = True) -> Dict:
        """Async wrapper for file analysis - for FastAPI integration"""
        try:
            # Use the async analyze_file method directly (no need for executor)
            result = await self.analyze_file(file_path, detailed=detailed)
            return result
            
        except Exception as e:
            return {
                "file_path": file_path,
                "error": str(e),
                "success": False,
                "all_issues": []
            }
    
    async def analyze_directory_async(self, directory_path: str, max_files: int = 10, detailed: bool = True) -> Dict:
        """Async wrapper for directory analysis - for FastAPI integration"""
        try:
            loop = asyncio.get_event_loop()
            
            if self.use_langgraph:
                result = await loop.run_in_executor(
                    None,
                    lambda: self.analyzer.analyze_directory(directory_path, max_files=max_files, detailed=detailed)
                )
            else:
                result = await loop.run_in_executor(
                    None,
                    lambda: self.analyzer.analyze_directory(directory_path, max_files=max_files, detailed=detailed)
                )
            
            return result
            
        except Exception as e:
            return {
                "directory_path": directory_path,
                "error": str(e),
                "success": False,
                "results": []
            }
    
    async def analyze_file(self, file_path: str, selected_agents: Optional[List[str]] = None, 
                          detailed: bool = False) -> Dict[str, Any]:
        """Analyze a single file"""
        
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code_content = f.read()
        except Exception as e:
            return {"error": f"Failed to read file: {e}"}
        
        # Detect language
        language = LanguageDetector.detect_language(file_path)
        
        print(f"\n[ANALYZING] {os.path.basename(file_path)}")
        print(f"[LANGUAGE] {language.title()}")
        print(f"[SIZE] {len(code_content):,} characters")
        
        if self.use_langgraph:
            # Use LangGraph workflow
            result = await self.analyzer.analyze_file(
                file_path=file_path,
                code_content=code_content,
                language=language,
                selected_agents=selected_agents
            )
        else:
            # Use legacy analyzer
            result = await self.analyzer.analyze_file(file_path, selected_agents)
        
        # Normalize result format for consistent reporting
        normalized_result = self._normalize_result_format(result, code_content)
        
        # Print results
        self._print_results(normalized_result, detailed)
        return normalized_result
    
    def _normalize_result_format(self, result: Dict[str, Any], code_content: str) -> Dict[str, Any]:
        """Normalize result format to ensure consistent reporting"""
        
        # If result is already in the expected format, return as-is
        if 'total_lines' in result and 'total_issues' in result:
            return result
        
        # Extract basic info
        file_path = result.get('file_path', '')
        language = result.get('language', 'unknown')
        
        # Calculate lines of code
        lines_of_code = len([line for line in code_content.split('\n') if line.strip()])
        
        # Extract issues and aggregate
        all_issues = result.get('all_issues', [])
        
        # Count issues by severity
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for issue in all_issues:
            severity = issue.get('severity', 'low').lower()
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        # Extract agent performance data
        agent_performance = {}
        if 'agent_performance' in result:
            agent_performance = result['agent_performance']
        elif 'agent_results' in result:
            # Convert from agent_results format
            for agent_name, agent_result in result.get('agent_results', {}).items():
                if hasattr(agent_result, 'issues'):
                    agent_performance[agent_name] = {
                        'issues_found': len(agent_result.issues),
                        'processing_time': getattr(agent_result, 'processing_time', 0.0),
                        'confidence': getattr(agent_result, 'confidence', 0.8),
                        'status': getattr(agent_result, 'status', 'completed')
                    }
                else:
                    # Handle dict format
                    agent_performance[agent_name] = {
                        'issues_found': len(agent_result.get('issues', [])),
                        'processing_time': agent_result.get('processing_time', 0.0),
                        'confidence': agent_result.get('confidence', 0.8),
                        'status': 'completed'
                    }
        
        # Debug LLM calls
        llm_calls_debug = result.get('llm_calls', result.get('total_llm_calls', 0))
        print(f"[MAIN] Debug LLM calls: llm_calls={result.get('llm_calls')}, total_llm_calls={result.get('total_llm_calls')}, final={llm_calls_debug}")

        # Build normalized result
        normalized = {
            'file_path': file_path,
            'language': language,
            'total_lines': lines_of_code,
            'total_issues': len(all_issues),
            'issues_by_severity': severity_counts,
            'agent_performance': agent_performance,
            'all_issues': all_issues,
            'processing_time': result.get('processing_time', result.get('total_processing_time', 0.0)),
            'total_tokens': result.get('total_tokens', 0),
            'total_llm_calls': result.get('llm_calls', result.get('total_llm_calls', 0)),
            'workflow_engine': result.get('workflow_engine', 'langgraph' if self.use_langgraph else 'legacy'),
            'completed_agents': result.get('completed_agents', []),
            'failed_agents': result.get('failed_agents', []),
            'cache_hit': result.get('cache_hit', False)
        }
        
        return normalized
    
    async def analyze_directory(self, directory_path: str, selected_agents: Optional[List[str]] = None,
                               detailed: bool = False, max_parallel: int = 4, max_files: Optional[int] = None) -> Dict[str, Any]:
        """Analyze all files in a directory"""
        
        if not os.path.exists(directory_path):
            return {"error": f"Directory not found: {directory_path}"}
        
        print(f"\n[DIRECTORY] ANALYZING: {directory_path}")
        
        if self.use_langgraph:
            # For directory analysis, we'll process files individually through LangGraph
            # This maintains the workflow benefits while handling multiple files
            code_files = self._discover_files(directory_path)
            
            if max_files:
                code_files = code_files[:max_files]
                print(f"[LIMIT] Limited to {max_files} files")
            
            print(f"[FILES] Found {len(code_files)} code files")
            
            # Check rate limits and adjust analysis scope if needed
            if selected_agents is None:
                selected_agents = ['security', 'complexity', 'performance', 'documentation']
            
            avg_file_size = sum(os.path.getsize(f) for f in code_files) // len(code_files) if code_files else 1000
            can_proceed, reason = rate_limit_manager.can_proceed_with_analysis(
                len(code_files), len(selected_agents), avg_file_size
            )
            
            if not can_proceed:
                print(f"[RATE-LIMIT] {reason}")
                fallback_agents, strategy = rate_limit_manager.get_fallback_strategy(selected_agents)
                print(f"[FALLBACK] {strategy}")
                print(f"[FALLBACK] Using agents: {', '.join(fallback_agents)}")
                selected_agents = fallback_agents
                
                # Apply conservative limits
                limits = rate_limit_manager.get_conservative_limits()
                max_parallel = min(max_parallel, limits['max_parallel_files'])
                print(f"[CONSERVATIVE] Reduced to {max_parallel} parallel files")
            
            all_results = []
            start_time = time.time()
            
            # Optimize for RAG: reduce parallelism when RAG is enabled to avoid resource contention
            if self.enable_rag:
                max_parallel = min(max_parallel, 2)  # Limit to 2 concurrent RAG processes
                print(f"[RAG] Limiting parallel processing to {max_parallel} for RAG efficiency")
            
            # Process files with controlled parallelism
            semaphore = asyncio.Semaphore(max_parallel)
            
            async def analyze_file_with_semaphore(file_path: str):
                async with semaphore:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            code_content = f.read()
                        language = LanguageDetector.detect_language(file_path)
                        
                        result = await self.analyzer.analyze_file(
                            file_path=file_path,
                            code_content=code_content,
                            language=language,
                            selected_agents=selected_agents
                        )
                        
                        # Normalize the result
                        normalized_result = self._normalize_result_format(result, code_content)
                        return normalized_result
                        
                    except Exception as e:
                        print(f"[ERROR] Failed to analyze {file_path}: {str(e)}")
                        return {"error": str(e), "file_path": file_path, "total_issues": 0}
            
            # Execute all analyses
            tasks = [analyze_file_with_semaphore(fp) for fp in code_files]
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            final_results = []
            for result in all_results:
                if isinstance(result, Exception):
                    final_results.append({"error": str(result)})
                else:
                    final_results.append(result)
            
            # Aggregate results
            result = self._aggregate_directory_results(final_results, time.time() - start_time)
        else:
            # Use legacy directory analysis
            result = await self.analyzer.analyze_directory(
                directory_path, selected_agents, max_parallel
            )
        
        # Print results
        self._print_results(result, detailed)
        return result
    
    def _discover_files(self, directory: str) -> List[str]:
        """Discover code files in directory"""
        code_files = []
        supported_extensions = set()
        
        for lang_config in LanguageDetector.LANGUAGES.values():
            supported_extensions.update(lang_config.extensions)
        
        ignore_dirs = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'build', 'dist', 'workflow'}
        
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                if Path(file).suffix.lower() in supported_extensions:
                    code_files.append(os.path.join(root, file))
        
        return sorted(code_files)
    
    def _aggregate_directory_results(self, results: List[Dict], total_time: float) -> Dict[str, Any]:
        """Aggregate results from multiple file analyses"""
        successful_results = [r for r in results if 'error' not in r and r is not None]
        error_results = [r for r in results if 'error' in r or r is None]
        
        if error_results:
            print(f"[WARNING] {len(error_results)} files failed to analyze:")
            for err in error_results[:3]:  # Show first 3 errors
                if err and 'file_path' in err:
                    print(f"   - {err['file_path']}: {err.get('error', 'Unknown error')}")
                elif err is None:
                    print(f"   - Unknown file: Analysis returned None")
        
        total_files = len(successful_results)
        total_lines = sum(r.get('total_lines', 0) for r in successful_results)
        total_issues = sum(r.get('total_issues', 0) for r in successful_results) 
        total_tokens = sum(r.get('total_tokens', 0) for r in successful_results)
        total_llm_calls = sum(r.get('total_llm_calls', 0) for r in successful_results)
        
        # Aggregate severity counts
        agg_severity = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        all_issues = []
        
        for result in successful_results:
            for severity, count in result.get('issues_by_severity', {}).items():
                agg_severity[severity] += count
            all_issues.extend(result.get('all_issues', []))
        
        # Language statistics
        languages = {}
        for result in successful_results:
            lang = result.get('language', 'unknown')
            if lang not in languages:
                languages[lang] = {'files': 0, 'issues': 0}
            languages[lang]['files'] += 1
            languages[lang]['issues'] += result.get('total_issues', 0)
        
        return {
            'total_files': total_files,
            'total_lines': total_lines,
            'total_issues': total_issues,
            'total_tokens': total_tokens,
            'total_llm_calls': total_llm_calls,
            'processing_time': total_time,
            'languages': languages,
            'issues_by_severity': agg_severity,
            'all_issues': all_issues,
            'individual_results': results,
            'workflow_engine': 'langgraph' if self.use_langgraph else 'legacy'
        }
    
    def _print_results(self, result: Dict[str, Any], detailed: bool = False):
        """Print analysis results with rich formatting"""
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return
        
        is_directory = 'languages' in result
        workflow_engine = result.get('workflow_engine', 'legacy')
        
        print(f"\n[ANALYSIS SUMMARY] ({'LangGraph' if workflow_engine == 'langgraph' else 'Legacy'} Engine)")
        print("=" * 70)
        
        if is_directory:
            print(f"[FILES] Files Processed: {result['total_files']}")
            print(f"[LINES] Total Lines: {result['total_lines']:,}")
            print(f"[LANGUAGES] Languages: {len(result['languages'])}")
            for lang, stats in result['languages'].items():
                print(f"   - {lang.title()}: {stats['files']} files, {stats['issues']} issues")
        else:
            print(f"[FILE] File: {os.path.basename(result.get('file_path', 'Unknown'))}")
            print(f"[LANGUAGE] Language: {result['language'].title()}")
            print(f"[LINES] Lines: {result['total_lines']:,}")
        
        print(f"[ISSUES] Total Issues: {result['total_issues']}")
        print(f"[TIME] Processing Time: {result['processing_time']:.2f}s")
        print(f"[TOKENS] LLM Tokens Used: {result.get('total_tokens', 0):,}")
        
        # Cache hit info
        if result.get('cache_hit'):
            print(f"[CACHE] Cache Hit: Analysis loaded from cache")
        
        if workflow_engine == 'langgraph':
            completed_agents = result.get('completed_agents', [])
            failed_agents = result.get('failed_agents', [])
            if completed_agents:
                print(f"[SUCCESS] Completed Agents: {', '.join(completed_agents)}")
            if failed_agents:
                print(f"[FAILED] Failed Agents: {', '.join(failed_agents)}")
        
        print(f"\n[SEVERITY] ISSUES BY SEVERITY:")
        severity_totals = result.get('issues_by_severity', {})
        for severity in ['critical', 'high', 'medium', 'low']:
            count = severity_totals.get(severity, 0)
            if count > 0:
                print(f"   [{severity.upper()}] {severity.title()}: {count}")
        
        if not is_directory and result.get('agent_performance'):
            print(f"\n[PERFORMANCE] AGENT PERFORMANCE:")
            for agent, perf in result['agent_performance'].items():
                status = 'SUCCESS' if perf.get('status', 'completed') == 'completed' else 'FAILED'
                print(f"   [{status}] {agent.title()}: {perf.get('issues_found', 0)} issues, "
                      f"{perf.get('processing_time', 0.0):.2f}s, confidence: {perf.get('confidence', 0.8):.2f}")
        
        if detailed and result.get('all_issues'):
            print(f"\n[DETAILS] DETAILED ISSUES (Top 20 - Ordered by Severity)")
            print("-" * 70)
            
            # Sort by severity
            severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            sorted_issues = sorted(result['all_issues'], 
                                 key=lambda x: severity_order.get(x.get('severity', 'low'), 3))
            
            for i, issue in enumerate(sorted_issues[:20], 1):
                print(f"\n{i}. [{issue.get('severity', 'unknown').upper()}] {issue.get('title', 'Unknown Issue')}")
                print(f"   [AGENT] {issue.get('agent', 'unknown').title()}")
                print(f"   [FILE] {os.path.basename(issue.get('file_path', ''))}")
                line_number = issue.get('line_number', issue.get('line', 0))
                if line_number and line_number > 0:
                    print(f"   [LINE] {line_number}")
                print(f"   [DESC] {issue.get('description', 'No description')}")
                print(f"   [FIX] {issue.get('suggestion', issue.get('fix', 'No suggestion'))}")
                
                # Show evidence if available
                if issue.get('evidence'):
                    print(f"   [CODE] {issue['evidence']}")
        elif detailed and result.get('total_issues', 0) == 0:
            print(f"\n[DETAILS] No issues found - code appears to be well-written!")

async def main():
    """Main CLI interface"""
    
    # Check for interactive Q&A mode first
    if len(sys.argv) >= 2 and sys.argv[1] in ['--qa', 'qa', 'interactive']:
        from interactive_qa import InteractiveQASession
        
        if len(sys.argv) < 3:
            print("Usage: python main.py interactive <codebase_path>")
            print("       python main.py --qa <codebase_path>")
            sys.exit(1)
        
        codebase_path = sys.argv[2]
        session = InteractiveQASession(codebase_path)
        try:
            await session.initialize()
            await session.start_session()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        return
    
    if len(sys.argv) < 3 or sys.argv[1] not in ['analyze', 'workflow', 'agents']:
        print("🤖 LangGraph Multi-Agent Code Quality Intelligence")
        print("═" * 60)
        print("Usage:")
        print("  python main.py analyze <path> [options]     - Analyze code with LangGraph")
        print("  python main.py interactive <codebase_path>  - Interactive Q&A session")
        print("  python main.py workflow                     - Show workflow architecture")
        print("  python main.py agents                       - Show available agents")
        print()
        print("Analysis Options:")
        print("  --detailed              Show top 20 issues ordered by severity")
        print("  --agents AGENTS         Specific agents (security,performance,complexity,etc)")
        print("  --rag                   Enable RAG system for enhanced context")
        print("  --no-cache              Disable intelligent caching")
        print("  --legacy                Use legacy async orchestration instead of LangGraph")
        print("  --parallel N            Set parallel workers for directory analysis (default: 4)")
        print("  --max-files N           Limit analysis to N files")
        print()
        print("Examples:")
        print("  python main.py analyze app.py --detailed")
        print("  python main.py analyze . --agents security,performance --rag") 
        print("  python main.py interactive /path/to/codebase")
        print()
        print("LangGraph Features:")
        print("  🧠 Intelligent agent dependency management")
        print("  🔄 State-aware workflow orchestration") 
        print("  📊 Cross-agent insight sharing")
        print("  🚦 Dynamic routing and conditional execution")
        print("  📈 Real-time progress tracking")
        return
    
    command = sys.argv[1]
    
    if command == 'workflow':
        print("🔄 LangGraph Workflow Architecture")
        print("═" * 50)
        print("🚦 Agent Dependencies:")
        print("   Security Agent     → (No dependencies)")
        print("   Complexity Agent   → (No dependencies)")  
        print("   Performance Agent  → Complexity insights")
        print("   Testing Agent      → Security + Complexity insights")
        print("   Documentation Agent → Complexity insights")
        print("   Duplication Agent  → Complexity insights")
        print()
        print("📊 Workflow Flow:")
        print("   1. Initialize Analysis")
        print("   2. Setup RAG Context")
        print("   3. Route Available Agents")
        print("   4. Execute Agents (Dependency-aware)")
        print("   5. Share Cross-Agent Insights")
        print("   6. Aggregate Results")
        print("   7. Finalize Analysis")
        return
    
    if command == 'agents':
        print("🤖 Available LangGraph Agents")
        print("═" * 50)
        agents = [
            ("🛡️ Security", "Vulnerability detection, hardcoded secrets, injection attacks"),
            ("🔧 Complexity", "Code structure analysis, SOLID principles, complexity metrics"),
            ("⚡ Performance", "Algorithm efficiency, bottlenecks, optimization opportunities"),  
            ("🧪 Testing", "Test coverage analysis, missing tests, test quality"),
            ("📚 Documentation", "Missing docstrings, API documentation, code comments"),
            ("🔄 Duplication", "Code reuse analysis, repeated patterns, refactoring opportunities")
        ]
        
        for name, desc in agents:
            print(f"   {name:<15} {desc}")
        return
    
    # Parse arguments
    path = sys.argv[2]
    detailed = '--detailed' in sys.argv
    enable_rag = '--rag' in sys.argv
    enable_cache = '--no-cache' not in sys.argv
    use_langgraph = '--legacy' not in sys.argv
    
    # Parse agents
    selected_agents = None
    for i, arg in enumerate(sys.argv):
        if arg == '--agents' and i + 1 < len(sys.argv):
            selected_agents = [a.strip() for a in sys.argv[i + 1].split(',')]
            break
    
    # Parse parallel workers and max files
    parallel_workers = 4
    max_files = None
    
    for i, arg in enumerate(sys.argv):
        if arg == '--parallel' and i + 1 < len(sys.argv):
            try:
                parallel_workers = int(sys.argv[i + 1])
            except ValueError:
                print("⚠️ Invalid parallel workers value, using default: 4")
        elif arg == '--max-files' and i + 1 < len(sys.argv):
            try:
                max_files = int(sys.argv[i + 1])
            except ValueError:
                print("⚠️ Invalid max-files value, ignoring limit")
    
    # Initialize CQI system
    cqi = LangGraphCQI(
        enable_rag=enable_rag,
        enable_cache=enable_cache,
        use_langgraph=use_langgraph
    )
    
    try:
        if os.path.isfile(path):
            await cqi.analyze_file(path, selected_agents, detailed)
        elif os.path.isdir(path):
            await cqi.analyze_directory(path, selected_agents, detailed, parallel_workers, max_files)
        else:
            print(f"[ERROR] Path not found: {path}")
            return
    
    except KeyboardInterrupt:
        print(f"\n[INTERRUPT] Analysis interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())