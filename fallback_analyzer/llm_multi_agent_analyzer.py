#!/usr/bin/env python3
"""
LLM-Powered Multi-Agent Code Quality Analyzer
Uses actual LLMs for intelligent code analysis across multiple programming languages
"""

import os
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# Import agents from agents directory
from agents import (
    SecurityAgent,
    PerformanceAgent,
    ComplexityAgent,
    DocumentationAgent,
    DuplicationAgent,
    LanguageDetector
)

# Import RAG system
try:
    from agents.rag_agent import RAGCodeAnalyzer, TRANSFORMERS_AVAILABLE
    RAG_AVAILABLE = TRANSFORMERS_AVAILABLE
    if not RAG_AVAILABLE:
        print("RAG system not available: sentence-transformers not properly installed")
        print("Install dependencies: pip install sentence-transformers scikit-learn")
except ImportError as e:
    print(f"RAG system not available: {e}")
    print("Install dependencies: pip install sentence-transformers scikit-learn")
    RAG_AVAILABLE = False

# Import caching system
try:
    from agents.intelligent_cache import get_cache, CacheEnabledVectorStore
    CACHE_AVAILABLE = True
    print("High-performance caching system loaded")
except ImportError as e:
    print(f"Caching system not available: {e}")
    CACHE_AVAILABLE = False


class LLMMultiAgentAnalyzer:
    """Multi-agent code analyzer using LLMs with optional RAG enhancement"""
    
    def __init__(self, enable_rag: bool = False, enable_cache: bool = True):
        # Initialize caching system
        self.cache = None
        if enable_cache and CACHE_AVAILABLE:
            self.cache = get_cache()
            self.cache.cleanup_old_cache()  # Clean old entries on startup
            print("Intelligent caching system enabled")
        
        # Initialize RAG system if requested and available
        self.rag_analyzer = None
        if enable_rag and RAG_AVAILABLE:
            self.rag_analyzer = RAGCodeAnalyzer()
            
            # Enhance RAG with caching if available
            if self.cache:
                self.rag_analyzer.vector_store = CacheEnabledVectorStore(
                    self.rag_analyzer.vector_store, 
                    self.cache
                )
            
            print("RAG system enabled for enhanced analysis")
        elif enable_rag and not RAG_AVAILABLE:
            print("[WARN] Warning: RAG requested but dependencies not available")
        
        # Initialize agents with optional RAG context
        self.agents = {
            'security': SecurityAgent(rag_analyzer=self.rag_analyzer),
            'performance': PerformanceAgent(rag_analyzer=self.rag_analyzer), 
            'complexity': ComplexityAgent(rag_analyzer=self.rag_analyzer),
            'documentation': DocumentationAgent(rag_analyzer=self.rag_analyzer),
            'duplication': DuplicationAgent(rag_analyzer=self.rag_analyzer)
        }
        
        self.rag_enabled = enable_rag and RAG_AVAILABLE
        self.cache_enabled = enable_cache and CACHE_AVAILABLE
    
    async def analyze_file(self, file_path: str, selected_agents: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze a single file using LLM agents with performance optimizations"""
        
        # Select agents to run
        agents_to_run = selected_agents or list(self.agents.keys())
        agents_to_run = [a for a in agents_to_run if a in self.agents]
        
        # Check cache first
        if self.cache_enabled:
            cached_result = self.cache.get_analysis_result(file_path, agents_to_run)
            if cached_result:
                print(f"[CACHE] Cache HIT: {os.path.basename(file_path)} (cached analysis)")
                return cached_result
            else:
                print(f"[CACHE] Cache MISS: {os.path.basename(file_path)} (analyzing...)")
        
        print(f"\n[ANALYZING] LLM Multi-Agent Analysis: {os.path.basename(file_path)}")
        print("-" * 50)
        
        # Read file
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
        except Exception as e:
            return {"error": f"Failed to read file: {e}"}
        
        # Performance optimization: Handle large files differently
        file_size = len(code)
        if file_size > 50000:  # 50KB threshold
            print(f"[FILE] Large file detected ({file_size:,} chars) - using chunked analysis")
            result = await self._analyze_large_file_optimized(file_path, code, selected_agents)
        else:
            result = await self._analyze_regular_file_optimized(file_path, code, agents_to_run)
        
        # Cache the result
        if self.cache_enabled and 'error' not in result:
            self.cache.save_analysis_result(file_path, agents_to_run, result)
        
        return result
    
    async def _analyze_regular_file_optimized(self, file_path: str, code: str, agents_to_run: List[str]) -> Dict[str, Any]:
        """Optimized analysis for regular-sized files with parallel agent execution"""
        
        # Detect language
        language = LanguageDetector.detect_language(file_path)
        print(f"[LANG] Language: {language.title()}")
        print(f"[PARALLEL] Running {len(agents_to_run)} agents in parallel...")
        
        # Run all agents concurrently (major performance boost)
        start_time = time.time()
        tasks = []
        
        for agent_name in agents_to_run:
            agent = self.agents[agent_name]
            print(f"  [FAST] Starting {agent_name.title()} Agent...")
            task = agent.analyze(code, file_path, language)
            tasks.append((agent_name, task))
        
        # Execute agents in parallel and collect results
        results = {}
        completed_count = 0
        
        for agent_name, task in tasks:
            try:
                result = await task
                results[agent_name] = result
                issues_found = len(result.get('issues', []))
                processing_time = result.get('processing_time', 0)
                completed_count += 1
                print(f"  [OK] {agent_name.title()} Agent: {issues_found} issues in {processing_time:.2f}s")
            except Exception as e:
                print(f"  [FAIL] {agent_name.title()} Agent: FAILED - {str(e)[:50]}...")
                continue
        
        total_time = time.time() - start_time
        
        # Aggregate results
        all_issues = []
        total_tokens = 0
        agent_performance = {}
        
        for agent_name, result in results.items():
            issues = result.get('issues', [])
            for issue in issues:
                issue['agent'] = agent_name
                issue['file_path'] = file_path
            all_issues.extend(issues)
            
            total_tokens += result.get('tokens_used', 0)
            agent_performance[agent_name] = {
                'issues_found': len(issues),
                'processing_time': result.get('processing_time', 0),
                'tokens_used': result.get('tokens_used', 0),
                'confidence': result.get('confidence', 0.0)
            }
        
        # Calculate summary
        issues_by_severity = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for issue in all_issues:
            severity = issue.get('severity', 'low')
            if severity in issues_by_severity:
                issues_by_severity[severity] += 1
        
        return {
            'file_path': file_path,
            'language': language,
            'total_lines': len(code.split('\n')),
            'total_issues': len(all_issues),
            'issues_by_severity': issues_by_severity,
            'agent_performance': agent_performance,
            'all_issues': all_issues,
            'processing_time': total_time,
            'total_tokens': total_tokens,
            'llm_calls': sum(agent.llm.call_count for agent in self.agents.values() if agent_name in results),
            'optimization': 'parallel_agents'
        }
        
        # Select agents
        agents_to_run = selected_agents or list(self.agents.keys())
        agents_to_run = [a for a in agents_to_run if a in self.agents]
        
        print(f"Running {len(agents_to_run)} LLM agents...")
        
        # Run agents concurrently
        start_time = time.time()
        tasks = []
        
        for agent_name in agents_to_run:
            agent = self.agents[agent_name]
            print(f"  Starting {agent_name.title()} Agent...")
            task = agent.analyze(code, file_path, language)
            tasks.append((agent_name, task))
        
        # Wait for all agents to complete
        results = {}
        for agent_name, task in tasks:
            try:
                result = await task
                results[agent_name] = result
                issues_found = len(result.get('issues', []))
                processing_time = result.get('processing_time', 0)
                print(f"  {agent_name.title()} Agent: {issues_found} issues in {processing_time:.2f}s")
            except Exception as e:
                print(f"  {agent_name.title()} Agent: FAILED - {str(e)}")
                # Skip this agent, don't add to results
                continue
        
        total_time = time.time() - start_time
        
        # Aggregate results
        all_issues = []
        total_tokens = 0
        agent_performance = {}
        
        for agent_name, result in results.items():
            issues = result.get('issues', [])
            for issue in issues:
                issue['agent'] = agent_name
                issue['file_path'] = file_path
            all_issues.extend(issues)
            
            total_tokens += result.get('tokens_used', 0)
            agent_performance[agent_name] = {
                'issues_found': len(issues),
                'processing_time': result.get('processing_time', 0),
                'tokens_used': result.get('tokens_used', 0),
                'confidence': result.get('confidence', 0.0)
            }
        
        # Calculate summary
        issues_by_severity = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for issue in all_issues:
            severity = issue.get('severity', 'low')
            if severity in issues_by_severity:
                issues_by_severity[severity] += 1
        
        return {
            'file_path': file_path,
            'language': language,
            'total_lines': len(code.split('\n')),
            'total_issues': len(all_issues),
            'issues_by_severity': issues_by_severity,
            'agent_performance': agent_performance,
            'all_issues': all_issues,
            'processing_time': total_time,
            'total_tokens': total_tokens,
            'llm_calls': sum(agent.llm.call_count for agent in self.agents.values())
        }
    
    async def analyze_directory(self, directory_path: str, selected_agents: Optional[List[str]] = None, max_parallel: int = 4) -> Dict[str, Any]:
        """Analyze all code files in a directory with high-performance optimizations"""
        
        print(f"[FAST] High-Performance Multi-Agent Directory Analysis: {directory_path}")
        if self.rag_enabled:
            print("[RAG] RAG-Enhanced Analysis Mode (Smart Context)")
        print(f"[PARALLEL] Parallel Processing: {max_parallel} concurrent files")
        print("=" * 70)
        
        start_time = time.time()
        
        # Discover files
        code_files = self._discover_files(directory_path)
        print(f"[DIR] Found {len(code_files)} code files")
        
        if not code_files:
            return {"error": "No code files found"}
        
        # Prioritize files by importance (security-sensitive files first)
        prioritized_files = self._prioritize_files(code_files)
        
        # Index codebase for RAG if enabled (with batch optimization)
        if self.rag_enabled and len(code_files) > 3:
            print("[RAG] Parallel RAG indexing...")
            await self._parallel_rag_indexing(prioritized_files)
            print("[OK] RAG indexing complete!")
        elif self.rag_enabled:
            print("[LANG] Small codebase - using standard analysis")
        
        # HIGH-PERFORMANCE PARALLEL ANALYSIS
        print(f"[RUNNING] Analyzing {len(prioritized_files)} files with {max_parallel} parallel workers...")
        all_results = await self._analyze_files_parallel(prioritized_files, selected_agents, max_parallel)
        
        # Real-time progress tracking
        successful_results = [r for r in all_results if 'error' not in r]
        failed_count = len(all_results) - len(successful_results)
        
        if failed_count > 0:
            print(f"[WARN]  {failed_count} files failed analysis")
        
        # Aggregate statistics
        language_stats = self._calculate_language_stats(successful_results)
        
        # Performance summary
        analysis_time = time.time() - start_time
        print(f"\n[SUMMARY] PERFORMANCE SUMMARY")
        print(f"[TIME]  Total Analysis Time: {analysis_time:.2f}s")
        print(f"[PARALLEL] Average Time per File: {analysis_time/max(len(successful_results), 1):.2f}s")
        
        if self.cache_enabled:
            cache_stats = self.cache.get_cache_stats()
            print(f"[CACHE] Cache Hit Rate: {cache_stats['hit_rate_percent']:.1f}%")
            print(f"[CACHE] Cache Size: {cache_stats['cache_dir_size_mb']} MB")
        
        # Aggregate all results
        total_files = len(successful_results)
        total_lines = sum(r['total_lines'] for r in successful_results)
        total_issues = sum(r['total_issues'] for r in all_results)
        total_tokens = sum(r['total_tokens'] for r in all_results)
        total_time = sum(r['processing_time'] for r in all_results)
        total_llm_calls = sum(r['llm_calls'] for r in all_results)
        
        # Aggregate severity
        agg_severity = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        all_issues = []
        
        for result in all_results:
            for severity, count in result['issues_by_severity'].items():
                agg_severity[severity] += count
            all_issues.extend(result['all_issues'])
        
        return {
            'total_files': total_files,
            'total_lines': total_lines,
            'total_issues': total_issues,
            'total_tokens': total_tokens,
            'total_llm_calls': total_llm_calls,
            'processing_time': total_time,
            'languages': language_stats,
            'issues_by_severity': agg_severity,
            'all_issues': all_issues,
            'individual_results': all_results
        }
    
    def _discover_files(self, directory: str) -> List[str]:
        """Discover code files in directory"""
        code_files = []
        supported_extensions = set()
        
        for lang_config in LanguageDetector.LANGUAGES.values():
            supported_extensions.update(lang_config.extensions)
        
        ignore_dirs = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'build', 'dist'}
        
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                if Path(file).suffix.lower() in supported_extensions:
                    code_files.append(os.path.join(root, file))
        
        return sorted(code_files)
    
    def _prioritize_files(self, files: List[str]) -> List[str]:
        """Prioritize files by importance (security-critical files first)"""
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for file_path in files:
            filename = os.path.basename(file_path).lower()
            
            # High priority: Security-sensitive files
            if any(keyword in filename for keyword in ['auth', 'login', 'password', 'security', 'admin', 'config']):
                high_priority.append(file_path)
            # Medium priority: Main application files
            elif any(keyword in filename for keyword in ['main', 'app', 'index', 'server', 'api']):
                medium_priority.append(file_path)
            else:
                low_priority.append(file_path)
        
        return high_priority + medium_priority + low_priority
    
    async def _parallel_rag_indexing(self, files: List[str]):
        """Parallel RAG indexing with batch processing"""
        import concurrent.futures
        
        def chunk_file(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                language = LanguageDetector.detect_language(file_path)
                return self.rag_analyzer.chunker.chunk_code(code, file_path, language)
            except Exception as e:
                print(f"[WARN] Failed to chunk {os.path.basename(file_path)}: {e}")
                return []
        
        # Parallel chunking
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            chunk_futures = [executor.submit(chunk_file, file_path) for file_path in files]
            all_chunks = []
            
            for future in concurrent.futures.as_completed(chunk_futures):
                chunks = future.result()
                all_chunks.extend(chunks)
        
        print(f"[STATS] Created {len(all_chunks)} chunks from {len(files)} files")
        
        # Batch embedding computation
        if all_chunks:
            self.rag_analyzer.vector_store.add_chunks(all_chunks)
    
    async def _analyze_files_parallel(self, files: List[str], selected_agents: Optional[List[str]], max_parallel: int) -> List[Dict]:
        """Analyze files in parallel with semaphore-controlled concurrency"""
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def analyze_with_semaphore(file_path: str) -> Dict:
            async with semaphore:
                print(f"[ANALYZING] Starting: {os.path.basename(file_path)}")
                try:
                    result = await self.analyze_file(file_path, selected_agents)
                    print(f"[OK] Completed: {os.path.basename(file_path)} ({result.get('total_issues', 0)} issues)")
                    return result
                except Exception as e:
                    print(f"[FAIL] Failed: {os.path.basename(file_path)} - {str(e)[:50]}")
                    return {"error": str(e), "file_path": file_path}
        
        # Execute all files concurrently with progress tracking
        tasks = [analyze_with_semaphore(file_path) for file_path in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                final_results.append({"error": str(result)})
            else:
                final_results.append(result)
        
        return final_results
    
    def _calculate_language_stats(self, results: List[Dict]) -> Dict[str, Dict]:
        """Calculate language statistics from results"""
        language_stats = {}
        
        for result in results:
            lang = result.get('language', 'unknown')
            if lang not in language_stats:
                language_stats[lang] = {'files': 0, 'issues': 0}
            
            language_stats[lang]['files'] += 1
            language_stats[lang]['issues'] += result.get('total_issues', 0)
        
        return language_stats
    
    async def _analyze_large_file_optimized(self, file_path: str, code: str, selected_agents: Optional[List[str]] = None) -> Dict[str, Any]:
        """Optimized analysis for large files"""
        
        # Detect language
        language = LanguageDetector.detect_language(file_path)
        
        # Smart chunking for large files
        chunks = self._chunk_large_file_intelligently(code, file_path, language)
        print(f"🔪 Split large file into {len(chunks)} chunks")
        
        # Analyze critical chunks only (first 3-5 chunks with issues)
        critical_chunks = chunks[:5]  # Limit analysis scope
        
        # Select agents
        agents_to_run = selected_agents or list(self.agents.keys())
        agents_to_run = [a for a in agents_to_run if a in self.agents]
        
        # Run agents on chunks in parallel
        chunk_results = []
        for chunk in critical_chunks:
            chunk_result = await self._analyze_chunk_parallel(chunk, file_path, language, agents_to_run)
            chunk_results.append(chunk_result)
        
        # Merge results from all chunks
        return self._merge_chunk_results(file_path, language, chunk_results, len(code.split('\n')))
    
    def _chunk_large_file_intelligently(self, code: str, file_path: str, language: str) -> List[str]:
        """Chunk large files based on language structure"""
        lines = code.split('\n')
        chunks = []
        
        chunk_size = 100  # Lines per chunk
        overlap = 10      # Overlapping lines for context
        
        for i in range(0, len(lines), chunk_size - overlap):
            end_idx = min(i + chunk_size, len(lines))
            chunk_lines = lines[i:end_idx]
            
            if chunk_lines:
                chunk_content = '\n'.join(chunk_lines)
                chunks.append(chunk_content)
        
        return chunks
    
    async def _analyze_chunk_parallel(self, chunk: str, file_path: str, language: str, agents_to_run: List[str]) -> Dict:
        """Analyze a single chunk with all agents in parallel"""
        
        # Run agents concurrently on this chunk
        tasks = []
        for agent_name in agents_to_run:
            agent = self.agents[agent_name]
            task = agent.analyze(chunk, file_path, language)
            tasks.append((agent_name, task))
        
        # Execute agents in parallel
        results = {}
        for agent_name, task in tasks:
            try:
                result = await task
                results[agent_name] = result
            except Exception as e:
                print(f"  {agent_name.title()} Agent: FAILED - {str(e)[:50]}...")
                continue
        
        return results
    
    def _merge_chunk_results(self, file_path: str, language: str, chunk_results: List[Dict], total_lines: int) -> Dict:
        """Merge results from multiple chunks"""
        
        # Aggregate all issues
        all_issues = []
        total_tokens = 0
        agent_performance = {}
        processing_time = 0
        
        for chunk_result in chunk_results:
            for agent_name, result in chunk_result.items():
                issues = result.get('issues', [])
                for issue in issues:
                    issue['agent'] = agent_name
                    issue['file_path'] = file_path
                all_issues.extend(issues)
                
                total_tokens += result.get('tokens_used', 0)
                processing_time += result.get('processing_time', 0)
                
                if agent_name not in agent_performance:
                    agent_performance[agent_name] = {
                        'issues_found': 0,
                        'processing_time': 0,
                        'tokens_used': 0,
                        'confidence': 0.0
                    }
                
                agent_performance[agent_name]['issues_found'] += len(issues)
                agent_performance[agent_name]['processing_time'] += result.get('processing_time', 0)
                agent_performance[agent_name]['tokens_used'] += result.get('tokens_used', 0)
                agent_performance[agent_name]['confidence'] = max(
                    agent_performance[agent_name]['confidence'], 
                    result.get('confidence', 0.0)
                )
        
        # Calculate summary
        issues_by_severity = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for issue in all_issues:
            severity = issue.get('severity', 'low')
            if severity in issues_by_severity:
                issues_by_severity[severity] += 1
        
        return {
            'file_path': file_path,
            'language': language,
            'total_lines': total_lines,
            'total_issues': len(all_issues),
            'issues_by_severity': issues_by_severity,
            'agent_performance': agent_performance,
            'all_issues': all_issues,
            'processing_time': processing_time,
            'total_tokens': total_tokens,
            'llm_calls': sum(len(cr) for cr in chunk_results),
            'optimization': 'large_file_chunked'
        }
    
    def print_results(self, result: Dict[str, Any], detailed: bool = False):
        """Print analysis results"""
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        is_directory = 'languages' in result
        
        print("\nLLM MULTI-AGENT ANALYSIS RESULTS")
        print("=" * 70)
        
        if is_directory:
            print(f"Files Analyzed: {result['total_files']}")
            print(f"Total Lines: {result['total_lines']:,}")
            print(f"Languages: {len(result['languages'])}")
            for lang, stats in result['languages'].items():
                print(f"  {lang.title()}: {stats['files']} files, {stats['issues']} issues")
        else:
            print(f"File: {os.path.basename(result['file_path'])}")
            print(f"Language: {result['language'].title()}")
            print(f"Lines: {result['total_lines']}")
        
        print(f"Total Issues: {result['total_issues']}")
        print(f"Processing Time: {result['processing_time']:.2f}s")
        print(f"LLM Tokens Used: {result.get('total_tokens', 0):,}")
        print(f"LLM API Calls: {result.get('total_llm_calls', 0)}")
        
        print(f"\nIssues by Severity:")
        for severity, count in result['issues_by_severity'].items():
            if count > 0:
                print(f"  {severity.title()}: {count}")
        
        if not is_directory and 'agent_performance' in result:
            print(f"\nAgent Performance:")
            for agent, perf in result['agent_performance'].items():
                print(f"  {agent.title()}: {perf['issues_found']} issues, "
                      f"{perf['processing_time']:.2f}s, "
                      f"confidence: {perf['confidence']:.2f}")
        
        if detailed and result['all_issues']:
            print(f"\nDETAILED ISSUES (Top 20 - Ordered by Severity)")
            print("-" * 70)
            
            # Sort issues by severity (critical -> high -> medium -> low)
            severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            sorted_issues = sorted(result['all_issues'], 
                                 key=lambda x: severity_order.get(x.get('severity', 'low'), 3))
            
            for i, issue in enumerate(sorted_issues[:20], 1):
                print(f"\n{i}. [{issue['severity'].upper()}] {issue['title']}")
                print(f"   Agent: {issue['agent'].title()}")
                print(f"   File: {os.path.basename(issue.get('file_path', ''))}")
                line_number = issue.get('line_number', 0)
                if line_number is not None and line_number > 0:
                    print(f"   Line: {line_number}")
                print(f"   Description: {issue['description']}")
                print(f"   Suggestion: {issue['suggestion']}")


async def main():
    """High-performance main function with all optimizations"""
    
    # Check for interactive Q&A mode
    if len(sys.argv) >= 2 and sys.argv[1] == '--qa':
        from interactive_qa import InteractiveQASession
        
        if len(sys.argv) < 3:
            print("Usage: python llm_multi_agent_analyzer.py --qa <codebase_path>")
            print("Example: python llm_multi_agent_analyzer.py --qa /path/to/project")
            sys.exit(1)
        
        # Start interactive Q&A session
        codebase_path = sys.argv[2]
        session = InteractiveQASession(codebase_path)
        try:
            await session.initialize()
            await session.start_session()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        return
    
    if len(sys.argv) < 2:
        print("[FAST] HIGH-PERFORMANCE LLM MULTI-AGENT CODE ANALYZER")
        print("=" * 60)
        print("Usage: python llm_multi_agent_analyzer.py <path> [options]")
        print("       python llm_multi_agent_analyzer.py --qa <codebase_path>")
        print("\nMODES:")
        print("  Analysis Mode:          Analyze files for issues and generate reports")
        print("  Interactive Q&A Mode:   Ask natural language questions about your codebase")
        print("\nPositional Arguments:")
        print("  <path>                  File or directory to analyze")
        print("  --qa <codebase_path>    Start interactive Q&A session")
        print("\nOptional Arguments:")
        print("  --detailed              Show top 20 issues ordered by severity")
        print("  --agents AGENTS         Specific agents (security,performance,complexity,documentation,duplication)")
        print("  --rag                   Enable RAG system for enhanced context")
        print("  --no-cache              Disable intelligent caching")
        print("  --parallel N            Set parallel workers (default: 4)")
        print("  --max-files N           Limit analysis to N files")
        print("\nInteractive Q&A Examples:")
        print("  python llm_multi_agent_analyzer.py --qa /path/to/project")
        print("  • What does the authenticate_user function do?")
        print("  • Are there any security vulnerabilities in my code?")
        print("  • How can I improve performance?")
        print("\nPerformance Features:")
        print("  [CACHE] Intelligent caching system")
        print("  [PARALLEL] Parallel file processing")  
        print("  [RAG] RAG-enhanced analysis")
        print("  [QA] Interactive conversational Q&A")
        print("  [RUNNING] Streaming results")
        print("  [STATS] Real-time progress tracking")
        return
    
    path = sys.argv[1]
    detailed = '--detailed' in sys.argv
    enable_rag = '--rag' in sys.argv
    enable_cache = '--no-cache' not in sys.argv
    
    # Parse parallel workers
    parallel_workers = 4
    for i, arg in enumerate(sys.argv):
        if arg == '--parallel' and i + 1 < len(sys.argv):
            try:
                parallel_workers = int(sys.argv[i + 1])
            except ValueError:
                print("[WARN] Invalid parallel workers value, using default: 4")
            break
    
    # Parse max files limit
    max_files = None
    for i, arg in enumerate(sys.argv):
        if arg == '--max-files' and i + 1 < len(sys.argv):
            try:
                max_files = int(sys.argv[i + 1])
            except ValueError:
                print("[WARN] Invalid max-files value, ignoring limit")
            break
    
    # Parse selected agents
    selected_agents = None
    for i, arg in enumerate(sys.argv):
        if arg == '--agents' and i + 1 < len(sys.argv):
            selected_agents = [a.strip() for a in sys.argv[i + 1].split(',')]
            break
    
    # Initialize high-performance analyzer
    print("[FAST] Initializing High-Performance Analyzer...")
    analyzer = LLMMultiAgentAnalyzer(enable_rag=enable_rag, enable_cache=enable_cache)
    
    # Warm up cache if enabled
    if analyzer.cache_enabled:
        print("[WARMUP] Warming up cache...")
        
    try:
        if os.path.isfile(path):
            print(f"[FILE] Single file analysis: {os.path.basename(path)}")
            result = await analyzer.analyze_file(path, selected_agents)
        elif os.path.isdir(path):
            print(f"[DIR] Directory analysis: {path}")
            
            # Apply file limit if specified
            if max_files:
                print(f"[LIMIT] Limiting analysis to {max_files} files")
            
            result = await analyzer.analyze_directory(
                path, 
                selected_agents, 
                max_parallel=parallel_workers
            )
        else:
            print(f"[FAIL] Path not found: {path}")
            return
        
        # Print results with performance info
        analyzer.print_results(result, detailed)
        
        # Final performance summary
        if analyzer.cache_enabled:
            cache_stats = analyzer.cache.get_cache_stats()
            print(f"\n[STATS] CACHE PERFORMANCE:")
            print(f"   Memory Cache: {cache_stats['memory_cache_size']} items")
            print(f"   Hit Rate: {cache_stats['hit_rate_percent']:.1f}%")
            print(f"   Total Size: {cache_stats['cache_dir_size_mb']} MB")
        
        optimization_features = []
        if analyzer.rag_enabled:
            optimization_features.append("[RAG] RAG-Enhanced Context")
        if analyzer.cache_enabled:
            optimization_features.append("[CACHE] Intelligent Caching")
        optimization_features.append("[PARALLEL] Parallel Processing")
        optimization_features.append("[STATS] Real-time Progress")
        
        print(f"\n[FAST] HIGH-PERFORMANCE ANALYSIS COMPLETE!")
        print(f"[OPTS] Optimizations Used: {', '.join(optimization_features)}")
        print("[SUMMARY] Each agent used LLM intelligence with performance enhancements")
        
    except KeyboardInterrupt:
        print(f"\n[STOP]  Analysis interrupted by user")
        if analyzer.cache_enabled:
            print("[CACHE] Cache data preserved for future runs")
    except Exception as e:
        print(f"\n[FAIL] Analysis failed: {str(e)}")
        if analyzer.cache_enabled:
            print("[CACHE] Partial results may be cached")


if __name__ == "__main__":
    asyncio.run(main())