"""Performance optimizations for RAG-enhanced code analysis."""

import asyncio
import concurrent.futures
from typing import List, Dict, Any
import time
from pathlib import Path


class OptimizedRAGAnalyzer:
    """High-performance RAG analyzer with multiple optimizations"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.rag_analyzer = None
        self.agents = {}
        
    async def analyze_directory_parallel(self, directory_path: str) -> Dict[str, Any]:
        """Analyze directory with parallel processing optimizations"""
        
        print(f"🚀 Optimized Analysis: {directory_path}")
        start_time = time.time()
        
        # 1. PARALLEL FILE DISCOVERY AND CHUNKING
        code_files = self._discover_files(directory_path)
        print(f"📁 Found {len(code_files)} files")
        
        # 2. BATCH INDEXING (if RAG enabled)
        if self.rag_analyzer:
            await self._parallel_indexing(code_files)
        
        # 3. PARALLEL FILE ANALYSIS
        results = await self._analyze_files_parallel(code_files)
        
        total_time = time.time() - start_time
        print(f"⚡ Optimized analysis completed in {total_time:.2f}s")
        
        return self._aggregate_results(results, total_time)
    
    async def _parallel_indexing(self, files: List[str]):
        """Parallel chunking and embedding computation"""
        print("🧠 Parallel RAG indexing...")
        
        # Chunk files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            chunk_futures = []
            for file_path in files:
                future = executor.submit(self._chunk_single_file, file_path)
                chunk_futures.append(future)
            
            # Collect all chunks
            all_chunks = []
            for future in concurrent.futures.as_completed(chunk_futures):
                chunks = future.result()
                all_chunks.extend(chunks)
        
        print(f"📊 Created {len(all_chunks)} chunks in parallel")
        
        # Batch embedding computation (more efficient than individual)
        if all_chunks:
            await self._batch_compute_embeddings(all_chunks)
    
    async def _analyze_files_parallel(self, files: List[str]) -> List[Dict]:
        """Analyze multiple files concurrently"""
        print(f"🔄 Analyzing {len(files)} files in parallel...")
        
        # Create semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def analyze_single_file(file_path: str):
            async with semaphore:
                return await self._analyze_file_optimized(file_path)
        
        # Run file analyses concurrently
        tasks = [analyze_single_file(file_path) for file_path in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if not isinstance(r, Exception)]
        return valid_results
    
    async def _analyze_file_optimized(self, file_path: str) -> Dict:
        """Optimized single file analysis with parallel agents"""
        
        # Read file once
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
        
        # Skip very large files (chunk them differently)
        if len(code) > 50000:  # 50KB threshold
            return await self._analyze_large_file(file_path, code)
        
        # Run all agents in parallel for this file
        agent_tasks = []
        for agent_name, agent in self.agents.items():
            task = agent.analyze(code, file_path, self._detect_language(file_path))
            agent_tasks.append((agent_name, task))
        
        # Execute agents concurrently
        results = {}
        for agent_name, task in agent_tasks:
            try:
                result = await task
                results[agent_name] = result
            except Exception as e:
                print(f"⚠️ {agent_name} failed: {str(e)[:50]}...")
                continue
        
        return self._format_file_result(file_path, results)
    
    async def _analyze_large_file(self, file_path: str, code: str) -> Dict:
        """Special handling for large files"""
        print(f"📄 Large file optimization: {Path(file_path).name}")
        
        # Strategy 1: Chunk-based analysis
        chunks = self._smart_chunk_large_file(code, file_path)
        
        # Strategy 2: Focus on high-impact chunks only
        critical_chunks = self._identify_critical_chunks(chunks)
        
        # Strategy 3: Parallel chunk analysis
        chunk_results = []
        semaphore = asyncio.Semaphore(2)  # Limit concurrent chunks
        
        async def analyze_chunk(chunk):
            async with semaphore:
                return await self._analyze_code_chunk(chunk, file_path)
        
        tasks = [analyze_chunk(chunk) for chunk in critical_chunks]
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return self._merge_chunk_results(file_path, chunk_results)


class FastEmbeddingCache:
    """High-performance embedding cache with compression"""
    
    def __init__(self, cache_dir: str = ".rag_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._memory_cache = {}  # In-memory LRU cache
        self._max_memory_items = 1000
    
    def get_cached_embedding(self, content_hash: str):
        """Get embedding from cache (memory first, then disk)"""
        
        # Check memory cache first (fastest)
        if content_hash in self._memory_cache:
            return self._memory_cache[content_hash]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{content_hash}.npy"
        if cache_file.exists():
            import numpy as np
            embedding = np.load(cache_file)
            
            # Add to memory cache
            if len(self._memory_cache) < self._max_memory_items:
                self._memory_cache[content_hash] = embedding
            
            return embedding
        
        return None
    
    def cache_embedding(self, content_hash: str, embedding):
        """Cache embedding to memory and disk"""
        import numpy as np
        
        # Memory cache
        if len(self._memory_cache) < self._max_memory_items:
            self._memory_cache[content_hash] = embedding
        
        # Disk cache
        cache_file = self.cache_dir / f"{content_hash}.npy"
        np.save(cache_file, embedding)


class SmartContextSelector:
    """Intelligent context selection to reduce token usage"""
    
    def __init__(self, max_context_tokens: int = 500):
        self.max_context_tokens = max_context_tokens
    
    def select_optimal_context(self, query: str, chunks: List, current_code: str):
        """Select minimal but high-impact context"""
        
        # Strategy 1: Diversity-based selection
        diverse_chunks = self._select_diverse_chunks(chunks)
        
        # Strategy 2: Relevance scoring  
        scored_chunks = self._score_relevance(query, diverse_chunks, current_code)
        
        # Strategy 3: Token budget optimization
        optimal_chunks = self._optimize_token_budget(scored_chunks)
        
        return optimal_chunks
    
    def _select_diverse_chunks(self, chunks):
        """Select chunks from different files/types for diversity"""
        diverse = {}
        
        for chunk, similarity in chunks:
            file_key = Path(chunk.file_path).name
            chunk_type = chunk.chunk_type
            
            key = f"{file_key}_{chunk_type}"
            if key not in diverse or diverse[key][1] < similarity:
                diverse[key] = (chunk, similarity)
        
        return list(diverse.values())
    
    def _score_relevance(self, query: str, chunks, current_code: str):
        """Advanced relevance scoring"""
        scored = []
        
        for chunk, similarity in chunks:
            # Base similarity score
            score = similarity
            
            # Bonus for cross-language patterns
            if chunk.language != self._detect_language_from_code(current_code):
                score += 0.1
            
            # Bonus for different vulnerability types
            if self._has_different_vuln_type(chunk.content, current_code):
                score += 0.15
            
            # Penalty for very similar code (avoid redundancy)
            if self._too_similar_to_current(chunk.content, current_code):
                score -= 0.2
            
            scored.append((chunk, score))
        
        return sorted(scored, key=lambda x: x[1], reverse=True)
    
    def _optimize_token_budget(self, scored_chunks):
        """Select chunks that fit within token budget"""
        selected = []
        token_count = 0
        
        for chunk, score in scored_chunks:
            estimated_tokens = len(chunk.content) // 4  # Rough estimation
            
            if token_count + estimated_tokens <= self.max_context_tokens:
                selected.append((chunk, score))
                token_count += estimated_tokens
            else:
                break
        
        return selected


# Performance monitoring
class PerformanceProfiler:
    """Track and optimize performance bottlenecks"""
    
    def __init__(self):
        self.timings = {}
        self.call_counts = {}
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations"""
        return self._TimingContext(self, operation_name)
    
    class _TimingContext:
        def __init__(self, profiler, operation_name):
            self.profiler = profiler
            self.operation_name = operation_name
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time.time() - self.start_time
            
            if self.operation_name not in self.profiler.timings:
                self.profiler.timings[self.operation_name] = []
                self.profiler.call_counts[self.operation_name] = 0
            
            self.profiler.timings[self.operation_name].append(duration)
            self.profiler.call_counts[self.operation_name] += 1
    
    def get_performance_report(self) -> str:
        """Generate performance analysis report"""
        report = "🔍 PERFORMANCE ANALYSIS REPORT\n"
        report += "=" * 50 + "\n\n"
        
        total_time = 0
        for operation, times in self.timings.items():
            avg_time = sum(times) / len(times)
            total_operation_time = sum(times)
            call_count = self.call_counts[operation]
            
            report += f"{operation}:\n"
            report += f"  Calls: {call_count}\n" 
            report += f"  Avg: {avg_time:.3f}s\n"
            report += f"  Total: {total_operation_time:.3f}s\n"
            report += f"  % of total: {(total_operation_time/sum(sum(t) for t in self.timings.values()))*100:.1f}%\n\n"
            
            total_time += total_operation_time
        
        report += f"TOTAL ANALYSIS TIME: {total_time:.2f}s\n"
        return report