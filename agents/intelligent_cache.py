"""Intelligent caching system for high-performance code analysis."""

import os
import pickle
import hashlib
import time
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import OrderedDict
import numpy as np


class HighPerformanceCache:
    """Multi-layer caching system with LRU memory cache and persistent disk cache"""
    
    def __init__(self, cache_dir: str = ".analysis_cache", max_memory_items: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_memory_items = max_memory_items
        
        # Multi-layer cache
        self.memory_cache = OrderedDict()  # LRU cache in memory
        self.embedding_cache = {}  # Specialized embedding cache
        self.analysis_cache_dir = self.cache_dir / "analyses"
        self.embedding_cache_dir = self.cache_dir / "embeddings"
        
        # Create subdirectories
        self.analysis_cache_dir.mkdir(exist_ok=True)
        self.embedding_cache_dir.mkdir(exist_ok=True)
        
        # Cache statistics
        self.stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'saves': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        print(f"[CACHE] Intelligent cache initialized: {cache_dir}")
    
    def get_file_hash(self, file_path: str, content: str = None) -> str:
        """Generate hash for file content and metadata"""
        if content is None:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception:
                return None
        
        # Include file metadata in hash
        stat = os.stat(file_path)
        metadata = f"{file_path}:{stat.st_mtime}:{len(content)}"
        
        hash_input = f"{metadata}:{content}".encode('utf-8')
        return hashlib.sha256(hash_input).hexdigest()
    
    def get_analysis_result(self, file_path: str, agent_types: List[str]) -> Optional[Dict]:
        """Get cached analysis result for a file"""
        file_hash = self.get_file_hash(file_path)
        if not file_hash:
            return None
        
        cache_key = f"{file_hash}:{'_'.join(sorted(agent_types))}"
        
        with self._lock:
            # Check memory cache first (fastest)
            if cache_key in self.memory_cache:
                self.stats['memory_hits'] += 1
                # Move to end (LRU)
                result = self.memory_cache.pop(cache_key)
                self.memory_cache[cache_key] = result
                return result
            
            # Check disk cache
            cache_file = self.analysis_cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        result = json.load(f)
                    
                    # Add to memory cache
                    self._add_to_memory_cache(cache_key, result)
                    
                    self.stats['disk_hits'] += 1
                    return result
                except Exception as e:
                    print(f"[WARNING] Cache read error: {e}")
            
            self.stats['misses'] += 1
            return None
    
    def save_analysis_result(self, file_path: str, agent_types: List[str], result: Dict):
        """Save analysis result to cache"""
        file_hash = self.get_file_hash(file_path)
        if not file_hash:
            return
        
        cache_key = f"{file_hash}:{'_'.join(sorted(agent_types))}"
        
        with self._lock:
            # Add timestamp and version
            cached_result = {
                **result,
                'cached_at': time.time(),
                'cache_version': '1.0'
            }
            
            # Save to memory cache
            self._add_to_memory_cache(cache_key, cached_result)
            
            # Save to disk cache (async to not block)
            self._save_to_disk_async(cache_key, cached_result)
            
            self.stats['saves'] += 1
    
    def get_embedding(self, content_hash: str) -> Optional[np.ndarray]:
        """Get cached embedding"""
        with self._lock:
            # Check memory first
            if content_hash in self.embedding_cache:
                return self.embedding_cache[content_hash]
            
            # Check disk
            cache_file = self.embedding_cache_dir / f"{content_hash}.npy"
            if cache_file.exists():
                try:
                    embedding = np.load(cache_file)
                    
                    # Add to memory cache if there's space
                    if len(self.embedding_cache) < self.max_memory_items // 2:
                        self.embedding_cache[content_hash] = embedding
                    
                    return embedding
                except Exception:
                    pass
            
            return None
    
    def save_embedding(self, content_hash: str, embedding: np.ndarray):
        """Save embedding to cache"""
        with self._lock:
            # Memory cache
            if len(self.embedding_cache) < self.max_memory_items // 2:
                self.embedding_cache[content_hash] = embedding
            
            # Disk cache
            try:
                cache_file = self.embedding_cache_dir / f"{content_hash}.npy"
                np.save(cache_file, embedding)
            except Exception as e:
                print(f"[WARNING] Embedding cache save error: {e}")
    
    def _add_to_memory_cache(self, key: str, value: Any):
        """Add to memory cache with LRU eviction"""
        if key in self.memory_cache:
            self.memory_cache.pop(key)
        elif len(self.memory_cache) >= self.max_memory_items:
            # Remove oldest item
            self.memory_cache.popitem(last=False)
        
        self.memory_cache[key] = value
    
    def _save_to_disk_async(self, cache_key: str, result: Dict):
        """Save to disk cache asynchronously"""
        def save_worker():
            try:
                cache_file = self.analysis_cache_dir / f"{cache_key}.json"
                with open(cache_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
            except Exception as e:
                print(f"[WARNING] Async cache save error: {e}")
        
        # Run in background thread
        threading.Thread(target=save_worker, daemon=True).start()
    
    def cleanup_old_cache(self, max_age_days: int = 7):
        """Clean up old cache entries"""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        removed_count = 0
        
        # Clean analysis cache
        for cache_file in self.analysis_cache_dir.glob("*.json"):
            try:
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    removed_count += 1
            except Exception:
                pass
        
        # Clean embedding cache
        for cache_file in self.embedding_cache_dir.glob("*.npy"):
            try:
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    removed_count += 1
            except Exception:
                pass
        
        if removed_count > 0:
            print(f"[CLEANUP] Cleaned up {removed_count} old cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = sum(self.stats.values()) - self.stats['saves']
        hit_rate = ((self.stats['memory_hits'] + self.stats['disk_hits']) / max(total_requests, 1)) * 100
        
        return {
            **self.stats,
            'memory_cache_size': len(self.memory_cache),
            'embedding_cache_size': len(self.embedding_cache),
            'hit_rate_percent': round(hit_rate, 2),
            'cache_dir_size_mb': self._get_cache_size_mb()
        }
    
    def _get_cache_size_mb(self) -> float:
        """Calculate total cache size in MB"""
        total_size = 0
        
        for cache_file in self.cache_dir.rglob("*"):
            if cache_file.is_file():
                try:
                    total_size += cache_file.stat().st_size
                except Exception:
                    pass
        
        return round(total_size / (1024 * 1024), 2)
    
    def warm_cache(self, files: List[str]):
        """Pre-warm cache by loading frequently accessed data"""
        print(f"[WARM] Warming cache for {len(files)} files...")
        
        # Load recent analysis results
        recent_analyses = []
        for cache_file in self.analysis_cache_dir.glob("*.json"):
            if time.time() - cache_file.stat().st_mtime < 3600:  # Last hour
                try:
                    with open(cache_file, 'r') as f:
                        result = json.load(f)
                        recent_analyses.append((cache_file.stem, result))
                except Exception:
                    continue
        
        # Add to memory cache
        for cache_key, result in recent_analyses[:self.max_memory_items//2]:
            self.memory_cache[cache_key] = result
        
        print(f"[WARM] Loaded {len(recent_analyses)} recent analyses into memory")


# Global cache instance
_cache_instance = None
_cache_lock = threading.Lock()

def get_cache() -> HighPerformanceCache:
    """Get global cache instance (singleton)"""
    global _cache_instance
    
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = HighPerformanceCache()
    
    return _cache_instance


class CacheEnabledVectorStore:
    """Enhanced vector store with intelligent caching"""
    
    def __init__(self, base_vector_store, cache: HighPerformanceCache = None):
        self.base_store = base_vector_store
        self.cache = cache or get_cache()
    
    def add_chunks_with_cache(self, chunks: List, force_recompute: bool = False):
        """Add chunks with intelligent caching"""
        
        if force_recompute:
            return self.base_store.add_chunks(chunks, force_recompute)
        
        # Check cache for embeddings
        cached_chunks = []
        new_chunks = []
        
        for chunk in chunks:
            content_hash = hashlib.sha256(chunk.content.encode()).hexdigest()
            cached_embedding = self.cache.get_embedding(content_hash)
            
            if cached_embedding is not None:
                chunk.embedding = cached_embedding
                cached_chunks.append(chunk)
            else:
                new_chunks.append(chunk)
        
        print(f"[STATS] Cache: {len(cached_chunks)} embeddings cached, {len(new_chunks)} to compute")
        
        # Compute new embeddings
        if new_chunks:
            # Create rich text for new chunks
            texts = []
            for chunk in new_chunks:
                text = f"File: {os.path.basename(chunk.file_path)}\\n"
                text += f"Language: {chunk.language}\\n"
                text += f"Type: {chunk.chunk_type}\\n"
                text += f"Code:\\n{chunk.content}"
                texts.append(text)
            
            # Batch compute embeddings
            embeddings = self.base_store.model.encode(texts, show_progress_bar=True)
            
            # Cache new embeddings
            for chunk, embedding in zip(new_chunks, embeddings):
                chunk.embedding = embedding
                content_hash = hashlib.sha256(chunk.content.encode()).hexdigest()
                self.cache.save_embedding(content_hash, embedding)
        
        # Update base store
        all_chunks = cached_chunks + new_chunks
        self.base_store.chunks = all_chunks
        if all_chunks:
            self.base_store.embeddings = np.vstack([c.embedding for c in all_chunks])
        else:
            self.base_store.embeddings = None
    
    def add_chunks(self, chunks: List, force_recompute: bool = False):
        """Delegate to cached version"""
        return self.add_chunks_with_cache(chunks, force_recompute)
    
    @property
    def chunks(self):
        """Delegate chunks property to base store"""
        return self.base_store.chunks
    
    @chunks.setter
    def chunks(self, value):
        """Delegate chunks setter to base store"""
        self.base_store.chunks = value
    
    @property
    def embeddings(self):
        """Delegate embeddings property to base store"""
        return self.base_store.embeddings
    
    @embeddings.setter
    def embeddings(self, value):
        """Delegate embeddings setter to base store"""
        self.base_store.embeddings = value
    
    def search_similar(self, query: str, top_k: int = 5, **kwargs):
        """Delegate to base store"""
        return self.base_store.search_similar(query, top_k, **kwargs)