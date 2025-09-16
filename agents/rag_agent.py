"""RAG-enhanced code analysis agent for large codebases."""

import os
import hashlib
import pickle
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import re

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Create dummy classes for type hints
    class SentenceTransformer:
        pass


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata"""
    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    chunk_type: str  # 'function', 'class', 'block', 'full_file'
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


class CodeChunker:
    """Intelligent code chunking for different programming languages"""
    
    def __init__(self, max_chunk_size: int = 500):
        self.max_chunk_size = max_chunk_size
        
        # Language-specific patterns for intelligent chunking
        self.language_patterns = {
            'python': {
                'function': r'^def\s+\w+.*?:',
                'class': r'^class\s+\w+.*?:',
                'import': r'^(?:from\s+\w+\s+)?import\s+',
            },
            'java': {
                'method': r'(public|private|protected)\s+.*?\w+\s*\([^)]*\)\s*\{',
                'class': r'(public\s+)?(class|interface)\s+\w+',
                'import': r'^import\s+',
            },
            'javascript': {
                'function': r'function\s+\w+\s*\([^)]*\)\s*\{',
                'arrow_function': r'\w+\s*=\s*\([^)]*\)\s*=>\s*\{',
                'class': r'class\s+\w+',
            }
        }
    
    def chunk_code(self, code: str, file_path: str, language: str) -> List[CodeChunk]:
        """Intelligently chunk code based on language structure"""
        lines = code.split('\n')
        chunks = []
        
        # Try language-specific chunking first
        if language in self.language_patterns:
            chunks.extend(self._smart_chunk_by_language(lines, file_path, language))
        
        # If no smart chunks found or file is too large, use sliding window
        if not chunks or len(code) > self.max_chunk_size * 3:
            chunks.extend(self._sliding_window_chunk(lines, file_path, language))
        
        # Always include a full-file chunk if file is small enough
        if len(code) <= self.max_chunk_size:
            chunks.append(CodeChunk(
                content=code,
                file_path=file_path,
                start_line=1,
                end_line=len(lines),
                language=language,
                chunk_type='full_file'
            ))
        
        return chunks
    
    def _smart_chunk_by_language(self, lines: List[str], file_path: str, language: str) -> List[CodeChunk]:
        """Create chunks based on language-specific structures"""
        chunks = []
        patterns = self.language_patterns.get(language, {})
        
        current_chunk_lines = []
        current_start = 1
        in_function = False
        brace_count = 0
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Check for function/method/class start
            is_structure_start = any(re.match(pattern, line_stripped, re.IGNORECASE) 
                                   for pattern in patterns.values())
            
            if is_structure_start and current_chunk_lines:
                # Save previous chunk
                chunk_content = '\n'.join(current_chunk_lines)
                if chunk_content.strip():
                    chunks.append(CodeChunk(
                        content=chunk_content,
                        file_path=file_path,
                        start_line=current_start,
                        end_line=i - 1,
                        language=language,
                        chunk_type='block'
                    ))
                
                current_chunk_lines = [line]
                current_start = i
                in_function = True
                brace_count = line.count('{') - line.count('}')
            else:
                current_chunk_lines.append(line)
                if in_function:
                    brace_count += line.count('{') - line.count('}')
                    if brace_count <= 0 and any(c in line for c in ['}', 'end', 'return']):
                        in_function = False
            
            # Force chunk if getting too large
            if len('\n'.join(current_chunk_lines)) > self.max_chunk_size:
                chunk_content = '\n'.join(current_chunk_lines)
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=current_start,
                    end_line=i,
                    language=language,
                    chunk_type='large_block'
                ))
                current_chunk_lines = []
                current_start = i + 1
                in_function = False
                brace_count = 0
        
        # Add remaining lines
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            if chunk_content.strip():
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=current_start,
                    end_line=len(lines),
                    language=language,
                    chunk_type='block'
                ))
        
        return chunks
    
    def _sliding_window_chunk(self, lines: List[str], file_path: str, language: str) -> List[CodeChunk]:
        """Create overlapping chunks using sliding window"""
        chunks = []
        window_size = 50  # lines per chunk
        overlap = 10      # overlapping lines
        
        for i in range(0, len(lines), window_size - overlap):
            end_idx = min(i + window_size, len(lines))
            chunk_lines = lines[i:end_idx]
            
            if chunk_lines:
                chunk_content = '\n'.join(chunk_lines)
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=i + 1,
                    end_line=end_idx,
                    language=language,
                    chunk_type='sliding_window'
                ))
        
        return chunks


class CodeVectorStore:
    """Vector store for code embeddings with similarity search"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
            
        try:
            self.model = SentenceTransformer(model_name)
        except Exception:
            # Fallback to a simpler model
            try:
                self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            except Exception as e:
                raise ImportError(f"Could not load sentence transformer model: {e}")
        
        self.chunks: List[CodeChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.cache_file = "code_embeddings_cache.pkl"
        
        # Load cached embeddings if available
        self._load_cache()
    
    def add_chunks(self, chunks: List[CodeChunk], force_recompute: bool = False):
        """Add code chunks and compute embeddings"""
        new_chunks = []
        
        for chunk in chunks:
            # Create hash for chunk to check if it's changed
            chunk_hash = hashlib.md5(chunk.content.encode()).hexdigest()
            chunk.metadata = chunk.metadata or {}
            chunk.metadata['hash'] = chunk_hash
            
            # Check if we already have this chunk
            existing = next((c for c in self.chunks if 
                           c.file_path == chunk.file_path and 
                           c.start_line == chunk.start_line and 
                           c.metadata.get('hash') == chunk_hash), None)
            
            if not existing or force_recompute:
                new_chunks.append(chunk)
            else:
                new_chunks.append(existing)  # Use cached version
        
        if new_chunks:
            # Compute embeddings for new chunks
            texts = []
            for chunk in chunks:
                # Create rich text representation for embedding
                text = f"File: {os.path.basename(chunk.file_path)}\n"
                text += f"Language: {chunk.language}\n"
                text += f"Type: {chunk.chunk_type}\n"
                text += f"Code:\n{chunk.content}"
                texts.append(text)
            
            print(f"Computing embeddings for {len(texts)} code chunks...")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            
            # Update chunks with embeddings
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            # Update our storage
            self.chunks = new_chunks
            self.embeddings = np.vstack([c.embedding for c in self.chunks])
            
            # Cache the results
            self._save_cache()
    
    def search_similar(self, query: str, top_k: int = 5, 
                      filter_language: Optional[str] = None,
                      filter_file: Optional[str] = None) -> List[Tuple[CodeChunk, float]]:
        """Search for code chunks similar to query"""
        if not self.chunks or self.embeddings is None:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Compute similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Filter results
        results = []
        for i, (chunk, similarity) in enumerate(zip(self.chunks, similarities)):
            # Apply filters
            if filter_language and chunk.language != filter_language:
                continue
            if filter_file and filter_file not in chunk.file_path:
                continue
            
            results.append((chunk, float(similarity)))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _save_cache(self):
        """Save embeddings to cache file"""
        try:
            cache_data = {
                'chunks': self.chunks,
                'model_name': self.model.__class__.__name__
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"Warning: Could not save embeddings cache: {e}")
    
    def _load_cache(self):
        """Load embeddings from cache file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.chunks = cache_data.get('chunks', [])
                if self.chunks and all(c.embedding is not None for c in self.chunks):
                    self.embeddings = np.vstack([c.embedding for c in self.chunks])
                    print(f"Loaded {len(self.chunks)} cached embeddings")
        except Exception as e:
            print(f"Warning: Could not load embeddings cache: {e}")
            self.chunks = []
            self.embeddings = None


class RAGCodeAnalyzer:
    """RAG-enhanced code analyzer for efficient large codebase analysis"""
    
    def __init__(self, max_context_chunks: int = 3):
        self.chunker = CodeChunker()
        self.vector_store = CodeVectorStore()
        self.max_context_chunks = max_context_chunks
    
    def index_codebase(self, file_paths: List[str], language_detector):
        """Index entire codebase for RAG"""
        print(f"Indexing {len(file_paths)} files for RAG...")
        
        all_chunks = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                
                language = language_detector.detect_language(file_path)
                chunks = self.chunker.chunk_code(code, file_path, language)
                all_chunks.extend(chunks)
                
            except Exception as e:
                print(f"Warning: Could not process {file_path}: {e}")
        
        print(f"Created {len(all_chunks)} code chunks")
        self.vector_store.add_chunks(all_chunks)
        print("RAG indexing complete!")
    
    def get_relevant_context(self, agent_type: str, file_path: str, language: str, max_tokens: int = 500) -> str:
        """Get relevant code context for analysis with token optimization"""
        # Create query based on agent type and file
        query_map = {
            'security': f"security vulnerabilities SQL injection XSS authentication {language}",
            'performance': f"performance optimization algorithms efficiency {language}",
            'complexity': f"code complexity readability maintainability {language}",
            'documentation': f"documentation comments docstrings API {language}",
            'duplication': f"duplicate code similar patterns repeated {language}"
        }
        
        query = query_map.get(agent_type, f"code analysis {language}")
        query += f" {os.path.basename(file_path)}"
        
        # Search for relevant chunks (get more candidates)
        similar_chunks = self.vector_store.search_similar(
            query=query,
            top_k=min(8, len(self.vector_store.chunks)),  # Get more candidates
            filter_language=None  # Don't filter by language for cross-language patterns
        )
        
        # Smart context selection with token budget
        selected_chunks = self._select_optimal_context(similar_chunks, max_tokens)
        
        # Build context
        context_parts = []
        for chunk, similarity in selected_chunks:
            if similarity > 0.3:  # Similarity threshold
                # Truncate very long chunks
                content = chunk.content
                if len(content) > 200:
                    content = content[:200] + "..."
                
                context_parts.append(f"Related code from {os.path.basename(chunk.file_path)} "
                                   f"(lines {chunk.start_line}-{chunk.end_line}):\n{content}\n")
        
        return "\n---\n".join(context_parts) if context_parts else ""
    
    def _select_optimal_context(self, chunks: List, max_tokens: int) -> List:
        """Select diverse, high-value context within token budget"""
        if not chunks:
            return []
        
        # Diversify by file and chunk type
        diverse_chunks = {}
        for chunk, similarity in chunks:
            file_key = os.path.basename(chunk.file_path)
            chunk_key = f"{file_key}_{chunk.chunk_type}"
            
            if chunk_key not in diverse_chunks or diverse_chunks[chunk_key][1] < similarity:
                diverse_chunks[chunk_key] = (chunk, similarity)
        
        # Sort by similarity and select within token budget
        sorted_chunks = sorted(diverse_chunks.values(), key=lambda x: x[1], reverse=True)
        
        selected = []
        token_count = 0
        
        for chunk, similarity in sorted_chunks:
            estimated_tokens = len(chunk.content) // 4  # Rough estimation
            
            if token_count + estimated_tokens <= max_tokens:
                selected.append((chunk, similarity))
                token_count += estimated_tokens
            
            if len(selected) >= 3:  # Max 3 context chunks
                break
        
        return selected