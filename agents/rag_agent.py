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


class AgentKnowledgeBase:
    """Agent-specific knowledge bases with patterns and best practices"""

    SECURITY_KNOWLEDGE = {
        'vulnerability_patterns': {
            'sql_injection': [
                "String concatenation in SQL queries",
                "Missing parameterized queries",
                "Direct user input in database queries",
                "Unescaped SQL parameters"
            ],
            'xss': [
                "Unescaped user input in HTML output",
                "Missing Content-Security-Policy headers",
                "Direct DOM manipulation with user data",
                "Unsafe innerHTML usage"
            ],
            'authentication': [
                "Hardcoded credentials",
                "Weak password policies",
                "Missing authentication checks",
                "Session management issues"
            ],
            'authorization': [
                "Missing access controls",
                "Privilege escalation risks",
                "Inadequate role-based security",
                "Direct object references"
            ],
            'crypto': [
                "Weak encryption algorithms",
                "Hardcoded cryptographic keys",
                "Improper random number generation",
                "Missing certificate validation"
            ]
        },
        'secure_patterns': {
            'python': [
                "Use parameterized queries with SQLAlchemy",
                "Validate input with marshmallow schemas",
                "Use secrets module for cryptographic operations",
                "Implement proper exception handling without exposing internals"
            ],
            'java': [
                "Use PreparedStatement for database queries",
                "Validate input with Bean Validation",
                "Use SecureRandom for cryptographic operations",
                "Implement proper error handling"
            ],
            'javascript': [
                "Use parameterized queries with ORM",
                "Sanitize input with libraries like DOMPurify",
                "Use crypto.getRandomValues() for random numbers",
                "Implement Content Security Policy"
            ]
        }
    }

    PERFORMANCE_KNOWLEDGE = {
        'optimization_patterns': {
            'algorithms': [
                "Use appropriate data structures (HashMap vs ArrayList)",
                "Implement efficient sorting algorithms",
                "Cache expensive computations",
                "Use lazy loading for large datasets"
            ],
            'database': [
                "Use database indexing effectively",
                "Implement connection pooling",
                "Batch database operations",
                "Use pagination for large result sets"
            ],
            'memory': [
                "Avoid memory leaks with proper cleanup",
                "Use object pooling for frequently created objects",
                "Implement efficient string concatenation",
                "Use streaming for large file processing"
            ],
            'concurrency': [
                "Use thread pools instead of creating new threads",
                "Implement proper synchronization",
                "Use async/await patterns correctly",
                "Avoid blocking operations in event loops"
            ]
        },
        'anti_patterns': {
            'python': [
                "Nested loops with O(n²) complexity",
                "Using + for string concatenation in loops",
                "Not using list comprehensions where appropriate",
                "Inefficient pandas operations"
            ],
            'java': [
                "Using Vector instead of ArrayList",
                "String concatenation in loops without StringBuilder",
                "Not closing resources properly",
                "Inefficient exception handling"
            ],
            'javascript': [
                "Blocking the event loop with synchronous operations",
                "Not using Promise.all for parallel operations",
                "Inefficient DOM manipulations",
                "Memory leaks with event listeners"
            ]
        }
    }

    COMPLEXITY_KNOWLEDGE = {
        'refactoring_patterns': {
            'method_extraction': [
                "Extract long methods into smaller functions",
                "Use descriptive method names",
                "Follow single responsibility principle",
                "Reduce parameter count"
            ],
            'conditional_simplification': [
                "Replace complex conditionals with polymorphism",
                "Use guard clauses to reduce nesting",
                "Extract condition logic into methods",
                "Use strategy pattern for complex decisions"
            ],
            'data_organization': [
                "Replace primitive obsession with objects",
                "Use value objects for related data",
                "Implement proper encapsulation",
                "Reduce coupling between classes"
            ]
        },
        'complexity_metrics': {
            'cyclomatic': "Number of decision points + 1",
            'cognitive': "Mental effort required to understand code",
            'nesting_depth': "Maximum levels of nested control structures",
            'parameter_count': "Number of parameters in methods/functions"
        },
        'best_practices': {
            'python': [
                "Use type hints for better readability",
                "Follow PEP 8 style guidelines",
                "Use dataclasses for simple data containers",
                "Implement proper error handling"
            ],
            'java': [
                "Use interfaces to reduce coupling",
                "Follow SOLID principles",
                "Use design patterns appropriately",
                "Implement proper exception hierarchy"
            ],
            'javascript': [
                "Use const/let instead of var",
                "Implement proper error handling with try/catch",
                "Use modern ES6+ features appropriately",
                "Follow consistent naming conventions"
            ]
        }
    }

    DOCUMENTATION_KNOWLEDGE = {
        'style_guides': {
            'python': [
                "Use Google or NumPy docstring format",
                "Document all public methods and classes",
                "Include type hints in function signatures",
                "Provide usage examples in docstrings"
            ],
            'java': [
                "Use JavaDoc format for documentation",
                "Document all public APIs",
                "Include @param, @return, and @throws tags",
                "Provide comprehensive class-level documentation"
            ],
            'javascript': [
                "Use JSDoc format for documentation",
                "Document all exported functions and classes",
                "Include parameter types and return values",
                "Provide usage examples"
            ]
        },
        'documentation_patterns': {
            'api_documentation': [
                "Document all endpoints with request/response examples",
                "Include authentication requirements",
                "Specify error codes and messages",
                "Provide rate limiting information"
            ],
            'code_comments': [
                "Explain why, not what",
                "Document complex business logic",
                "Include references to external resources",
                "Update comments when code changes"
            ],
            'readme_structure': [
                "Clear project description and purpose",
                "Installation and setup instructions",
                "Usage examples and API documentation",
                "Contributing guidelines and license information"
            ]
        }
    }

    @classmethod
    def get_agent_knowledge(cls, agent_type: str) -> Dict[str, Any]:
        """Get knowledge base for specific agent type"""
        knowledge_map = {
            'security': cls.SECURITY_KNOWLEDGE,
            'performance': cls.PERFORMANCE_KNOWLEDGE,
            'complexity': cls.COMPLEXITY_KNOWLEDGE,
            'documentation': cls.DOCUMENTATION_KNOWLEDGE
        }
        return knowledge_map.get(agent_type, {})

    @classmethod
    def get_relevant_patterns(cls, agent_type: str, language: str, query_context: str = "") -> List[str]:
        """Get relevant patterns for agent type and language"""
        knowledge = cls.get_agent_knowledge(agent_type)
        if not knowledge:
            return []

        patterns = []

        # Add language-specific patterns
        for category, content in knowledge.items():
            if isinstance(content, dict):
                if language in content:
                    patterns.extend(content[language])
                elif 'general' in content or len(content) > 0:
                    # Add general patterns if language-specific not found
                    for key, values in content.items():
                        if isinstance(values, list):
                            patterns.extend(values[:2])  # Limit to avoid overwhelming

        return patterns[:10]  # Limit total patterns


class RAGCodeAnalyzer:
    """RAG-enhanced code analyzer with agent-specific knowledge bases"""

    def __init__(self, max_context_chunks: int = 3):
        self.chunker = CodeChunker()
        self.vector_store = CodeVectorStore()
        self.max_context_chunks = max_context_chunks
        self.knowledge_base = AgentKnowledgeBase()
    
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
        """Get relevant code context with agent-specific knowledge integration"""

        # Get agent-specific knowledge patterns
        relevant_patterns = self.knowledge_base.get_relevant_patterns(agent_type, language)

        # Enhanced query creation using knowledge base
        base_query_map = {
            'security': f"security vulnerabilities SQL injection XSS authentication {language}",
            'performance': f"performance optimization algorithms efficiency {language}",
            'complexity': f"code complexity readability maintainability {language}",
            'documentation': f"documentation comments docstrings API {language}",
            'duplication': f"duplicate code similar patterns repeated {language}"
        }

        # Enhance query with knowledge base patterns
        base_query = base_query_map.get(agent_type, f"code analysis {language}")
        if relevant_patterns:
            # Add key patterns to search query
            pattern_keywords = " ".join(relevant_patterns[:3])  # Use top 3 patterns
            base_query += f" {pattern_keywords}"

        query = base_query + f" {os.path.basename(file_path)}"

        # Search for relevant chunks (get more candidates)
        similar_chunks = self.vector_store.search_similar(
            query=query,
            top_k=min(10, len(self.vector_store.chunks)),  # Increased candidates
            filter_language=None  # Don't filter by language for cross-language patterns
        )

        # Smart context selection with token budget
        selected_chunks = self._select_optimal_context(similar_chunks, max_tokens // 2)  # Reserve half for knowledge

        # Build enhanced context with knowledge base
        context_parts = []

        # Add agent-specific knowledge context
        if relevant_patterns and agent_type != 'duplication':  # Skip for duplication agent
            knowledge_context = self._build_knowledge_context(agent_type, language, relevant_patterns)
            if knowledge_context:
                context_parts.append(f"Agent-specific knowledge for {agent_type} analysis:\n{knowledge_context}")

        # Add similar code chunks
        for chunk, similarity in selected_chunks:
            if similarity > 0.3:  # Similarity threshold
                # Truncate very long chunks
                content = chunk.content
                if len(content) > 200:
                    content = content[:200] + "..."

                context_parts.append(f"Related code from {os.path.basename(chunk.file_path)} "
                                   f"(lines {chunk.start_line}-{chunk.end_line}):\n{content}\n")

        return "\n---\n".join(context_parts) if context_parts else ""

    def _build_knowledge_context(self, agent_type: str, language: str, patterns: List[str]) -> str:
        """Build knowledge-based context for analysis"""
        knowledge = self.knowledge_base.get_agent_knowledge(agent_type)
        if not knowledge:
            return ""

        context_lines = []

        if agent_type == 'security':
            context_lines.append("Security patterns to look for:")
            context_lines.extend([f"- {pattern}" for pattern in patterns[:5]])

            # Add language-specific secure patterns if available
            secure_patterns = knowledge.get('secure_patterns', {}).get(language, [])
            if secure_patterns:
                context_lines.append("\nRecommended secure practices:")
                context_lines.extend([f"- {pattern}" for pattern in secure_patterns[:3]])

        elif agent_type == 'performance':
            context_lines.append("Performance optimization areas:")
            context_lines.extend([f"- {pattern}" for pattern in patterns[:5]])

            # Add anti-patterns to watch for
            anti_patterns = knowledge.get('anti_patterns', {}).get(language, [])
            if anti_patterns:
                context_lines.append("\nPerformance anti-patterns to avoid:")
                context_lines.extend([f"- {pattern}" for pattern in anti_patterns[:3]])

        elif agent_type == 'complexity':
            context_lines.append("Complexity reduction techniques:")
            context_lines.extend([f"- {pattern}" for pattern in patterns[:5]])

            # Add best practices
            best_practices = knowledge.get('best_practices', {}).get(language, [])
            if best_practices:
                context_lines.append("\nCode quality best practices:")
                context_lines.extend([f"- {practice}" for practice in best_practices[:3]])

        elif agent_type == 'documentation':
            context_lines.append("Documentation standards:")
            context_lines.extend([f"- {pattern}" for pattern in patterns[:5]])

            # Add style guide recommendations
            style_guides = knowledge.get('style_guides', {}).get(language, [])
            if style_guides:
                context_lines.append(f"\n{language.title()} documentation style:")
                context_lines.extend([f"- {guide}" for guide in style_guides[:3]])

        return "\n".join(context_lines)
    
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