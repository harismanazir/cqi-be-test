"""Enhanced Base LLM agent with comprehensive LangSmith tracing."""

import os
import json
import time
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import functools

# Load environment variables
load_dotenv()

# LangSmith Integration with tracing
try:
    from langsmith import Client, traceable
    from custom_langsmith import get_enhanced_prompt
    LANGSMITH_INTEGRATION = True
except ImportError:
    # Fallback decorators and functions
    def traceable(name=None, **kwargs):
        def decorator(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def get_enhanced_prompt(prompt_name: str, fallback_prompt: str, **kwargs) -> str:
        return fallback_prompt
    
    LANGSMITH_INTEGRATION = False

try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Error: langchain_groq not installed. Please install with: pip install langchain-groq")

import asyncio

# Existing imports (keep your existing LanguageDetector, GroqLLM, etc.)
@dataclass
class LanguageConfig:
    """Configuration for language-specific analysis"""
    name: str
    extensions: List[str]
    comment_styles: List[str]
    string_delimiters: List[str]
    special_patterns: Dict[str, str]

class LanguageDetector:
    """Advanced language detection with configuration"""
    
    LANGUAGES = {
        'python': LanguageConfig(
            name='Python',
            extensions=['.py', '.pyw'],
            comment_styles=['#', '"""', "'''"],
            string_delimiters=['"', "'"],
            special_patterns={'function': r'def\s+\w+', 'class': r'class\s+\w+'}
        ),
        'javascript': LanguageConfig(
            name='JavaScript',
            extensions=['.js', '.jsx', '.mjs'],
            comment_styles=['//', '/*'],
            string_delimiters=['"', "'", '`'],
            special_patterns={'function': r'function\s+\w+\s*\(', 'class': r'class\s+\w+'}
        ),
        'typescript': LanguageConfig(
            name='TypeScript',
            extensions=['.ts', '.tsx'],
            comment_styles=['//', '/*'],
            string_delimiters=['"', "'", '`'],
            special_patterns={'function': r'function\s+\w+\s*\(', 'class': r'class\s+\w+', 'interface': r'interface\s+\w+'}
        ),
        'java': LanguageConfig(
            name='Java',
            extensions=['.java'],
            comment_styles=['//', '/*'],
            string_delimiters=['"'],
            special_patterns={'function': r'(public|private|protected)?\s*(static\s+)?\w+\s+\w+\s*\(', 'class': r'(public\s+)?class\s+\w+'}
        ),
        'go': LanguageConfig(
            name='Go',
            extensions=['.go'],
            comment_styles=['//', '/*'],
            string_delimiters=['"', '`'],
            special_patterns={'function': r'func\s+\w+\s*\(', 'struct': r'type\s+\w+\s+struct'}
        ),
        'cpp': LanguageConfig(
            name='C++',
            extensions=['.cpp', '.cxx', '.cc', '.hpp', '.hxx', '.h'],
            comment_styles=['//', '/*'],
            string_delimiters=['"', "'"],
            special_patterns={'function': r'\w+\s+\w+\s*\(', 'class': r'class\s+\w+'}
        ),
        'c': LanguageConfig(
            name='C',
            extensions=['.c', '.h'],
            comment_styles=['//', '/*'],
            string_delimiters=['"', "'"],
            special_patterns={'function': r'\w+\s+\w+\s*\(', 'struct': r'struct\s+\w+'}
        ),
        'csharp': LanguageConfig(
            name='C#',
            extensions=['.cs'],
            comment_styles=['//', '/*'],
            string_delimiters=['"', "'"],
            special_patterns={'function': r'(public|private|protected)?\s*(static\s+)?\w+\s+\w+\s*\(', 'class': r'(public\s+)?class\s+\w+'}
        ),
        'html': LanguageConfig(
            name='HTML',
            extensions=['.html', '.htm', '.xhtml'],
            comment_styles=['<!--'],
            string_delimiters=['"', "'"],
            special_patterns={'tag': r'<\w+', 'script': r'<script', 'style': r'<style'}
        ),
        'css': LanguageConfig(
            name='CSS',
            extensions=['.css', '.scss', '.sass', '.less'],
            comment_styles=['/*'],
            string_delimiters=['"', "'"],
            special_patterns={'selector': r'[.#]?\w+\s*{', 'property': r'\w+\s*:'}
        ),
        'php': LanguageConfig(
            name='PHP',
            extensions=['.php', '.phtml'],
            comment_styles=['//', '/*', '#'],
            string_delimiters=['"', "'"],
            special_patterns={'function': r'function\s+\w+\s*\(', 'class': r'class\s+\w+'}
        ),
        'ruby': LanguageConfig(
            name='Ruby',
            extensions=['.rb', '.rake'],
            comment_styles=['#'],
            string_delimiters=['"', "'"],
            special_patterns={'function': r'def\s+\w+', 'class': r'class\s+\w+'}
        ),
        'rust': LanguageConfig(
            name='Rust',
            extensions=['.rs'],
            comment_styles=['//', '/*'],
            string_delimiters=['"', "'"],
            special_patterns={'function': r'fn\s+\w+\s*\(', 'struct': r'struct\s+\w+', 'impl': r'impl\s+\w+'}
        ),
        'kotlin': LanguageConfig(
            name='Kotlin',
            extensions=['.kt', '.kts'],
            comment_styles=['//', '/*'],
            string_delimiters=['"', "'"],
            special_patterns={'function': r'fun\s+\w+\s*\(', 'class': r'class\s+\w+'}
        ),
        'swift': LanguageConfig(
            name='Swift',
            extensions=['.swift'],
            comment_styles=['//', '/*'],
            string_delimiters=['"', "'"],
            special_patterns={'function': r'func\s+\w+\s*\(', 'class': r'class\s+\w+', 'struct': r'struct\s+\w+'}
        ),
        'scala': LanguageConfig(
            name='Scala',
            extensions=['.scala'],
            comment_styles=['//', '/*'],
            string_delimiters=['"', "'"],
            special_patterns={'function': r'def\s+\w+\s*\(', 'class': r'class\s+\w+', 'object': r'object\s+\w+'}
        ),
        'r': LanguageConfig(
            name='R',
            extensions=['.r', '.R'],
            comment_styles=['#'],
            string_delimiters=['"', "'"],
            special_patterns={'function': r'\w+\s*<-\s*function\s*\('}
        ),
        'matlab': LanguageConfig(
            name='MATLAB',
            extensions=['.m'],
            comment_styles=['%'],
            string_delimiters=['"', "'"],
            special_patterns={'function': r'function\s+.*\s*=\s*\w+\s*\('}
        ),
        'shell': LanguageConfig(
            name='Shell',
            extensions=['.sh', '.bash', '.zsh', '.fish'],
            comment_styles=['#'],
            string_delimiters=['"', "'"],
            special_patterns={'function': r'\w+\s*\(\s*\)\s*{'}
        ),
        'powershell': LanguageConfig(
            name='PowerShell',
            extensions=['.ps1', '.psm1'],
            comment_styles=['#', '<#'],
            string_delimiters=['"', "'"],
            special_patterns={'function': r'function\s+\w+\s*\('}
        ),
        'sql': LanguageConfig(
            name='SQL',
            extensions=['.sql'],
            comment_styles=['--', '/*'],
            string_delimiters=['"', "'"],
            special_patterns={'function': r'CREATE\s+(FUNCTION|PROCEDURE)\s+\w+', 'table': r'CREATE\s+TABLE\s+\w+'}
        ),
        'json': LanguageConfig(
            name='JSON',
            extensions=['.json'],
            comment_styles=[],
            string_delimiters=['"'],
            special_patterns={'object': r'\{', 'array': r'\['}
        ),
        'yaml': LanguageConfig(
            name='YAML',
            extensions=['.yml', '.yaml'],
            comment_styles=['#'],
            string_delimiters=['"', "'"],
            special_patterns={'key': r'\w+\s*:'}
        ),
        'xml': LanguageConfig(
            name='XML',
            extensions=['.xml', '.xsd', '.xsl'],
            comment_styles=['<!--'],
            string_delimiters=['"', "'"],
            special_patterns={'tag': r'<\w+', 'element': r'<\w+.*?>'}
        ),
        'dart': LanguageConfig(
            name='Dart',
            extensions=['.dart'],
            comment_styles=['//', '/*'],
            string_delimiters=['"', "'"],
            special_patterns={'function': r'\w+\s+\w+\s*\(', 'class': r'class\s+\w+'}
        ),
        'lua': LanguageConfig(
            name='Lua',
            extensions=['.lua'],
            comment_styles=['--'],
            string_delimiters=['"', "'"],
            special_patterns={'function': r'function\s+\w+\s*\('}
        ),
        'perl': LanguageConfig(
            name='Perl',
            extensions=['.pl', '.pm'],
            comment_styles=['#'],
            string_delimiters=['"', "'"],
            special_patterns={'function': r'sub\s+\w+\s*{'}
        ),
        'haskell': LanguageConfig(
            name='Haskell',
            extensions=['.hs'],
            comment_styles=['--'],
            string_delimiters=['"', "'"],
            special_patterns={'function': r'\w+\s*::'}
        ),
        'elixir': LanguageConfig(
            name='Elixir',
            extensions=['.ex', '.exs'],
            comment_styles=['#'],
            string_delimiters=['"', "'"],
            special_patterns={'function': r'def\s+\w+\s*\(', 'module': r'defmodule\s+\w+'}
        ),
        'erlang': LanguageConfig(
            name='Erlang',
            extensions=['.erl', '.hrl'],
            comment_styles=['%'],
            string_delimiters=['"', "'"],
            special_patterns={'function': r'\w+\s*\(.*\)\s*->'}
        ),
        'clojure': LanguageConfig(
            name='Clojure',
            extensions=['.clj', '.cljs', '.cljc'],
            comment_styles=[';'],
            string_delimiters=['"'],
            special_patterns={'function': r'\(defn\s+\w+'}
        ),
        'dockerfile': LanguageConfig(
            name='Dockerfile',
            extensions=['Dockerfile', '.dockerfile'],
            comment_styles=['#'],
            string_delimiters=['"', "'"],
            special_patterns={'instruction': r'^[A-Z]+\s+'}
        )
    }
    
    @classmethod
    def detect_language(cls, file_path: str) -> str:
        """Detect language from file extension"""
        from pathlib import Path
        extension = Path(file_path).suffix.lower()
        for lang_id, config in cls.LANGUAGES.items():
            if extension in config.extensions:
                return lang_id
        return 'unknown'

    @classmethod
    def get_language_config(cls, language: str) -> Optional[LanguageConfig]:
        """Get language configuration for a specific language"""
        return cls.LANGUAGES.get(language.lower())

class TracedGroqLLM:
    """Groq LLM with comprehensive LangSmith tracing"""
    
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.model_name = model_name
        self.call_count = 0
        
        if GROQ_AVAILABLE:
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            
            self.llm = ChatGroq(
                api_key=api_key,
                model_name=model_name,
                temperature=0.1,
                max_tokens=1500,
                timeout=60
            )
        else:
            raise RuntimeError("langchain-groq is not installed")
    
    @traceable(
        name="groq_llm_generate",
        metadata={"component": "llm", "provider": "groq"}
    )
    async def generate(self, prompt: str, max_tokens: int = 1000, 
                      agent_context: Dict[str, Any] = None) -> str:
        """Generate response with detailed tracing"""
        self.call_count += 1
        
        # Add metadata for tracing
        trace_metadata = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "prompt_length": len(prompt),
            "call_number": self.call_count,
            "agent_context": agent_context or {}
        }
        
        if not GROQ_AVAILABLE or not self.llm:
            raise RuntimeError("Groq API is not available")
        
        try:
            start_time = time.time()
            messages = [HumanMessage(content=prompt)]
            response = await self.llm.ainvoke(messages)
            end_time = time.time()
            
            # Add response metadata to trace
            response_metadata = {
                "response_length": len(response.content),
                "processing_time": end_time - start_time,
                "success": True
            }
            
            # Merge metadata (LangSmith will capture this automatically)
            return response.content
            
        except Exception as e:
            # Trace the error
            error_metadata = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "success": False
            }
            raise RuntimeError(f"Groq API call failed: {e}")

class CodeValidator:
    """Validates code before analysis to prevent false positives"""
    
    @staticmethod
    @traceable(name="code_validation")
    def is_empty_or_minimal(code: str) -> bool:
        """Check if code is empty or too minimal for meaningful analysis"""
        # Remove comments and whitespace
        cleaned = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        cleaned = re.sub(r'//.*$', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return len(cleaned) < 20
    
    @staticmethod
    @traceable(name="functionality_check")
    def has_actual_functionality(code: str, language: str) -> bool:
        """Check if code has actual functionality beyond basic structure"""
        if CodeValidator.is_empty_or_minimal(code):
            return False
            
        # Language-specific patterns (simplified)
        if language == 'python':
            patterns = [r'def\s+\w+\s*\([^)]*\)\s*:', r'class\s+\w+.*:', r'if\s+.*:']
        else:
            patterns = [r'\w+\s*\([^)]*\)\s*{', r'if\s*\(', r'for\s*\(']
        
        return any(re.search(pattern, code, re.IGNORECASE) for pattern in patterns)

class BaseLLMAgent(ABC):
    """Enhanced base class with comprehensive LangSmith tracing"""
    
    def __init__(self, agent_type: str, rag_analyzer=None):
        self.agent_type = agent_type
        self.llm = TracedGroqLLM()
        self.total_tokens = 0
        self.processing_time = 0
        self.rag_analyzer = rag_analyzer
        self.validator = CodeValidator()
        
        # Tracing metadata
        self.agent_metadata = {
            "agent_type": agent_type,
            "version": "2.0.0",
            "capabilities": ["static_analysis", "llm_enhancement", "rag_context"],
            "created_at": time.time()
        }
    
    @abstractmethod
    def get_system_prompt(self, language: str) -> str:
        """Get system prompt for this agent and language"""
        pass
    
    @traceable(
        name="get_enhanced_system_prompt",
        metadata={"component": "prompt_management"}
    )
    def get_enhanced_system_prompt(self, language: str) -> str:
        """Get enhanced system prompt using LangSmith if available"""
        fallback_prompt = self.get_system_prompt(language)
        prompt_name = f"{self.agent_type}-agent"
        
        if LANGSMITH_INTEGRATION:
            try:
                enhanced_prompt = get_enhanced_prompt(
                    prompt_name=prompt_name,
                    fallback_prompt=fallback_prompt,
                    language=language,
                    agent_type=self.agent_type
                )
                return enhanced_prompt
            except Exception as e:
                print(f"[{self.agent_type.upper()}] LangSmith error: {e}")
                return fallback_prompt
        else:
            return fallback_prompt
    
    @traceable(
        name="create_analysis_prompt",
        metadata={"component": "prompt_creation"}
    )
    def create_analysis_prompt(self, code: str, file_path: str, language: str) -> Optional[str]:
        """Create enhanced analysis prompt with validation and tracing"""
        language_config = LanguageDetector.get_language_config(language)
        language_name = language_config.name if language_config else language.title()
        
        # Pre-validation with tracing
        if not self.validator.has_actual_functionality(code, language):
            return None
        
        # Get RAG context with tracing
        rag_context = ""
        rag_chunks_used = 0
        if self.rag_analyzer:
            context = self.rag_analyzer.get_relevant_context(self.agent_type, file_path, language)
            if context:
                rag_context = f"\n\nRelevant context from codebase:\n{context}\n"
                rag_chunks_used = len(context.split("---")) if "---" in context else 1
        
        # Add metadata for this prompt creation
        prompt_metadata = {
            "file_path": file_path,
            "language": language_name,
            "code_length": len(code),
            "lines_of_code": len(code.split('\n')),
            "rag_chunks_used": rag_chunks_used,
            "has_rag_context": bool(rag_context)
        }
        
        # Create numbered code
        numbered_lines = []
        for i, line in enumerate(code.split('\n'), 1):
            numbered_lines.append(f"{i:4d}: {line}")
        numbered_code = '\n'.join(numbered_lines)
        
        system_prompt = self.get_enhanced_system_prompt(language)
        
        prompt = f"""
{system_prompt}

CRITICAL ACCURACY REQUIREMENTS:
- ONLY report issues that actually exist in the code
- If no real issues exist, return an empty issues array []
- Verify each issue by checking the specific line numbers
- Do not invent issues - be conservative and accurate

Analyze this {language_name} code for {self.agent_type} issues:

File: {file_path}
Language: {language_name}
{rag_context}

Code (with line numbers):
```{language}
{numbered_code}
```

Provide analysis in JSON format:
{{
  "issues": [
    {{
      "severity": "critical|high|medium|low",
      "title": "Issue title",
      "description": "Detailed description",
      "line_number": 123,
      "suggestion": "How to fix this",
      "category": "{self.agent_type}",
      "evidence": "Quote relevant code"
    }}
  ],
  "metrics": {{"confidence": 0.95}},
  "confidence": 0.95
}}
"""
        return prompt
    
    @traceable(
        name="agent_analyze",
        metadata={"component": "agent_analysis"}
    )
    async def analyze(self, code: str, file_path: str, language: str) -> Dict[str, Any]:
        """Enhanced analysis with comprehensive tracing"""
        start_time = time.time()
        
        # Add tracing metadata
        analysis_metadata = {
            "agent_type": self.agent_type,
            "file_path": file_path,
            "language": language,
            "code_size": len(code),
            "lines_of_code": len(code.split('\n'))
        }
        
        try:
            # Pre-analysis validation with tracing
            if self.validator.is_empty_or_minimal(code):
                return self._create_skip_result(file_path, language, "minimal_code", start_time)
            
            if not self.validator.has_actual_functionality(code, language):
                return self._create_skip_result(file_path, language, "no_functionality", start_time)
            
            # Create prompt with tracing
            prompt = self.create_analysis_prompt(code, file_path, language)
            if prompt is None:
                return self._create_skip_result(file_path, language, "prompt_creation_failed", start_time)
            
            # LLM generation with agent context
            agent_context = {
                "agent_type": self.agent_type,
                "file_path": file_path,
                "language": language
            }
            
            response = await self.llm.generate(
                prompt, 
                max_tokens=1500,
                agent_context=agent_context
            )
            
            # Parse and validate response with tracing
            result = self._parse_and_validate_response(response, code, file_path)
            
            # Add comprehensive metadata
            processing_time = time.time() - start_time
            result.update({
                'agent': self.agent_type,
                'language': language,
                'file_path': file_path,
                'tokens_used': len(prompt) // 4,
                'processing_time': processing_time,
                'langsmith_enhanced': LANGSMITH_INTEGRATION,
                'llm_calls': getattr(self.llm, 'call_count', 1),
                'analysis_metadata': analysis_metadata
            })
            
            # Validate and trace issues
            validated_issues = self._validate_reported_issues(result['issues'], code)
            result['issues'] = validated_issues
            
            # Update totals
            self.total_tokens += result['tokens_used']
            self.processing_time += result['processing_time']
            
            print(f"[{self.agent_type.upper()}] Found {len(validated_issues)} validated issues")
            
            return result
            
        except Exception as e:
            error_result = self._create_error_result(file_path, language, str(e), start_time)
            print(f"[{self.agent_type.upper()}] Analysis failed: {str(e)}")
            return error_result
    
    def _create_skip_result(self, file_path: str, language: str, reason: str, start_time: float) -> Dict[str, Any]:
        """Create result for skipped analysis"""
        return {
            'agent': self.agent_type,
            'language': language,
            'file_path': file_path,
            'issues': [],
            'metrics': {'confidence': 1.0, 'skipped_reason': reason},
            'confidence': 1.0,
            'tokens_used': 0,
            'processing_time': time.time() - start_time,
            'llm_calls': 0,
            'langsmith_enhanced': LANGSMITH_INTEGRATION,
            'status': 'skipped'
        }
    
    def _create_error_result(self, file_path: str, language: str, error: str, start_time: float) -> Dict[str, Any]:
        """Create result for failed analysis"""
        return {
            'agent': self.agent_type,
            'language': language,
            'file_path': file_path,
            'issues': [],
            'metrics': {'confidence': 0.0, 'error': error},
            'confidence': 0.0,
            'tokens_used': 0,
            'processing_time': time.time() - start_time,
            'langsmith_enhanced': LANGSMITH_INTEGRATION,
            'status': 'failed',
            'error_message': error
        }
    
    @traceable(name="parse_llm_response")
    def _parse_and_validate_response(self, response: str, code: str, file_path: str) -> Dict[str, Any]:
        """Enhanced JSON parsing with tracing"""
        parse_metadata = {
            "response_length": len(response),
            "file_path": file_path
        }
        
        try:
            # Try direct JSON parsing
            json_start = response.find('{')
            if json_start != -1:
                brace_count = 0
                json_end = json_start
                for i in range(json_start, len(response)):
                    if response[i] == '{':
                        brace_count += 1
                    elif response[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break

                json_content = response[json_start:json_end]
                result = json.loads(json_content)
                
                if self._validate_response_structure(result):
                    return result
        
        except json.JSONDecodeError as e:
            print(f"[{self.agent_type.upper()}] JSON parsing failed: {e}")
        
        # Return empty result if parsing fails
        return {
            "issues": [],
            "metrics": {"confidence": 0.0, "parsing_failed": True},
            "confidence": 0.0
        }
    
    def _validate_response_structure(self, result: Dict) -> bool:
        """Validate response structure"""
        if not isinstance(result, dict) or 'issues' not in result:
            return False
        
        if not isinstance(result['issues'], list):
            return False
        
        for issue in result['issues']:
            if not isinstance(issue, dict):
                return False
            required_fields = ['severity', 'title', 'description', 'line_number', 'suggestion']
            if not all(field in issue for field in required_fields):
                return False
        
        return True
    
    @traceable(name="validate_issues")
    def _validate_reported_issues(self, issues: List[Dict], code: str) -> List[Dict]:
        """Validate that reported issues actually exist"""
        if not issues:
            return []
        
        validated_issues = []
        code_lines = code.split('\n')
        
        for issue in issues:
            line_number = issue.get('line_number', 0)
            
            if line_number is None or line_number <= 0 or line_number > len(code_lines):
                continue
            
            line_content = code_lines[line_number - 1] if line_number <= len(code_lines) else ""
            
            if not line_content.strip() or line_content.strip().startswith('#'):
                continue
            
            issue['evidence'] = line_content.strip()
            validated_issues.append(issue)
        
        return validated_issues