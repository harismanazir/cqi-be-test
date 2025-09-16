"""Enhanced Base LLM agent with LangSmith integration."""

import os
import json
import time
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangSmith Integration - Import with fallback
try:
    from custom_langsmith import get_enhanced_prompt
    LANGSMITH_INTEGRATION = True
except ImportError:
    # Fallback if custom_langsmith module is not available
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


class GroqLLM:
    """Real Groq LLM interface with enhanced error handling"""
    
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
                temperature=0.1,  # Low temperature for more focused responses
                max_tokens=1000,
                timeout=60
            )
        else:
            raise RuntimeError("langchain-groq is not installed. Install with: pip install langchain-groq")
    
    async def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response using real Groq API only"""
        self.call_count += 1
        
        if not GROQ_AVAILABLE or not self.llm:
            raise RuntimeError("Groq API is not available. Please install langchain-groq and set GROQ_API_KEY")
        
        try:
            # Use real Groq API only
            messages = [HumanMessage(content=prompt)]
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            raise RuntimeError(f"Groq API call failed: {e}")


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
        'java': LanguageConfig(
            name='Java', 
            extensions=['.java'],
            comment_styles=['//', '/*', '/**'],
            string_delimiters=['"'],
            special_patterns={'function': r'(public|private|protected)\s+\w+\s+\w+\s*\(', 'class': r'class\s+\w+'}
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
            special_patterns={'function': r'function\s+\w+\s*\(', 'class': r'class\s+\w+'}
        ),
        'cpp': LanguageConfig(
            name='C++',
            extensions=['.cpp', '.cc', '.cxx', '.hpp'],
            comment_styles=['//', '/*'],
            string_delimiters=['"'],
            special_patterns={'function': r'\w+\s+\w+\s*\([^)]*\)\s*{', 'class': r'class\s+\w+'}
        ),
        'c': LanguageConfig(
            name='C',
            extensions=['.c', '.h'],
            comment_styles=['//', '/*'],
            string_delimiters=['"'],
            special_patterns={'function': r'\w+\s+\w+\s*\([^)]*\)\s*{'}
        ),
        'csharp': LanguageConfig(
            name='C#',
            extensions=['.cs'],
            comment_styles=['//', '/*', '///'],
            string_delimiters=['"'],
            special_patterns={'function': r'(public|private|protected)\s+\w+\s+\w+\s*\(', 'class': r'class\s+\w+'}
        ),
        'go': LanguageConfig(
            name='Go',
            extensions=['.go'],
            comment_styles=['//', '/*'],
            string_delimiters=['"', '`'],
            special_patterns={'function': r'func\s+\w+\s*\(', 'struct': r'type\s+\w+\s+struct'}
        ),
        'rust': LanguageConfig(
            name='Rust',
            extensions=['.rs'],
            comment_styles=['//', '/*'],
            string_delimiters=['"'],
            special_patterns={'function': r'fn\s+\w+\s*\(', 'struct': r'struct\s+\w+'}
        ),
        'php': LanguageConfig(
            name='PHP',
            extensions=['.php'],
            comment_styles=['//', '/*', '#'],
            string_delimiters=['"', "'"],
            special_patterns={'function': r'function\s+\w+\s*\(', 'class': r'class\s+\w+'}
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
        """Get configuration for a language"""
        return cls.LANGUAGES.get(language)


class CodeValidator:
    """Validates code before analysis to prevent false positives"""
    
    @staticmethod
    def is_empty_or_minimal(code: str) -> bool:
        """Check if code is empty or too minimal for meaningful analysis"""
        # Remove comments and whitespace
        cleaned = re.sub(r'#.*$', '', code, flags=re.MULTILINE)  # Python comments
        cleaned = re.sub(r'//.*$', '', cleaned, flags=re.MULTILINE)  # C-style comments
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)  # Block comments
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # Normalize whitespace
        
        # Check if remaining code is meaningful
        if len(cleaned) < 20:  # Less than 20 characters
            return True
        
        # Check for only imports/basic structure
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        non_import_lines = [line for line in lines if not (
            line.startswith('import ') or 
            line.startswith('from ') or
            line.startswith('#') or
            line.startswith('//') or
            line.startswith('/*') or
            line == '#!/usr/bin/env python' or
            line.startswith('<?php') or
            line.startswith('package ')
        )]
        
        return len(non_import_lines) < 3  # Less than 3 meaningful lines
    
    @staticmethod
    def has_actual_functionality(code: str, language: str) -> bool:
        """Check if code has actual functionality beyond basic structure"""
        if CodeValidator.is_empty_or_minimal(code):
            return False
            
        # Language-specific checks
        if language == 'python':
            # Look for function definitions, class methods, or meaningful logic
            patterns = [
                r'def\s+\w+\s*\([^)]*\)\s*:',  # Function definitions
                r'class\s+\w+.*:',             # Class definitions  
                r'if\s+.*:',                   # Conditional logic
                r'for\s+.*:',                  # Loops
                r'while\s+.*:',                # While loops
                r'try\s*:',                    # Exception handling
                r'with\s+.*:',                 # Context managers
            ]
        elif language in ['javascript', 'typescript']:
            patterns = [
                r'function\s+\w+\s*\(',        # Function declarations
                r'\w+\s*=\s*function',         # Function expressions
                r'\w+\s*=>\s*{',              # Arrow functions
                r'if\s*\(',                    # Conditionals
                r'for\s*\(',                   # Loops
                r'while\s*\(',                 # While loops
                r'class\s+\w+',                # Classes
            ]
        elif language == 'java':
            patterns = [
                r'(public|private|protected)\s+\w+\s+\w+\s*\(',  # Methods
                r'if\s*\(',                    # Conditionals
                r'for\s*\(',                   # Loops
                r'while\s*\(',                 # While loops
                r'class\s+\w+',                # Classes
                r'try\s*{',                    # Exception handling
            ]
        else:
            # Generic patterns for other languages
            patterns = [
                r'\w+\s*\([^)]*\)\s*{',       # Function-like patterns
                r'if\s*\(',                    # Conditionals
                r'for\s*\(',                   # Loops
                r'while\s*\(',                 # While loops
            ]
        
        return any(re.search(pattern, code, re.IGNORECASE) for pattern in patterns)


class BaseLLMAgent(ABC):
    """Enhanced base class for LLM-powered analysis agents with LangSmith integration"""
    
    def __init__(self, agent_type: str, rag_analyzer=None):
        self.agent_type = agent_type
        self.llm = GroqLLM()
        self.total_tokens = 0
        self.processing_time = 0
        self.rag_analyzer = rag_analyzer
        self.validator = CodeValidator()
    
    @abstractmethod
    def get_system_prompt(self, language: str) -> str:
        """Get system prompt for this agent and language - must be implemented by subclasses"""
        pass
    
    def get_enhanced_system_prompt(self, language: str) -> str:
        """Get enhanced system prompt using LangSmith if available"""
        # Get the base prompt from the agent
        fallback_prompt = self.get_system_prompt(language)

        # Generate LangSmith prompt name based on agent type
        prompt_name = f"{self.agent_type}-agent"

        # Try to get enhanced prompt from LangSmith Hub
        if LANGSMITH_INTEGRATION:
            try:
                enhanced_prompt = get_enhanced_prompt(
                    prompt_name=prompt_name,
                    fallback_prompt=fallback_prompt,
                    language=language,
                    agent_type=self.agent_type
                )
                
                # Log if we got an enhanced prompt
                if enhanced_prompt != fallback_prompt:
                    print(f"[{self.agent_type.upper()}] Using LangSmith enhanced prompt")
                
                return enhanced_prompt
                
            except Exception as e:
                print(f"[{self.agent_type.upper()}] LangSmith error, using fallback: {e}")
                return fallback_prompt
        else:
            print(f"[{self.agent_type.upper()}] Using fallback prompt (LangSmith not available)")
            return fallback_prompt
    
    def create_analysis_prompt(self, code: str, file_path: str, language: str) -> str:
        """Create enhanced analysis prompt with validation"""
        language_config = LanguageDetector.get_language_config(language)
        language_name = language_config.name if language_config else language.title()
        
        # Pre-validation checks
        if not self.validator.has_actual_functionality(code, language):
            return None  # Skip analysis for minimal code
        
        # Get RAG context if available
        rag_context = ""
        if self.rag_analyzer:
            context = self.rag_analyzer.get_relevant_context(self.agent_type, file_path, language)
            if context:
                rag_context = f"\n\nRelevant context from codebase:\n{context}\n"
        
        # Add line numbers to code for accurate reporting
        numbered_lines = []
        for i, line in enumerate(code.split('\n'), 1):
            numbered_lines.append(f"{i:4d}: {line}")
        numbered_code = '\n'.join(numbered_lines)
        
        # Use enhanced system prompt with LangSmith integration
        system_prompt = self.get_enhanced_system_prompt(language)
        
        # Enhanced prompt with strict accuracy requirements
        base_prompt = f"""
{system_prompt}

CRITICAL ACCURACY REQUIREMENTS:
- ONLY report issues that actually exist in the code
- If no real issues exist, return an empty issues array []
- Verify each issue by checking the specific line numbers
- Do not invent issues - be conservative and accurate
- If the code is well-written, it's okay to find zero issues

Analyze this {language_name} code for {self.agent_type} issues:

File: {file_path}
Language: {language_name}
{rag_context}

Code (with line numbers):
```{language}
{numbered_code}
```

IMPORTANT: Use the exact line numbers shown above. Only report genuine issues that you can verify exist in the code.

Provide analysis in JSON format:
{{
  "issues": [
    {{
      "severity": "critical|high|medium|low",
      "title": "Issue title",
      "description": "Detailed description with evidence",
      "line_number": 123,
      "suggestion": "How to fix this",
      "category": "{self.agent_type}",
      "evidence": "Quote the relevant code that demonstrates this issue"
    }}
  ],
  "metrics": {{"confidence": 0.95}},
  "confidence": 0.95
}}

REMEMBER: Return empty issues array if no real {self.agent_type} issues exist. Quality over quantity.
"""
        return base_prompt
    
    async def analyze(self, code: str, file_path: str, language: str) -> Dict[str, Any]:
        """Enhanced analysis with validation and accuracy checks"""
        start_time = time.time()
        
        try:
            # Pre-analysis validation
            if self.validator.is_empty_or_minimal(code):
                print(f"[{self.agent_type.upper()}] Skipping minimal/empty file: {os.path.basename(file_path)}")
                return {
                    'agent': self.agent_type,
                    'language': language,
                    'file_path': file_path,
                    'issues': [],
                    'metrics': {'confidence': 1.0, 'skipped_reason': 'minimal_code'},
                    'confidence': 1.0,
                    'tokens_used': 0,
                    'processing_time': 0.01,
                    'llm_calls': 0,
                    'langsmith_enhanced': LANGSMITH_INTEGRATION
                }
            
            if not self.validator.has_actual_functionality(code, language):
                print(f"[{self.agent_type.upper()}] No meaningful functionality found: {os.path.basename(file_path)}")
                return {
                    'agent': self.agent_type,
                    'language': language,
                    'file_path': file_path,
                    'issues': [],
                    'metrics': {'confidence': 1.0, 'skipped_reason': 'no_functionality'},
                    'confidence': 1.0,
                    'tokens_used': 0,
                    'processing_time': 0.01,
                    'llm_calls': 0,
                    'langsmith_enhanced': LANGSMITH_INTEGRATION
                }
            
            prompt = self.create_analysis_prompt(code, file_path, language)
            if prompt is None:
                # Return early for minimal code
                return {
                    'agent': self.agent_type,
                    'language': language,
                    'file_path': file_path,
                    'issues': [],
                    'metrics': {'confidence': 1.0},
                    'confidence': 1.0,
                    'tokens_used': 0,
                    'processing_time': 0.01,
                    'llm_calls': 0,
                    'langsmith_enhanced': LANGSMITH_INTEGRATION
                }
            
            response = await self.llm.generate(prompt, max_tokens=1500)
            
            # Enhanced JSON parsing with validation
            result = self._parse_and_validate_response(response, code, file_path)
            
            # Add metadata
            result['agent'] = self.agent_type
            result['language'] = language
            result['file_path'] = file_path
            result['tokens_used'] = len(prompt) // 4  # Rough estimation
            result['processing_time'] = time.time() - start_time
            result['langsmith_enhanced'] = LANGSMITH_INTEGRATION
            # Track actual LLM calls
            llm_calls = getattr(self.llm, 'call_count', 1) if self.llm else 1
            result['llm_calls'] = llm_calls

            self.total_tokens += result['tokens_used']
            self.processing_time += result['processing_time']
            
            # Final validation of reported issues
            validated_issues = self._validate_reported_issues(result['issues'], code)
            result['issues'] = validated_issues
            
            print(f"[{self.agent_type.upper()}] Found {len(validated_issues)} validated issues in {os.path.basename(file_path)}")
            
            return result
            
        except Exception as e:
            print(f"[{self.agent_type.upper()}] Analysis failed for {os.path.basename(file_path)}: {str(e)}")
            # Return empty result instead of fake errors
            return {
                'agent': self.agent_type,
                'language': language,
                'file_path': file_path,
                'issues': [],
                'metrics': {'confidence': 0.0, 'error': str(e)},
                'confidence': 0.0,
                'tokens_used': 0,
                'processing_time': time.time() - start_time,
                'langsmith_enhanced': LANGSMITH_INTEGRATION
            }
    
    def _parse_and_validate_response(self, response: str, code: str, file_path: str) -> Dict[str, Any]:
        """Enhanced JSON parsing with strict validation"""
        import re

        # Try direct JSON parsing first
        try:
            # Find the complete JSON object
            json_start = response.find('{')
            if json_start != -1:
                # Count braces to find the end of JSON
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
        except json.JSONDecodeError:
            pass

        # Clean and extract JSON
        response = response.strip()

        # Remove markdown code blocks if present
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*$', '', response)

        # Remove preamble text before JSON
        json_start = response.find('{')
        if json_start != -1:
            response = response[json_start:]

        # Find JSON object with proper validation
        json_match = re.search(r'\{.*?"issues"\s*:\s*\[.*?\].*?\}', response, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)

                # Clean common issues
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)

                # Fix unescaped quotes in evidence fields
                json_str = self._fix_json_escaping(json_str)

                result = json.loads(json_str)
                if self._validate_response_structure(result):
                    return result
            except json.JSONDecodeError:
                pass

        # If all parsing fails, return empty result (no fake issues)
        print(f"[WARNING] Could not parse LLM response for {os.path.basename(file_path)}, returning empty result")
        return {
            "issues": [],
            "metrics": {"confidence": 0.0, "parsing_failed": True},
            "confidence": 0.0
        }
    
    def _validate_response_structure(self, result: Dict) -> bool:
        """Validate that the response has proper structure"""
        if not isinstance(result, dict):
            return False
        if 'issues' not in result or not isinstance(result['issues'], list):
            return False

        # Validate each issue structure
        for issue in result['issues']:
            if not isinstance(issue, dict):
                return False
            required_fields = ['severity', 'title', 'description', 'line_number', 'suggestion']
            if not all(field in issue for field in required_fields):
                return False

        return True

    def _fix_json_escaping(self, json_str: str) -> str:
        """Fix common JSON escaping issues in LLM responses"""
        # Fix unescaped quotes inside evidence fields
        # Look for evidence fields and properly escape quotes within them

        def fix_evidence_field(match):
            full_match = match.group(0)
            evidence_content = match.group(1)

            # Escape unescaped quotes inside the evidence content
            # But be careful not to double-escape already escaped quotes
            fixed_content = evidence_content.replace('\\"', '__TEMP_ESCAPED__')  # Preserve existing escapes
            fixed_content = fixed_content.replace('"', '\\"')  # Escape unescaped quotes
            fixed_content = fixed_content.replace('__TEMP_ESCAPED__', '\\"')  # Restore original escapes

            return f'"evidence": "{fixed_content}"'

        # Pattern to match evidence fields with their content
        evidence_pattern = r'"evidence":\s*"([^"]*(?:\\.[^"]*)*)"'

        # First try: simple escape fix
        try:
            fixed_str = re.sub(evidence_pattern, fix_evidence_field, json_str)
            # Test if it parses
            json.loads(fixed_str)
            return fixed_str
        except:
            pass

        # Fallback: more aggressive fix - remove problematic evidence fields entirely
        try:
            # Remove evidence fields that are causing issues
            simplified_str = re.sub(r',?\s*"evidence":\s*"[^"]*"(?:\s*,)?', '', json_str)
            # Clean up any double commas
            simplified_str = re.sub(r',\s*,', ',', simplified_str)
            # Test if it parses
            json.loads(simplified_str)
            return simplified_str
        except:
            pass

        return json_str  # Return original if all fixes fail

    def _validate_reported_issues(self, issues: List[Dict], code: str) -> List[Dict]:
        """Validate that reported issues actually exist in the code"""
        if not issues:
            return []

        validated_issues = []
        code_lines = code.split('\n')

        for issue in issues:
            line_number = issue.get('line_number', 0)

            # Handle None line_number
            if line_number is None:
                line_number = 0

            # Validate line number
            if line_number <= 0 or line_number > len(code_lines):
                continue

            # Basic validation - the issue should reference something that actually exists
            line_content = code_lines[line_number - 1] if line_number <= len(code_lines) else ""

            # Skip obvious false positives (empty lines, comments only)
            if not line_content.strip() or line_content.strip().startswith('#') or line_content.strip().startswith('//'):
                continue

            # Add evidence field with actual code
            issue['evidence'] = line_content.strip()
            validated_issues.append(issue)
        return validated_issues