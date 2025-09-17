"""Enhanced Security Agent with comprehensive LangSmith tracing."""

from .base_agent import BaseLLMAgent, traceable
import re
from typing import Dict, List, Any

class SecurityAgent(BaseLLMAgent):
    """Enhanced Security Agent with detailed tracing and evaluation hooks"""
    
    def __init__(self, rag_analyzer=None):
        super().__init__('security', rag_analyzer)
        self.security_patterns = self._init_security_patterns()
        
        # Security-specific metadata for tracing
        self.agent_metadata.update({
            "specialization": "security_vulnerabilities",
            "pattern_count": len(self.security_patterns),
            "vulnerability_types": list(self.security_patterns.keys())
        })
    
    @traceable(
        name="init_security_patterns",
        metadata={"component": "pattern_initialization"}
    )
    def _init_security_patterns(self):
        """Initialize security patterns with tracing"""
        patterns = {
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']{3,}["\']',
                r'api_key\s*=\s*["\'][^"\']{10,}["\']',
                r'secret\s*=\s*["\'][^"\']{8,}["\']',
                r'token\s*=\s*["\'][^"\']{10,}["\']',
                r'["\'][A-Za-z0-9]{20,}["\']',
            ],
            'sql_injection': [
                r'["\'].*\+.*["\'].*sql',
                r'execute\s*\(\s*["\'].*\+',
                r'query\s*\(\s*["\'].*\+',
                r'cursor\.execute\s*\(\s*["\'].*%',
            ],
            'command_injection': [
                r'os\.system\s*\(',
                r'subprocess\.\w+\s*\([^)]*shell\s*=\s*True',
                r'eval\s*\(',
                r'exec\s*\(',
            ],
            'path_traversal': [
                r'\.\./',
                r'\.\.\\',
                r'os\.path\.join.*\.\.',
            ],
            'xss_vulnerabilities': [
                r'innerHTML\s*=.*\+',
                r'document\.write\s*\(',
                r'eval\s*\(',
            ]
        }
        return patterns
    
    @traceable(
        name="pre_validate_security",
        metadata={"component": "static_analysis"}
    )
    def _pre_validate_security_issues(self, code: str) -> Dict[str, Any]:
        """Enhanced pre-validation with detailed pattern matching"""
        found_patterns = {}
        total_matches = 0
        
        for category, patterns in self.security_patterns.items():
            category_matches = []
            for pattern in patterns:
                matches = list(re.finditer(pattern, code, re.IGNORECASE))
                if matches:
                    category_matches.extend([{
                        'pattern': pattern,
                        'line': code[:match.start()].count('\n') + 1,
                        'match': match.group()
                    } for match in matches])
            
            if category_matches:
                found_patterns[category] = category_matches
                total_matches += len(category_matches)
        
        validation_result = {
            'has_issues': total_matches > 0,
            'pattern_matches': found_patterns,
            'total_matches': total_matches,
            'categories_affected': list(found_patterns.keys())
        }
        
        return validation_result
    
    def get_system_prompt(self, language: str) -> str:
        return f"""You are a SECURITY EXPERT specializing in {language} code analysis.

CRITICAL ACCURACY REQUIREMENTS:
- ONLY report security issues that actually exist in the code
- Verify each issue by checking the specific line and code content
- If no security vulnerabilities exist, return empty issues array []
- Do not invent or imagine vulnerabilities
- Be extremely conservative - false positives harm credibility

SECURITY SCOPE - Only analyze these specific vulnerabilities:
- Hardcoded credentials (passwords, API keys, secrets, tokens)
- SQL injection vulnerabilities (string concatenation in queries)  
- Command injection (unsafe system calls, eval, exec)
- XSS vulnerabilities (unescaped user input in HTML)
- Path traversal vulnerabilities (../ patterns)
- Insecure cryptographic implementations
- Authentication/authorization bypass
- Input validation failures with security implications

VALIDATION REQUIREMENTS:
- For hardcoded secrets: Must be actual credentials, not examples or placeholders
- For SQL injection: Must show actual string concatenation in SQL queries
- For command injection: Must show unsafe execution of user input
- For each issue: Quote the exact problematic code as evidence

RESPONSE FORMAT - Valid JSON only:
{{
  "issues": [
    {{
      "severity": "critical|high|medium|low",
      "title": "Specific vulnerability name",
      "description": "Detailed explanation with evidence from code",
      "line_number": 123,
      "suggestion": "Specific fix recommendation",
      "category": "security",
      "vulnerability_type": "sql_injection|xss|command_injection|hardcoded_secrets|path_traversal",
      "evidence": "Brief description without quotes or code snippets"
    }}
  ],
  "metrics": {{"security_score": 0.8}},
  "confidence": 0.95
}}

REMEMBER: If the code is secure, return empty issues array."""
    
    @traceable(
        name="security_agent_analyze",
        metadata={"agent_type": "security", "component": "main_analysis"}
    )
    async def analyze(self, code: str, file_path: str, language: str) -> Dict[str, Any]:
        """Enhanced security analysis with comprehensive tracing"""
        
        # Pre-validation with detailed tracing
        validation_result = self._pre_validate_security_issues(code)
        
        # Add validation metadata to trace
        validation_metadata = {
            "static_analysis_completed": True,
            "patterns_found": validation_result['total_matches'],
            "vulnerability_categories": validation_result['categories_affected'],
            "requires_llm_analysis": validation_result['has_issues']
        }
        
        if not validation_result['has_issues']:
            print(f"[SECURITY] No security patterns detected in {file_path}")
            
            result = {
                'agent': 'security',
                'language': language,
                'file_path': file_path,
                'issues': [],
                'metrics': {
                    'security_score': 1.0, 
                    'confidence': 0.9,
                    'static_analysis': validation_result
                },
                'confidence': 0.9,
                'tokens_used': 0,
                'processing_time': 0.01,
                'llm_calls': 0,
                'status': 'completed_clean',
                'validation_metadata': validation_metadata
            }
            return result
        
        # Run full LLM analysis with enhanced context
        print(f"[SECURITY] Running enhanced analysis - found {validation_result['total_matches']} potential issues")
        
        # Add static analysis context to the analysis
        result = await super().analyze(code, file_path, language)
        
        # Enhance result with static analysis findings
        result['metrics']['static_analysis'] = validation_result
        result['validation_metadata'] = validation_metadata
        
        # Classify and score issues based on static analysis
        for issue in result.get('issues', []):
            issue['static_pattern_match'] = self._find_matching_pattern(
                issue, validation_result['pattern_matches']
            )
        
        return result
    
    @traceable(name="find_matching_pattern")
    def _find_matching_pattern(self, issue: Dict[str, Any], 
                              pattern_matches: Dict[str, List]) -> Dict[str, Any]:
        """Find which static pattern matches this LLM-detected issue"""
        issue_line = issue.get('line_number', 0)
        
        for category, matches in pattern_matches.items():
            for match in matches:
                if abs(match['line'] - issue_line) <= 2:  # Within 2 lines
                    return {
                        'category': category,
                        'pattern': match['pattern'],
                        'confidence_boost': 0.2  # Boost confidence for pattern-matched issues
                    }
        
        return {'category': 'llm_only', 'confidence_boost': 0.0}