"""Enhanced LLM-powered security analysis agent with improved accuracy."""

from .base_agent import BaseLLMAgent
import re


class SecurityAgent(BaseLLMAgent):
    """Enhanced LLM-powered security analysis agent"""
    
    def __init__(self, rag_analyzer=None):
        super().__init__('security', rag_analyzer)
        self.security_patterns = self._init_security_patterns()
    
    def _init_security_patterns(self):
        """Initialize security patterns for pre-validation"""
        return {
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']{3,}["\']',
                r'api_key\s*=\s*["\'][^"\']{10,}["\']',
                r'secret\s*=\s*["\'][^"\']{8,}["\']',
                r'token\s*=\s*["\'][^"\']{10,}["\']',
                r'["\'][A-Za-z0-9]{20,}["\']',  # Long strings that might be keys
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
            ]
        }
    
    def _pre_validate_security_issues(self, code: str) -> bool:
        """Check if code actually contains potential security issues"""
        for category, patterns in self.security_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    return True
        return False
    
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

DO NOT report:
- General code quality issues
- Performance problems  
- Documentation issues
- Theoretical vulnerabilities without evidence
- False positives from comments or test code

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
      "evidence": "Brief description without quotes or code snippets"
    }}
  ],
  "metrics": {{"security_score": 0.8}},
  "confidence": 0.95
}}

CRITICAL: For the "evidence" field, provide a simple description only - NO code snippets, quotes, or special characters that could break JSON parsing.

REMEMBER: If the code is secure, return empty issues array. Quality and accuracy over quantity."""
    
    async def analyze(self, code: str, file_path: str, language: str):
        """Enhanced security analysis with pre-validation"""
        
        # Pre-check: does code contain potential security issues?
        if not self._pre_validate_security_issues(code):
            print(f"[SECURITY] No security patterns detected in {file_path}")
            return {
                'agent': 'security',
                'language': language,
                'file_path': file_path,
                'issues': [],
                'metrics': {'security_score': 1.0, 'confidence': 0.9},
                'confidence': 0.9,
                'tokens_used': 0,
                'processing_time': 0.01
            }
        
        # Run full analysis if patterns found
        return await super().analyze(code, file_path, language)