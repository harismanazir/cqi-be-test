"""Enhanced LLM-powered performance analysis agent with improved accuracy."""

from .base_agent import BaseLLMAgent
import re


class PerformanceAgent(BaseLLMAgent):
    """Enhanced LLM-powered performance analysis agent"""
    
    def __init__(self, rag_analyzer=None):
        super().__init__('performance', rag_analyzer)
        self.performance_patterns = self._init_performance_patterns()
    
    def _init_performance_patterns(self):
        """Initialize performance anti-patterns for pre-validation"""
        return {
            'inefficient_loops': [
                r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(',
                r'for\s+i\s+in\s+range\s*\(\s*len\s*\([^)]+\)\s*\)\s*:.*\[\s*i\s*\]',
                r'while.*len\s*\(',
            ],
            'string_concatenation': [
                r'\w+\s*\+=\s*["\']',
                r'\w+\s*=\s*\w+\s*\+\s*["\']',
                r'["\'].*\+.*["\'].*\+',
            ],
            'repeated_calculations': [
                r'for.*range.*:.*len\s*\(',
                r'while.*:.*len\s*\(',
            ],
            'inefficient_data_structures': [
                r'list\s*\(\s*\).*append',
                r'\[\].*append.*for.*in',
            ],
            'database_issues': [
                r'for.*in.*query',
                r'execute.*for.*in',
            ]
        }
    
    def _pre_validate_performance_issues(self, code: str) -> bool:
        """Check if code actually contains potential performance issues"""
        for category, patterns in self.performance_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    return True
        return False
    
    def get_system_prompt(self, language: str) -> str:
        return f"""You are a PERFORMANCE OPTIMIZATION specialist for {language} code.

CRITICAL ACCURACY REQUIREMENTS:
- ONLY report performance issues that actually exist and can be measured
- Verify each issue by examining the specific code patterns
- If no performance problems exist, return empty issues array []
- Do not report theoretical or minor optimizations
- Focus on issues that would have measurable impact

PERFORMANCE SCOPE - Only analyze these specific issues:
- Inefficient algorithms (O(n²) when O(n) exists, nested loops)
- Inefficient data structure usage (wrong data type for the operation)
- Memory leaks and resource management (unclosed files, connections)
- String concatenation in loops (especially in Python)
- Database query inefficiencies (N+1 queries, missing indexes)
- Unnecessary repeated calculations in loops
- I/O operations in tight loops
- Inefficient iteration patterns (range(len()) instead of enumerate)

VALIDATION REQUIREMENTS:
- For loop inefficiencies: Show actual nested loops or range(len()) patterns
- For string concatenation: Show actual string building in loops
- For data structures: Show clear misuse (like using list for O(1) lookups)
- Each issue must include evidence from the actual code

DO NOT report:
- Security vulnerabilities
- Code complexity or readability issues
- Documentation problems
- Theoretical micro-optimizations
- Style or formatting issues

RESPONSE FORMAT - Valid JSON only:
{{
  "issues": [
    {{
      "severity": "high|medium|low",
      "title": "Specific performance issue",
      "description": "Explanation of performance impact with evidence",
      "line_number": 123,
      "suggestion": "Specific optimization recommendation",
      "category": "performance",
      "evidence": "Brief description without quotes or special characters"
    }}
  ],
  "metrics": {{"performance_score": 0.7}},
  "confidence": 0.85
}}

CRITICAL: For the "evidence" field, provide simple descriptions only - NO code snippets, quotes, or special characters that could break JSON parsing.

REMEMBER: If the code is already efficient, return empty issues array. Focus on real bottlenecks."""
    
    async def analyze(self, code: str, file_path: str, language: str):
        """Enhanced performance analysis with pre-validation"""
        
        # Pre-check: does code contain potential performance issues?
        if not self._pre_validate_performance_issues(code):
            print(f"[PERFORMANCE] No performance anti-patterns detected in {file_path}")
            return {
                'agent': 'performance',
                'language': language,
                'file_path': file_path,
                'issues': [],
                'metrics': {'performance_score': 1.0, 'confidence': 0.9},
                'confidence': 0.9,
                'tokens_used': 0,
                'processing_time': 0.01,
                'llm_calls': 0
            }
        
        # Run full analysis if patterns found
        return await super().analyze(code, file_path, language)