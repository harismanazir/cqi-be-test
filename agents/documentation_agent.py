"""Enhanced LLM-powered documentation analysis agent with improved accuracy."""

from .base_agent import BaseLLMAgent
import re


class DocumentationAgent(BaseLLMAgent):
    """Enhanced LLM-powered documentation analysis agent"""
    
    def __init__(self, rag_analyzer=None):
        super().__init__('documentation', rag_analyzer)
    
    def _has_documentation_issues(self, code: str, language: str) -> bool:
        """Check if code actually has documentation issues"""
        lines = code.split('\n')
        
        # Find function/class definitions
        function_pattern = r'^\s*def\s+\w+\s*\('
        class_pattern = r'^\s*class\s+\w+'
        
        undocumented_functions = 0
        undocumented_classes = 0
        total_functions = 0
        total_classes = 0
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for function definitions
            if re.match(function_pattern, line):
                total_functions += 1
                # Check if next non-empty line is a docstring
                j = i + 1
                has_docstring = False
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    next_line = lines[j].strip()
                    if next_line.startswith('"""') or next_line.startswith("'''"):
                        has_docstring = True
                if not has_docstring:
                    undocumented_functions += 1
            
            # Check for class definitions
            elif re.match(class_pattern, line):
                total_classes += 1
                # Check if next non-empty line is a docstring
                j = i + 1
                has_docstring = False
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    next_line = lines[j].strip()
                    if next_line.startswith('"""') or next_line.startswith("'''"):
                        has_docstring = True
                if not has_docstring:
                    undocumented_classes += 1
            
            i += 1
        
        # Only flag as having issues if there are multiple undocumented items
        return (undocumented_functions >= 2 or undocumented_classes >= 1 or 
                (total_functions > 5 and undocumented_functions > 0))
    
    def get_system_prompt(self, language: str) -> str:
        return f"""You are a DOCUMENTATION specialist for {language} code.

CRITICAL ACCURACY REQUIREMENTS:
- ONLY report documentation issues that actually impact code understanding
- Verify that functions/classes truly lack necessary documentation
- If documentation is adequate, return empty issues array []
- Focus on public APIs and complex functions that need documentation
- Do not report minor or cosmetic documentation issues

DOCUMENTATION SCOPE - Only analyze these specific issues:
- Missing docstrings for public functions/methods (especially complex ones)
- Missing docstrings for public classes and modules
- Missing parameter descriptions for functions with >3 parameters
- Missing return value descriptions for complex functions
- Unclear or misleading function/variable names
- Missing inline comments for complex logic sections

VALIDATION REQUIREMENTS:
- For missing docstrings: Verify function/class is public and non-trivial
- For parameter docs: Count actual parameters and complexity
- For unclear names: Show specific examples of confusing names
- Each issue must reference actual code that lacks necessary documentation

FOCUS ON:
- Public APIs that other developers will use
- Complex algorithms that need explanation
- Functions with multiple parameters or return types
- Classes with non-obvious behavior

DO NOT report:
- Security vulnerabilities
- Performance issues
- Code complexity problems
- Missing docs for trivial getters/setters
- Style or formatting issues
- Documentation for internal/private methods (unless very complex)

RESPONSE FORMAT - Valid JSON only:
{{
  "issues": [
    {{
      "severity": "medium|low",
      "title": "Specific documentation issue",
      "description": "What documentation is missing and why it's needed",
      "line_number": 123,
      "suggestion": "Specific documentation to add",
      "category": "documentation",
      "evidence": "Brief description without quotes or special characters"
    }}
  ],
  "metrics": {{"documentation_coverage": 0.4}},
  "confidence": 0.80
}}

CRITICAL: For the "evidence" field, provide simple descriptions only - NO code snippets, quotes, or special characters that could break JSON parsing.

REMEMBER: If code is simple or already well-documented, return empty issues array. Focus on meaningful documentation gaps."""
    
    async def analyze(self, code: str, file_path: str, language: str):
        """Enhanced documentation analysis with pre-validation"""
        
        # Pre-check: does code have documentation issues?
        if not self._has_documentation_issues(code, language):
            print(f"[DOCUMENTATION] No significant documentation issues in {file_path}")
            return {
                'agent': 'documentation',
                'language': language,
                'file_path': file_path,
                'issues': [],
                'metrics': {'documentation_coverage': 1.0, 'confidence': 0.9},
                'confidence': 0.9,
                'tokens_used': 0,
                'processing_time': 0.01,
                'llm_calls': 0
            }
        
        # Run full analysis if issues found
        return await super().analyze(code, file_path, language)