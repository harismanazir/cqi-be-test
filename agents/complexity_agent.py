"""Enhanced LLM-powered complexity analysis agent with improved accuracy."""

from .base_agent import BaseLLMAgent
import re


class ComplexityAgent(BaseLLMAgent):
    """Enhanced LLM-powered complexity analysis agent"""
    
    def __init__(self, rag_analyzer=None):
        super().__init__('complexity', rag_analyzer)
        self.complexity_patterns = self._init_complexity_patterns()
    
    def _init_complexity_patterns(self):
        """Initialize complexity anti-patterns for pre-validation"""
        return {
            'long_functions': r'def\s+\w+[^:]*:\s*(?:\n.*){50,}',  # Functions longer than 50 lines
            'deep_nesting': r'(\s{8,}if|\s{12,}for|\s{12,}while)',  # 3+ levels of indentation
            'many_parameters': r'def\s+\w+\s*\([^)]{100,}\)',  # Very long parameter lists
            'long_lines': r'.{120,}',  # Lines longer than 120 characters
            'complex_expressions': r'[^=]*==.*and.*or.*or.*and',  # Complex boolean expressions
            'too_many_returns': r'def[^:]+:(?:(?:\n.*)*?return.*){4,}',  # Multiple return statements
        }
    
    def _pre_validate_complexity_issues(self, code: str) -> bool:
        """Check if code actually contains complexity issues using static analysis"""
        lines = code.split('\n')

        # More thorough analysis with proper language detection
        language = self._detect_language(code)

        # Get function/method definitions
        functions = self._extract_functions(lines, language)

        # Check each function for complexity issues
        has_issues = False

        for func_info in functions:
            # Check function length (exclude comments and blank lines)
            actual_lines = self._count_actual_lines(func_info['body'])
            if actual_lines > 50:
                print(f"[COMPLEXITY-DEBUG] Function {func_info['name']} has {actual_lines} lines (>50)")
                has_issues = True

            # Check parameter count
            param_count = self._count_parameters(func_info['signature'])
            if param_count > 7:
                print(f"[COMPLEXITY-DEBUG] Function {func_info['name']} has {param_count} parameters (>7)")
                has_issues = True

            # Check nesting depth
            max_nesting = self._calculate_max_nesting(func_info['body'], language)
            if max_nesting > 5:
                print(f"[COMPLEXITY-DEBUG] Function {func_info['name']} has nesting depth {max_nesting} (>5)")
                has_issues = True

        # Check for long lines
        for i, line in enumerate(lines):
            if len(line) > 120:
                print(f"[COMPLEXITY-DEBUG] Line {i+1} has {len(line)} characters (>120)")
                has_issues = True

        return has_issues

    def _detect_language(self, code: str) -> str:
        """Detect programming language from code"""
        if re.search(r'def\s+\w+|import\s+\w+|from\s+\w+', code):
            return 'python'
        elif re.search(r'public\s+class|private\s+\w+|import\s+java', code):
            return 'java'
        elif re.search(r'function\s+\w+|const\s+\w+|var\s+\w+', code):
            return 'javascript'
        return 'unknown'

    def _extract_functions(self, lines: list, language: str) -> list:
        """Extract function definitions and their bodies"""
        functions = []
        current_function = None

        for i, line in enumerate(lines):
            # Check for function start
            if language == 'python':
                match = re.match(r'^(\s*)def\s+(\w+)\s*\(([^)]*)\)\s*:', line)
                if match:
                    indent, name, params = match.groups()
                    current_function = {
                        'name': name,
                        'signature': params,
                        'start_line': i,
                        'indent': len(indent),
                        'body': []
                    }
                    functions.append(current_function)
                    continue
            elif language == 'java':
                match = re.match(r'^\s*(public|private|protected).*?\s+(\w+)\s*\(([^)]*)\)\s*{?', line)
                if match:
                    access, name, params = match.groups()
                    current_function = {
                        'name': name,
                        'signature': params,
                        'start_line': i,
                        'indent': len(line) - len(line.lstrip()),
                        'body': []
                    }
                    functions.append(current_function)
                    continue

            # Add line to current function body
            if current_function is not None:
                line_indent = len(line) - len(line.lstrip())

                # Check if we're still in the function
                if line.strip() and line_indent <= current_function['indent'] and not line.startswith(' ' * (current_function['indent'] + 1)):
                    # Function ended
                    current_function = None
                else:
                    current_function['body'].append(line)

        return functions

    def _count_actual_lines(self, body_lines: list) -> int:
        """Count non-empty, non-comment lines"""
        count = 0
        for line in body_lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('//') and not stripped.startswith('*'):
                count += 1
        return count

    def _count_parameters(self, signature: str) -> int:
        """Count function parameters"""
        if not signature.strip():
            return 0

        # Simple parameter counting - split by comma but handle nested structures
        params = signature.split(',')
        # Filter out empty parameters
        actual_params = [p.strip() for p in params if p.strip()]
        return len(actual_params)

    def _calculate_max_nesting(self, body_lines: list, language: str) -> int:
        """Calculate maximum nesting depth in function body"""
        max_nesting = 0

        for line in body_lines:
            if not line.strip():
                continue

            # Calculate nesting level based on indentation
            if language == 'python':
                # For Python, count by 4-space indents
                base_indent = 4  # Assume function body starts at 4 spaces
                line_indent = len(line) - len(line.lstrip())
                nesting = max(0, (line_indent - base_indent) // 4)
            else:
                # For other languages, count braces or similar
                nesting = line.count('{') - line.count('}')

            max_nesting = max(max_nesting, nesting)

        return max_nesting
    
    def get_system_prompt(self, language: str) -> str:
        return f"""You are a CODE COMPLEXITY specialist for {language} code.

CRITICAL ACCURACY REQUIREMENTS:
- ONLY report complexity issues that actually exist and measurably impact maintainability
- MANUALLY count lines, nesting levels, and parameters before reporting
- If the code is well-structured and readable, return empty issues array []
- Do not report subjective or minor complexity issues
- Focus ONLY on objectively measurable complexity problems

COMPLEXITY THRESHOLDS - Only report if these thresholds are ACTUALLY exceeded:
- Long functions/methods: >50 lines of actual code (excluding comments/blank lines)
- Deep nesting levels: >5 levels of indentation (count spaces or braces)
- Too many parameters: >7 parameters in a single function
- Very long lines: >120 characters that hurt readability
- High cyclomatic complexity: >10 decision points (if/else/for/while/switch)

MANDATORY VERIFICATION PROCESS:
1. For long functions: MANUALLY count non-empty, non-comment lines in each function
2. For deep nesting: MANUALLY count indentation levels at the deepest point
3. For parameters: MANUALLY count comma-separated parameters in function signatures
4. For line length: MANUALLY check character count of each line
5. ONLY report if manual verification confirms the threshold is exceeded

MEASUREMENTS TO INCLUDE (with actual counts):
- Function length: "Function has X lines of code" (where X is manually counted)
- Nesting depth: "Code nested X levels deep at line Y" (where X is manually counted)
- Parameter count: "Function has X parameters" (where X is manually counted)
- Line length: "Line X has Y characters" (where Y is manually counted)

EXAMPLES OF ACCURATE REPORTING:
- GOOD: "Function processData has 75 lines of code" (if manually counted to be 75+)
- BAD: "Function processData has 75 lines of code" (if actually only 20 lines)
- GOOD: "Code nested 6 levels deep at line 45" (if manually verified to be 6+ levels)
- BAD: "Code nested 5 levels deep" (if actually only 3 levels)

DO NOT report:
- Security vulnerabilities
- Performance issues
- Documentation problems
- Minor style preferences
- Theoretical complexity without manual verification
- Issues where manual count doesn't exceed thresholds

RESPONSE FORMAT - Valid JSON only:
{{
  "issues": [
    {{
      "severity": "high|medium|low",
      "title": "Specific complexity issue with verified measurement",
      "description": "Detailed explanation with manually verified metrics",
      "line_number": 123,
      "suggestion": "Specific refactoring recommendation",
      "category": "complexity",
      "evidence": "Manually verified measurement without special characters"
    }}
  ],
  "metrics": {{"complexity_score": 0.6}},
  "confidence": 0.90
}}

CRITICAL: For the "evidence" field, provide simple manually verified measurements only - NO quotes, code snippets, or special characters.

FINAL REMINDER: If manual verification shows the code is well-structured and under thresholds, return empty issues array. Accuracy is more important than finding issues."""
    
    async def analyze(self, code: str, file_path: str, language: str):
        """Enhanced complexity analysis with static detection + LLM enhancement"""

        # Perform static analysis first to detect issues
        static_issues = self._perform_static_analysis(code, file_path, language)

        if static_issues is not None:
            print(f"[COMPLEXITY] Static analysis found {len(static_issues)} verified issues in {file_path}")

            # If we have issues, enhance them with LLM context
            if static_issues:
                print(f"[COMPLEXITY] Enhancing descriptions with LLM context...")
                enhanced_issues = await self._enhance_issues_with_llm(code, static_issues, language)

                return {
                    'agent': 'complexity',
                    'language': language,
                    'file_path': file_path,
                    'issues': enhanced_issues,
                    'metrics': {'complexity_score': 0.8, 'confidence': 0.95},
                    'confidence': 0.95,
                    'tokens_used': len(enhanced_issues) * 100,  # Estimate
                    'processing_time': 0.5,
                    'llm_calls': 1
                }
            else:
                # No issues found, return clean result
                return {
                    'agent': 'complexity',
                    'language': language,
                    'file_path': file_path,
                    'issues': [],
                    'metrics': {'complexity_score': 1.0, 'confidence': 0.95},
                    'confidence': 0.95,
                    'tokens_used': 0,
                    'processing_time': 0.01,
                    'llm_calls': 0
                }

        # Fall back to LLM analysis for edge cases
        print(f"[COMPLEXITY] Running full LLM analysis for {file_path}")
        return await super().analyze(code, file_path, language)

    def _perform_static_analysis(self, code: str, file_path: str, language: str):
        """Perform comprehensive static analysis and return issues directly"""
        lines = code.split('\n')
        detected_language = self._detect_language(code)
        functions = self._extract_functions(lines, detected_language)

        issues = []

        # Check each function for complexity issues
        for func_info in functions:
            # Check function length (exclude comments and blank lines)
            actual_lines = self._count_actual_lines(func_info['body'])
            if actual_lines > 50:
                print(f"[COMPLEXITY-DEBUG] Function {func_info['name']} has {actual_lines} lines (>50)")
                issues.append({
                    'severity': 'high',
                    'title': f"Function {func_info['name']} has {actual_lines} lines of code",
                    'description': f"The function {func_info['name']} has {actual_lines} lines of code, which exceeds the recommended threshold of 50 lines and may make it difficult to understand and maintain.",
                    'line_number': func_info['start_line'] + 1,
                    'suggestion': "Consider breaking this function into smaller, more focused functions with single responsibilities.",
                    'category': 'complexity',
                    'evidence': f"Function has {actual_lines} lines of code"
                })

            # Check parameter count
            param_count = self._count_parameters(func_info['signature'])
            if param_count > 7:
                print(f"[COMPLEXITY-DEBUG] Function {func_info['name']} has {param_count} parameters (>7)")
                issues.append({
                    'severity': 'high' if param_count > 10 else 'medium',
                    'title': f"Function {func_info['name']} has {param_count} parameters",
                    'description': f"The function {func_info['name']} has {param_count} parameters, which exceeds the recommended threshold of 7 parameters and may make it difficult to understand and use.",
                    'line_number': func_info['start_line'] + 1,
                    'suggestion': "Consider using a data class, configuration object, or breaking the function into smaller functions.",
                    'category': 'complexity',
                    'evidence': f"Function has {param_count} parameters"
                })

            # Check nesting depth
            max_nesting = self._calculate_max_nesting(func_info['body'], detected_language)
            if max_nesting > 5:
                print(f"[COMPLEXITY-DEBUG] Function {func_info['name']} has nesting depth {max_nesting} (>5)")
                issues.append({
                    'severity': 'medium',
                    'title': f"Function {func_info['name']} has deep nesting ({max_nesting} levels)",
                    'description': f"The function {func_info['name']} has {max_nesting} levels of nesting, which exceeds the recommended threshold of 5 levels and may make it difficult to understand.",
                    'line_number': func_info['start_line'] + 1,
                    'suggestion': "Consider extracting nested logic into separate functions or using early returns to reduce nesting.",
                    'category': 'complexity',
                    'evidence': f"Code nested {max_nesting} levels deep"
                })

        # Check for long lines
        for i, line in enumerate(lines):
            if len(line) > 120:
                print(f"[COMPLEXITY-DEBUG] Line {i+1} has {len(line)} characters (>120)")
                issues.append({
                    'severity': 'low',
                    'title': f"Line {i+1} has {len(line)} characters",
                    'description': f"Line {i+1} has {len(line)} characters, which exceeds the recommended threshold of 120 characters and may hurt readability.",
                    'line_number': i + 1,
                    'suggestion': "Consider breaking this line into multiple lines for better readability and maintainability.",
                    'category': 'complexity',
                    'evidence': f"Line has {len(line)} characters"
                })

        # Return the static analysis results
        return issues

    async def _enhance_issues_with_llm(self, code: str, static_issues: list, language: str):
        """Enhance static analysis issues with LLM-generated contextual descriptions"""
        if not static_issues:
            return static_issues

        lines = code.split('\n')
        enhanced_issues = []

        for issue in static_issues:
            line_number = issue.get('line_number', 1)

            # Get context around the issue (5 lines before and after)
            start_line = max(0, line_number - 6)
            end_line = min(len(lines), line_number + 5)
            context_lines = lines[start_line:end_line]

            # Add line numbers to context
            context_with_numbers = []
            for i, line in enumerate(context_lines):
                actual_line_num = start_line + i + 1
                marker = " >>> " if actual_line_num == line_number else "     "
                context_with_numbers.append(f"{actual_line_num:3d}{marker}{line}")

            context = "\n".join(context_with_numbers)

            # Create enhancement prompt
            enhancement_prompt = f"""You are a code analysis expert. I have detected a complexity issue using static analysis, and I need you to provide a better, more contextual description.

STATIC ANALYSIS FINDINGS:
- Issue Type: {issue['title']}
- Severity: {issue['severity']}
- Basic Description: {issue['description']}
- Line: {line_number}

CODE CONTEXT ({language}):
```
{context}
```

Please provide:
1. A more contextual and specific description that references the actual code
2. Why this specific pattern is problematic in this context
3. A specific, actionable suggestion for this particular case

Keep it concise (2-3 sentences) and focus on the actual code shown. Respond in JSON format:
{{
  "enhanced_description": "Your enhanced description here",
  "specific_suggestion": "Your specific suggestion here"
}}"""

            try:
                # Use the base agent's LLM call
                response = await self._call_llm(enhancement_prompt)

                if response and 'enhanced_description' in response:
                    # Update the issue with enhanced descriptions
                    enhanced_issue = issue.copy()
                    enhanced_issue['description'] = response['enhanced_description']
                    enhanced_issue['suggestion'] = response.get('specific_suggestion', issue['suggestion'])
                    enhanced_issues.append(enhanced_issue)
                else:
                    # Fall back to original if enhancement fails
                    enhanced_issues.append(issue)

            except Exception as e:
                print(f"[COMPLEXITY] Failed to enhance issue: {e}")
                # Fall back to original issue
                enhanced_issues.append(issue)

        return enhanced_issues

    async def _call_llm(self, prompt: str):
        """Call LLM with error handling and JSON parsing"""
        try:
            # Use the LLM directly
            if hasattr(self, 'llm') and self.llm:
                response = await self.llm.generate(prompt, max_tokens=500)

                # Try to parse as JSON
                import json
                if isinstance(response, str):
                    # Clean the response to extract JSON
                    response = response.strip()
                    if response.startswith('```json'):
                        response = response[7:]
                    if response.endswith('```'):
                        response = response[:-3]

                    return json.loads(response.strip())

                return response
            else:
                print("[COMPLEXITY] No LLM available for enhancement")
                return None

        except Exception as e:
            print(f"[COMPLEXITY] LLM enhancement error: {e}")
            return None
    