"""Enhanced Documentation Agent with comprehensive LangSmith tracing."""

from .base_agent import BaseLLMAgent, traceable
import re
from typing import Dict, List, Any

class DocumentationAgent(BaseLLMAgent):
    """Enhanced Documentation Agent with detailed tracing and analysis"""
    
    def __init__(self, rag_analyzer=None):
        super().__init__('documentation', rag_analyzer)
        
        # Documentation-specific metadata
        self.agent_metadata.update({
            "specialization": "documentation_analysis",
            "analysis_methods": ["docstring_detection", "comment_analysis", "api_documentation"],
            "documentation_standards": ["python_docstrings", "javadoc", "jsdoc", "inline_comments"]
        })
    
    @traceable(
        name="analyze_documentation_coverage",
        metadata={"component": "static_analysis", "agent": "documentation"}
    )
    def _has_documentation_issues(self, code: str, language: str) -> Dict[str, Any]:
        """Enhanced documentation analysis with detailed metrics"""
        lines = code.split('\n')
        
        analysis_result = {
            "functions_found": 0,
            "classes_found": 0,
            "undocumented_functions": 0,
            "undocumented_classes": 0,
            "documentation_coverage": 0.0,
            "missing_docstrings": [],
            "has_issues": False,
            "complexity_analysis": {}
        }
        
        # Language-specific patterns
        if language == 'python':
            function_pattern = r'^\s*def\s+(\w+)\s*\('
            class_pattern = r'^\s*class\s+(\w+)'
            docstring_patterns = [r'^\s*"""', r"^\s*'''"]
        elif language == 'java':
            function_pattern = r'^\s*(public|private|protected).*?\s+(\w+)\s*\([^)]*\)\s*{'
            class_pattern = r'^\s*(public\s+)?(class|interface)\s+(\w+)'
            docstring_patterns = [r'^\s*/\*\*']
        elif language in ['javascript', 'typescript']:
            function_pattern = r'^\s*(function\s+(\w+)|(\w+)\s*[:=]\s*function|\s*(\w+)\s*\([^)]*\)\s*[={])'
            class_pattern = r'^\s*class\s+(\w+)'
            docstring_patterns = [r'^\s*/\*\*']
        else:
            # Generic patterns
            function_pattern = r'^\s*\w+.*?\([^)]*\)'
            class_pattern = r'^\s*(class|struct)\s+\w+'
            docstring_patterns = [r'^\s*/\*\*', r'^\s*"""']
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for function definitions
            func_match = re.match(function_pattern, line)
            if func_match:
                analysis_result["functions_found"] += 1
                
                if language == 'python':
                    func_name = func_match.group(1)
                elif language == 'java':
                    func_name = func_match.group(2) if func_match.group(2) else "unknown"
                else:
                    func_name = func_match.group(2) or func_match.group(3) or func_match.group(4) or "unknown"
                
                # Check for documentation
                has_docstring = self._check_for_docstring(lines, i + 1, docstring_patterns)
                
                if not has_docstring:
                    analysis_result["undocumented_functions"] += 1
                    analysis_result["missing_docstrings"].append({
                        "type": "function",
                        "name": func_name,
                        "line": i + 1,
                        "complexity": self._estimate_function_complexity(lines, i)
                    })
            
            # Check for class definitions
            class_match = re.match(class_pattern, line)
            if class_match:
                analysis_result["classes_found"] += 1
                
                if language == 'python':
                    class_name = class_match.group(1)
                elif language == 'java':
                    class_name = class_match.group(3) if class_match.group(3) else "unknown"
                else:
                    class_name = class_match.group(1) if class_match.group(1) else "unknown"
                
                # Check for documentation
                has_docstring = self._check_for_docstring(lines, i + 1, docstring_patterns)
                
                if not has_docstring:
                    analysis_result["undocumented_classes"] += 1
                    analysis_result["missing_docstrings"].append({
                        "type": "class",
                        "name": class_name,
                        "line": i + 1,
                        "public_methods": self._count_public_methods(lines, i, language)
                    })
            
            i += 1
        
        # Calculate coverage
        total_items = analysis_result["functions_found"] + analysis_result["classes_found"]
        if total_items > 0:
            documented_items = total_items - len(analysis_result["missing_docstrings"])
            analysis_result["documentation_coverage"] = documented_items / total_items
        
        # Determine if there are significant issues
        analysis_result["has_issues"] = (
            analysis_result["undocumented_functions"] >= 2 or 
            analysis_result["undocumented_classes"] >= 1 or 
            (total_items > 5 and analysis_result["documentation_coverage"] < 0.6)
        )
        
        return analysis_result
    
    @traceable(name="check_docstring_presence")
    def _check_for_docstring(self, lines: List[str], start_idx: int, patterns: List[str]) -> bool:
        """Check if docstring exists after function/class definition"""
        for i in range(start_idx, min(start_idx + 3, len(lines))):
            if i >= len(lines):
                break
            
            line = lines[i].strip()
            if not line:
                continue
            
            for pattern in patterns:
                if re.match(pattern, line):
                    return True
        
        return False
    
    @traceable(name="estimate_function_complexity")
    def _estimate_function_complexity(self, lines: List[str], func_start: int) -> str:
        """Estimate function complexity for documentation priority"""
        # Simple heuristic: count control structures and parameters
        complexity_indicators = 0
        
        # Look at the function signature for parameter count
        func_line = lines[func_start] if func_start < len(lines) else ""
        param_count = func_line.count(',') + (1 if '(' in func_line and ')' in func_line else 0)
        
        # Look ahead for complexity indicators
        for i in range(func_start, min(func_start + 20, len(lines))):
            if i >= len(lines):
                break
            
            line = lines[i].lower()
            complexity_indicators += line.count('if ') + line.count('for ') + line.count('while ')
            complexity_indicators += line.count('try:') + line.count('except:')
        
        if param_count > 3 or complexity_indicators > 5:
            return "high"
        elif param_count > 1 or complexity_indicators > 2:
            return "medium"
        else:
            return "low"
    
    @traceable(name="count_public_methods")
    def _count_public_methods(self, lines: List[str], class_start: int, language: str) -> int:
        """Count public methods in a class"""
        method_count = 0
        indent_level = len(lines[class_start]) - len(lines[class_start].lstrip()) if class_start < len(lines) else 0
        
        for i in range(class_start + 1, min(class_start + 50, len(lines))):
            if i >= len(lines):
                break
            
            line = lines[i]
            current_indent = len(line) - len(line.lstrip())
            
            # Stop if we've left the class
            if line.strip() and current_indent <= indent_level:
                break
            
            # Check for method definitions
            if language == 'python':
                if re.match(r'^\s*def\s+\w+', line) and not line.strip().startswith('def _'):
                    method_count += 1
            elif language == 'java':
                if re.match(r'^\s*public\s+.*\s+\w+\s*\(', line):
                    method_count += 1
        
        return method_count
    
    def get_system_prompt(self, language: str) -> str:
        return f"""You are a DOCUMENTATION specialist for {language} code.

CRITICAL ACCURACY REQUIREMENTS:

* ONLY report documentation issues that actually impact code understanding
* Verify that functions/classes truly lack necessary documentation
* If documentation is adequate, return empty issues array []
* Focus on public APIs and complex functions that need documentation
* Do not report minor or cosmetic documentation issues

DOCUMENTATION SCOPE only analyze these specific issues:

* Missing docstrings for public functions/methods (especially complex ones)
* Missing docstrings for public classes and modules
* Missing parameter descriptions for functions with more than 3 parameters
* Missing return value descriptions for complex functions
* Unclear or misleading function/variable names
* Missing inline comments for complex logic sections

VALIDATION REQUIREMENTS:

* For missing docstrings: Verify the function/class is public and non-trivial
* For parameter docs: Count actual parameters and assess complexity
* For unclear names: Show specific examples of confusing names
* Each issue must reference actual code that lacks necessary documentation

FOCUS ON:

* Public APIs that other developers will use
* Complex algorithms that need explanation
* Functions with multiple parameters or return types
* Classes with non-obvious behavior

RESPONSE FORMAT Valid JSON only:
{{
"issues": [
{{
"severity": "medium|low",
"title": "Specific documentation issue",
"description": "What documentation is missing and why it's needed",
"line_number": 123,
"suggestion": "Specific documentation to add",
"category": "documentation",
"documentation_type": "missing_docstring|unclear_naming|missing_comments",
"complexity_level": "high|medium|low",
"evidence": "Brief description without quotes or special characters"
}}
],
"metrics": {{"documentation_coverage": 0.4}},
"confidence": 0.80
}}

REMEMBER:

* If the code is simple or already well-documented, return empty issues array.
* Only flag documentation gaps that meaningfully affect comprehension of the code.
* Base your evaluation on reasoning about the code's purpose and complexity, not automated counts.
"""    
    @traceable(
        name="documentation_agent_analyze",
        metadata={"agent_type": "documentation", "component": "main_analysis"}
    )
    async def analyze(self, code: str, file_path: str, language: str) -> Dict[str, Any]:
        """Enhanced documentation analysis with comprehensive tracing"""
        
        # Comprehensive documentation analysis
        analysis_result = self._has_documentation_issues(code, language)
        
        # Create detailed metadata for tracing
        analysis_metadata = {
            "static_analysis_completed": True,
            "functions_analyzed": analysis_result["functions_found"],
            "classes_analyzed": analysis_result["classes_found"],
            "documentation_coverage": analysis_result["documentation_coverage"],
            "undocumented_items": len(analysis_result["missing_docstrings"]),
            "requires_llm_analysis": analysis_result["has_issues"]
        }
        
        if not analysis_result["has_issues"]:
            print(f"[DOCUMENTATION] Good documentation coverage ({analysis_result['documentation_coverage']:.1%}) in {file_path}")
            
            result = {
                'agent': 'documentation',
                'language': language,
                'file_path': file_path,
                'issues': [],
                'metrics': {
                    'documentation_coverage': analysis_result["documentation_coverage"],
                    'confidence': 0.9,
                    'static_analysis': analysis_result
                },
                'confidence': 0.9,
                'tokens_used': 0,
                'processing_time': 0.01,
                'llm_calls': 0,
                'status': 'completed_clean',
                'analysis_metadata': analysis_metadata
            }
            return result
        
        # Run enhanced LLM analysis with documentation context
        print(f"[DOCUMENTATION] Found {len(analysis_result['missing_docstrings'])} documentation issues, enhancing with LLM...")
        
        # Create enhanced prompt with static analysis context
        enhanced_prompt = self._create_documentation_prompt(code, file_path, language, analysis_result)
        
        # Run LLM analysis
        result = await self._run_documentation_analysis(enhanced_prompt, file_path, language)
        
        # Enhance result with static analysis findings
        result['metrics']['static_analysis'] = analysis_result
        result['analysis_metadata'] = analysis_metadata
        
        # Add documentation-specific insights to issues
        for issue in result.get('issues', []):
            issue['documentation_analysis'] = self._analyze_documentation_context(
                issue, analysis_result['missing_docstrings']
            )
        
        return result
    
    @traceable(name="create_documentation_prompt")
    def _create_documentation_prompt(self, code: str, file_path: str, language: str, 
                                   analysis_result: Dict[str, Any]) -> str:
        """Create enhanced prompt with documentation analysis context"""
        
        # Summarize missing documentation
        missing_summary = []
        for item in analysis_result['missing_docstrings']:
            missing_summary.append(f"- {item['type'].title()} '{item['name']}' at line {item['line']} (complexity: {item.get('complexity', 'unknown')})")
        
        system_prompt = self.get_system_prompt(language)
        
        # Create numbered code for better line reference
        numbered_lines = []
        for i, line in enumerate(code.split('\n'), 1):
            numbered_lines.append(f"{i:4d}: {line}")
        numbered_code = '\n'.join(numbered_lines)

        enhanced_prompt = f"""
{system_prompt}

CODE TO ANALYZE:
File: {file_path}
Language: {language}

Code (with line numbers):
```{language}
{numbered_code}
```

DOCUMENTATION CONTEXT:
- Focus on the identified missing documentation items above
- Prioritize high-complexity functions and public APIs
- Consider the maintainability impact of missing documentation
- Provide specific documentation templates and examples

Analyze the code for documentation gaps and provide actionable recommendations.
"""
        return enhanced_prompt
    
    @traceable(name="run_documentation_analysis")
    async def _run_documentation_analysis(self, prompt: str, file_path: str, language: str) -> Dict[str, Any]:
        """Run documentation analysis with enhanced context"""
        try:
            response = await self.llm.generate(prompt, max_tokens=1500)

            # Check if response is empty or invalid
            if not response or not response.strip():
                print(f"[DOCUMENTATION] Empty response from LLM")
                return {
                    "issues": [],
                    "metrics": {"confidence": 0.0, "empty_response": True},
                    "confidence": 0.0
                }

            result = self._parse_and_validate_response(response, "", file_path)

            # Check if parsing failed
            if result.get('metrics', {}).get('parsing_failed'):
                print(f"[DOCUMENTATION] JSON parsing failed for response: {response[:100]}...")
                return {
                    "issues": [],
                    "metrics": {"confidence": 0.0, "parsing_failed": True},
                    "confidence": 0.0
                }

            # Add documentation-specific metadata
            result['analysis_type'] = 'documentation_analysis'
            result['status'] = 'completed_enhanced'

            return result

        except Exception as e:
            print(f"[DOCUMENTATION] LLM analysis failed: {e}")
            return {
                "issues": [],
                "metrics": {"confidence": 0.0, "llm_failed": True, "error": str(e)},
                "confidence": 0.0
            }
    
    @traceable(name="analyze_documentation_context")
    def _analyze_documentation_context(self, issue: Dict[str, Any], 
                                     missing_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze documentation context for an issue"""
        issue_line = issue.get('line_number', 0)
        matching_items = []
        
        for item in missing_items:
            if abs(item['line'] - issue_line) <= 2:  # Within 2 lines
                matching_items.append({
                    'type': item['type'],
                    'name': item['name'],
                    'complexity': item.get('complexity', 'unknown'),
                    'line_distance': abs(item['line'] - issue_line)
                })
        
        # Determine documentation priority
        priority = "low"
        if matching_items:
            for item in matching_items:
                if item['complexity'] == 'high':
                    priority = "high"
                    break
                elif item['complexity'] == 'medium':
                    priority = "medium"
        
        return {
            'matching_items': matching_items,
            'static_analysis_confirmed': len(matching_items) > 0,
            'documentation_priority': priority,
            'confidence_boost': 0.2 if matching_items else 0.0
        }