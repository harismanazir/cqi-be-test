"""Enhanced Complexity Agent with comprehensive LangSmith tracing."""

import re
from .base_agent import BaseLLMAgent, traceable
from typing import Dict, List, Any, Tuple

class ComplexityAgent(BaseLLMAgent):
    """Complexity Agent with static analysis + LLM enhancement and detailed tracing"""

    def __init__(self, rag_analyzer=None):
        super().__init__('complexity', rag_analyzer)
        
        # Complexity-specific metadata
        self.agent_metadata.update({
            "specialization": "code_complexity",
            "analysis_methods": ["static_metrics", "llm_enhancement"],
            "complexity_thresholds": {
                "function_length": 50,
                "nesting_depth": 4,
                "parameter_count": 5,
                "line_length": 120
            }
        })

    @traceable(
        name="static_complexity_analysis",
        metadata={"component": "static_analysis", "agent": "complexity"}
    )
    def _run_static_analysis(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Perform comprehensive static complexity analysis with detailed tracing"""
        issues = []
        lines = code.split("\n")
        
        analysis_metadata = {
            "total_lines": len(lines),
            "language": language,
            "analysis_types": []
        }

        # Function length analysis
        function_issues = self._analyze_function_length(lines, language)
        issues.extend(function_issues)
        if function_issues:
            analysis_metadata["analysis_types"].append("function_length")

        # Nesting depth analysis
        nesting_issues = self._analyze_nesting_depth(lines, language)
        issues.extend(nesting_issues)
        if nesting_issues:
            analysis_metadata["analysis_types"].append("nesting_depth")

        # Parameter count analysis
        param_issues = self._analyze_parameter_count(lines, language)
        issues.extend(param_issues)
        if param_issues:
            analysis_metadata["analysis_types"].append("parameter_count")

        # Line length analysis
        line_issues = self._analyze_line_length(lines)
        issues.extend(line_issues)
        if line_issues:
            analysis_metadata["analysis_types"].append("line_length")

        # Add metadata to each issue
        for issue in issues:
            issue["static_analysis"] = True
            issue["analysis_metadata"] = analysis_metadata

        return issues

    @traceable(name="analyze_function_length")
    def _analyze_function_length(self, lines: List[str], language: str) -> List[Dict[str, Any]]:
        """Analyze function length with detailed metrics"""
        issues = []
        function_pattern = r'^\s*def\s+(\w+)\s*\(' if language == 'python' else r'^\s*(function\s+\w+|.*\w+\s*\([^)]*\)\s*{)'
        
        current_func = None
        func_start = 0
        func_name = ""
        
        for i, line in enumerate(lines, start=1):
            match = re.match(function_pattern, line)
            if match:
                # Process previous function
                if current_func:
                    func_length = i - func_start
                    if func_length > 50:
                        issues.append({
                            "severity": "high" if func_length > 100 else "medium",
                            "title": f"Function '{func_name}' too long ({func_length} lines)",
                            "description": f"Function exceeds recommended length of 50 lines. Current length: {func_length} lines.",
                            "line_number": func_start,
                            "category": "complexity",
                            "complexity_type": "function_length",
                            "suggestion": "Consider breaking this function into smaller, more focused functions.",
                            "evidence": f"Function spans {func_length} lines",
                            "metrics": {
                                "actual_length": func_length,
                                "threshold": 50,
                                "severity_ratio": func_length / 50
                            }
                        })
                
                # Start tracking new function
                func_name = match.group(1) if language == 'python' else "function"
                current_func = line.strip()
                func_start = i
        
        # Handle last function
        if current_func:
            func_length = len(lines) - func_start + 1
            if func_length > 50:
                issues.append({
                    "severity": "high" if func_length > 100 else "medium",
                    "title": f"Function '{func_name}' too long ({func_length} lines)",
                    "description": f"Function exceeds recommended length of 50 lines. Current length: {func_length} lines.",
                    "line_number": func_start,
                    "category": "complexity",
                    "complexity_type": "function_length",
                    "suggestion": "Consider breaking this function into smaller, more focused functions.",
                    "evidence": f"Function spans {func_length} lines",
                    "metrics": {
                        "actual_length": func_length,
                        "threshold": 50,
                        "severity_ratio": func_length / 50
                    }
                })
        
        return issues

    @traceable(name="analyze_nesting_depth")
    def _analyze_nesting_depth(self, lines: List[str], language: str) -> List[Dict[str, Any]]:
        """Analyze nesting depth with detailed tracking"""
        issues = []
        max_depth = 0
        current_depth = 0
        deepest_line = 0
        
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            
            # Calculate indentation-based nesting for Python
            if language == 'python':
                indent = len(line) - len(line.lstrip())
                depth = indent // 4  # Assuming 4-space indentation
            else:
                # For brace-based languages, count braces
                depth = current_depth + line.count('{') - line.count('}')
                current_depth = depth
            
            if depth > max_depth:
                max_depth = depth
                deepest_line = i
        
        if max_depth > 4:
            issues.append({
                "severity": "high" if max_depth > 6 else "medium",
                "title": f"Excessive nesting depth ({max_depth} levels)",
                "description": f"Code has {max_depth} levels of nesting, exceeding recommended maximum of 4.",
                "line_number": deepest_line,
                "category": "complexity",
                "complexity_type": "nesting_depth",
                "suggestion": "Consider extracting nested logic into separate functions or using early returns.",
                "evidence": f"Maximum nesting depth: {max_depth}",
                "metrics": {
                    "max_depth": max_depth,
                    "threshold": 4,
                    "deepest_line": deepest_line
                }
            })
        
        return issues

    @traceable(name="analyze_parameter_count")
    def _analyze_parameter_count(self, lines: List[str], language: str) -> List[Dict[str, Any]]:
        """Analyze function parameter counts"""
        issues = []
        
        if language == 'python':
            param_pattern = r'^\s*def\s+(\w+)\s*\((.*?)\)\s*:'
        else:
            param_pattern = r'^\s*(?:function\s+)?(\w+)\s*\((.*?)\)\s*[{:]'
        
        for i, line in enumerate(lines, start=1):
            match = re.match(param_pattern, line)
            if match:
                func_name = match.group(1)
                params_str = match.group(2).strip()
                
                if params_str:
                    # Count parameters (simple comma split)
                    params = [p.strip() for p in params_str.split(',') if p.strip()]
                    param_count = len(params)
                    
                    if param_count > 5:
                        issues.append({
                            "severity": "medium" if param_count <= 7 else "high",
                            "title": f"Function '{func_name}' has too many parameters ({param_count})",
                            "description": f"Function has {param_count} parameters, exceeding recommended maximum of 5.",
                            "line_number": i,
                            "category": "complexity",
                            "complexity_type": "parameter_count",
                            "suggestion": "Consider using a configuration object or breaking the function into smaller parts.",
                            "evidence": f"Parameter count: {param_count}",
                            "metrics": {
                                "parameter_count": param_count,
                                "threshold": 5,
                                "parameters": params[:3]  # Show first 3 params
                            }
                        })
        
        return issues

    @traceable(name="analyze_line_length")
    def _analyze_line_length(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Analyze line length violations"""
        issues = []
        
        for i, line in enumerate(lines, start=1):
            if len(line) > 120:
                issues.append({
                    "severity": "low",
                    "title": f"Line {i} exceeds maximum length ({len(line)} characters)",
                    "description": f"Line exceeds recommended maximum of 120 characters.",
                    "line_number": i,
                    "category": "complexity",
                    "complexity_type": "line_length",
                    "suggestion": "Consider breaking this line into multiple lines for better readability.",
                    "evidence": f"Line length: {len(line)} characters",
                    "metrics": {
                        "line_length": len(line),
                        "threshold": 120,
                        "excess_chars": len(line) - 120
                    }
                })
        
        return issues

    def get_system_prompt(self, language: str) -> str:
        """Enhanced system prompt with static analysis integration"""
        return f"""You are a COMPLEXITY analysis specialist for {language} code with access to static analysis results.

Your role is to enhance and validate static analysis findings with contextual insights.

STATIC ANALYSIS INTEGRATION:
- You will receive pre-computed complexity metrics from static analysis
- Validate these findings against the actual code
- Provide enhanced explanations and specific refactoring suggestions
- Flag any false positives from static analysis

COMPLEXITY FOCUS AREAS:
- Function length and single responsibility principle
- Nesting depth and control flow complexity
- Parameter count and function interfaces
- Line length and readability
- Cyclomatic complexity patterns

ENHANCEMENT GUIDELINES:
- Provide specific, actionable refactoring suggestions
- Explain the maintainability impact of each issue
- Consider the broader codebase context
- Prioritize issues by actual impact on maintainability

RESPONSE FORMAT (valid JSON only):
{{
  "issues": [
    {{
      "severity": "high|medium|low",
      "title": "Enhanced issue title with context",
      "description": "Detailed explanation of complexity impact",
      "line_number": 123,
      "suggestion": "Specific refactoring recommendation",
      "category": "complexity",
      "complexity_type": "function_length|nesting_depth|parameter_count|line_length",
      "evidence": "Relevant code pattern description",
      "confidence_adjustment": 0.1
    }}
  ],
  "metrics": {{"complexity_score": 0.7, "static_analysis_validated": true}},
  "confidence": 0.85
}}

CRITICAL: Only enhance or validate issues, don't create completely new ones unless static analysis missed something obvious."""

    @traceable(
        name="complexity_agent_analyze",
        metadata={"agent_type": "complexity", "component": "main_analysis"}
    )
    async def analyze(self, code: str, file_path: str, language: str) -> Dict[str, Any]:
        """Run static analysis + LLM enhancement with comprehensive tracing"""
        
        # Static analysis first
        static_issues = self._run_static_analysis(code, language)
        
        static_metadata = {
            "static_issues_found": len(static_issues),
            "issue_types": list(set(issue.get("complexity_type", "unknown") for issue in static_issues)),
            "severity_distribution": {}
        }
        
        # Calculate severity distribution
        for issue in static_issues:
            severity = issue.get("severity", "unknown")
            static_metadata["severity_distribution"][severity] = static_metadata["severity_distribution"].get(severity, 0) + 1

        if not static_issues:
            print(f"[COMPLEXITY] No complexity issues found in {file_path}")
            return {
                'agent': 'complexity',
                'language': language,
                'file_path': file_path,
                'issues': [],
                'metrics': {
                    'complexity_score': 0.0,
                    'static_analysis_completed': True,
                    'static_metadata': static_metadata
                },
                'confidence': 0.9,
                'tokens_used': 0,
                'processing_time': 0.01,
                'llm_calls': 0,
                'status': 'completed_clean'
            }

        print(f"[COMPLEXITY] Found {len(static_issues)} static issues, enhancing with LLM...")
        
        # Enhanced prompt with static analysis context
        enhanced_prompt = self._create_enhanced_prompt(code, file_path, language, static_issues)
        
        # Run LLM enhancement
        result = await self._run_llm_enhancement(enhanced_prompt, file_path, language)
        
        # Merge static and LLM results
        final_result = self._merge_static_and_llm_results(static_issues, result, static_metadata, file_path, language)
        
        return final_result

    @traceable(name="create_enhanced_prompt")
    def _create_enhanced_prompt(self, code: str, file_path: str, language: str, 
                               static_issues: List[Dict]) -> str:
        """Create enhanced prompt with static analysis context"""
        
        # Summarize static findings
        static_summary = []
        for issue in static_issues:
            static_summary.append(f"- Line {issue['line_number']}: {issue['title']}")
        
        system_prompt = self.get_system_prompt(language)
        
        # Create numbered code for better line reference
        numbered_lines = []
        for i, line in enumerate(code.split('\n'), 1):
            numbered_lines.append(f"{i:4d}: {line}")
        numbered_code = '\n'.join(numbered_lines)

        enhanced_prompt = f"""
{system_prompt}

STATIC ANALYSIS RESULTS:
The following complexity issues were detected through static analysis:

{chr(10).join(static_summary)}

CODE TO ANALYZE:
File: {file_path}
Language: {language}

Code (with line numbers):
```{language}
{numbered_code}
```

TASK:
1. Validate each static analysis finding
2. Provide enhanced explanations and context
3. Suggest specific refactoring approaches
4. Identify any additional complexity issues missed by static analysis

Focus on providing actionable insights that help developers understand WHY these patterns are problematic and HOW to fix them.
"""
        return enhanced_prompt

    @traceable(name="run_llm_enhancement")
    async def _run_llm_enhancement(self, prompt: str, file_path: str, language: str) -> Dict[str, Any]:
        """Run LLM enhancement with tracing"""
        try:
            response = await self.llm.generate(prompt, max_tokens=1500)
            return self._parse_and_validate_response(response, "", file_path)
        except Exception as e:
            print(f"[COMPLEXITY] LLM enhancement failed: {e}")
            return {"issues": [], "metrics": {"confidence": 0.0, "llm_failed": True}}

    @traceable(name="merge_static_llm_results")
    def _merge_static_and_llm_results(self, static_issues: List[Dict],
                                     llm_result: Dict[str, Any],
                                     static_metadata: Dict[str, Any],
                                     file_path: str,
                                     language: str) -> Dict[str, Any]:
        """Merge static analysis and LLM results"""
        
        # Start with static issues as base
        final_issues = static_issues.copy()
        
        # Enhance with LLM insights
        llm_issues = llm_result.get("issues", [])
        for llm_issue in llm_issues:
            # Try to match with existing static issue
            matched = False
            for static_issue in final_issues:
                if abs(static_issue["line_number"] - llm_issue.get("line_number", 0)) <= 2:
                    # Enhance the static issue with LLM insights
                    static_issue["description"] = llm_issue.get("description", static_issue["description"])
                    static_issue["suggestion"] = llm_issue.get("suggestion", static_issue["suggestion"])
                    static_issue["llm_enhanced"] = True
                    matched = True
                    break
            
            # Add new LLM-only issues
            if not matched and llm_issue.get("line_number", 0) > 0:
                llm_issue["llm_only"] = True
                final_issues.append(llm_issue)

        return {
            'agent': 'complexity',
            'language': language,
            'file_path': file_path,
            'issues': final_issues,
            'metrics': {
                'complexity_score': len(final_issues) * 0.1,
                'static_analysis_completed': True,
                'llm_enhancement_completed': True,
                'static_metadata': static_metadata,
                'confidence': llm_result.get("confidence", 0.8)
            },
            'confidence': llm_result.get("confidence", 0.8),
            'tokens_used': len(str(final_issues)) // 4,
            'processing_time': 0.5,
            'llm_calls': 1,
            'status': 'completed_enhanced'
        }