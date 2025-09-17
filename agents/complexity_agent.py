"""Enhanced Complexity Agent with comprehensive multi-language support and LangSmith tracing."""

import re
from .base_agent import BaseLLMAgent, traceable
from typing import Dict, List, Any, Tuple, Optional

class ComplexityAgent(BaseLLMAgent):
    """Complexity Agent with static analysis + LLM enhancement and detailed tracing for multiple languages"""

    def __init__(self, rag_analyzer=None):
        super().__init__('complexity', rag_analyzer)
        
        # Language-specific patterns and configurations
        self.language_patterns = self._initialize_language_patterns()
        
        # Complexity-specific metadata
        self.agent_metadata.update({
            "specialization": "code_complexity",
            "analysis_methods": ["static_metrics", "llm_enhancement"],
            "supported_languages": list(self.language_patterns.keys()),
            "complexity_thresholds": {
                "function_length": 50,
                "nesting_depth": 4,
                "parameter_count": 5,
                "line_length": 120,
                "cyclomatic_complexity": 10
            }
        })

    def _initialize_language_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize language-specific patterns and configurations"""
        return {
            'python': {
                'function_patterns': [
                    r'^\s*def\s+(\w+)\s*\(',
                    r'^\s*async\s+def\s+(\w+)\s*\(',
                    r'^\s*class\s+(\w+).*:.*def\s+(\w+)\s*\('  # class methods
                ],
                'class_patterns': [
                    r'^\s*class\s+(\w+)(?:\([^)]*\))?:'
                ],
                'control_flow': [
                    r'^\s*(if|elif|else|for|while|try|except|finally|with)\s*',
                    r'^\s*(match|case)\s+',  # Python 3.10+
                ],
                'indent_size': 4,
                'brace_based': False,
                'comment_patterns': [r'#.*$'],
                'string_patterns': [r'""".*?"""', r"'''.*?'''", r'"[^"]*"', r"'[^']*'"],
                'lambda_patterns': [r'lambda\s+[^:]+:']
            },
            'javascript': {
                'function_patterns': [
                    r'^\s*function\s+(\w+)\s*\(',
                    r'^\s*(\w+)\s*:\s*function\s*\(',
                    r'^\s*(\w+)\s*=\s*function\s*\(',
                    r'^\s*(\w+)\s*=\s*\([^)]*\)\s*=>\s*{',
                    r'^\s*async\s+function\s+(\w+)\s*\(',
                    r'^\s*(\w+)\s*=\s*async\s*\([^)]*\)\s*=>\s*{'
                ],
                'class_patterns': [
                    r'^\s*class\s+(\w+)(?:\s+extends\s+\w+)?'
                ],
                'control_flow': [
                    r'^\s*(if|else|for|while|do|switch|case|try|catch|finally)\s*',
                    r'.*\?\s*.*:\s*.*'  # ternary operator
                ],
                'indent_size': 2,
                'brace_based': True,
                'comment_patterns': [r'//.*$', r'/\*.*?\*/'],
                'string_patterns': [r'`[^`]*`', r'"[^"]*"', r"'[^']*'"],
                'lambda_patterns': [r'\([^)]*\)\s*=>\s*', r'\w+\s*=>\s*']
            },
            'typescript': {
                'function_patterns': [
                    r'^\s*function\s+(\w+)\s*\(',
                    r'^\s*(\w+)\s*:\s*\([^)]*\)\s*=>\s*\w+\s*=',
                    r'^\s*(\w+)\s*=\s*\([^)]*\)\s*:\s*\w+\s*=>\s*{',
                    r'^\s*async\s+function\s+(\w+)\s*\(',
                    r'^\s*(public|private|protected)?\s*(\w+)\s*\('
                ],
                'class_patterns': [
                    r'^\s*(export\s+)?(abstract\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?',
                    r'^\s*interface\s+(\w+)'
                ],
                'control_flow': [
                    r'^\s*(if|else|for|while|do|switch|case|try|catch|finally)\s*',
                    r'.*\?\s*.*:\s*.*'  # ternary operator
                ],
                'indent_size': 2,
                'brace_based': True,
                'comment_patterns': [r'//.*$', r'/\*.*?\*/'],
                'string_patterns': [r'`[^`]*`', r'"[^"]*"', r"'[^']*'"],
                'lambda_patterns': [r'\([^)]*\)\s*=>\s*', r'\w+\s*=>\s*']
            },
            'java': {
                'function_patterns': [
                    r'^\s*(public|private|protected)?\s*(static\s+)?(final\s+)?[\w<>,\[\]\s]+\s+(\w+)\s*\(',
                    r'^\s*(public|private|protected)?\s*(\w+)\s*\(',  # constructors
                ],
                'class_patterns': [
                    r'^\s*(public\s+)?(abstract\s+|final\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?',
                    r'^\s*(public\s+)?interface\s+(\w+)',
                    r'^\s*(public\s+)?enum\s+(\w+)'
                ],
                'control_flow': [
                    r'^\s*(if|else|for|while|do|switch|case|try|catch|finally)\s*',
                    r'.*\?\s*.*:\s*.*'  # ternary operator
                ],
                'indent_size': 4,
                'brace_based': True,
                'comment_patterns': [r'//.*$', r'/\*.*?\*/', r'/\*\*.*?\*/'],
                'string_patterns': [r'"[^"]*"'],
                'lambda_patterns': [r'\([^)]*\)\s*->\s*', r'\w+\s*->\s*']
            },
            'csharp': {
                'function_patterns': [
                    r'^\s*(public|private|protected|internal)?\s*(static\s+)?(virtual\s+|override\s+|abstract\s+)?(async\s+)?[\w<>,\[\]\s]+\s+(\w+)\s*\(',
                    r'^\s*(public|private|protected|internal)?\s*(\w+)\s*\(',  # constructors
                ],
                'class_patterns': [
                    r'^\s*(public\s+)?(abstract\s+|sealed\s+|static\s+)?class\s+(\w+)(?:\s*:\s*[\w,\s]+)?',
                    r'^\s*(public\s+)?interface\s+(\w+)',
                    r'^\s*(public\s+)?enum\s+(\w+)',
                    r'^\s*(public\s+)?struct\s+(\w+)'
                ],
                'control_flow': [
                    r'^\s*(if|else|for|foreach|while|do|switch|case|try|catch|finally|using)\s*',
                    r'.*\?\s*.*:\s*.*'  # ternary operator
                ],
                'indent_size': 4,
                'brace_based': True,
                'comment_patterns': [r'//.*$', r'/\*.*?\*/', r'///.*$'],
                'string_patterns': [r'@?"[^"]*"', r"'[^']*'"],
                'lambda_patterns': [r'\([^)]*\)\s*=>\s*', r'\w+\s*=>\s*']
            },
            'cpp': {
                'function_patterns': [
                    r'^\s*(?:(?:inline|static|virtual|const|explicit)\s+)*[\w:&*<>,\s]+\s+(\w+)\s*\(',
                    r'^\s*(\w+)::\s*(\w+)\s*\(',  # class method definitions
                    r'^\s*template\s*<[^>]*>\s*[\w:&*<>,\s]+\s+(\w+)\s*\('
                ],
                'class_patterns': [
                    r'^\s*class\s+(\w+)(?:\s*:\s*[^{]*)?',
                    r'^\s*struct\s+(\w+)(?:\s*:\s*[^{]*)?',
                    r'^\s*template\s*<[^>]*>\s*class\s+(\w+)',
                    r'^\s*namespace\s+(\w+)'
                ],
                'control_flow': [
                    r'^\s*(if|else|for|while|do|switch|case|try|catch)\s*',
                    r'.*\?\s*.*:\s*.*'  # ternary operator
                ],
                'indent_size': 2,
                'brace_based': True,
                'comment_patterns': [r'//.*$', r'/\*.*?\*/'],
                'string_patterns': [r'"[^"]*"', r"'[^']*'", r'R"\([^)]*\)"'],
                'lambda_patterns': [r'\[[^\]]*\]\s*\([^)]*\)\s*(?:->\s*[\w&*]+)?\s*{']
            },
            'go': {
                'function_patterns': [
                    r'^\s*func\s+(\w+)\s*\(',
                    r'^\s*func\s+\(\s*\w+\s+\*?\w+\s*\)\s+(\w+)\s*\(',  # method receivers
                ],
                'class_patterns': [
                    r'^\s*type\s+(\w+)\s+struct',
                    r'^\s*type\s+(\w+)\s+interface'
                ],
                'control_flow': [
                    r'^\s*(if|else|for|switch|case|select)\s*',
                    r'^\s*go\s+',
                    r'^\s*defer\s+'
                ],
                'indent_size': 1,  # Go uses tabs
                'brace_based': True,
                'comment_patterns': [r'//.*$', r'/\*.*?\*/'],
                'string_patterns': [r'`[^`]*`', r'"[^"]*"'],
                'lambda_patterns': [r'func\s*\([^)]*\)\s*[\w\s]*{']
            },
            'rust': {
                'function_patterns': [
                    r'^\s*(pub\s+)?(async\s+)?(unsafe\s+)?fn\s+(\w+)\s*[<\(]',
                    r'^\s*impl(?:\s*<[^>]*>)?\s+(?:\w+\s+for\s+)?(\w+)',
                ],
                'class_patterns': [
                    r'^\s*(pub\s+)?struct\s+(\w+)',
                    r'^\s*(pub\s+)?enum\s+(\w+)',
                    r'^\s*(pub\s+)?trait\s+(\w+)',
                    r'^\s*(pub\s+)?mod\s+(\w+)'
                ],
                'control_flow': [
                    r'^\s*(if|else|match|for|while|loop)\s*',
                    r'^\s*\w+\s*=>\s*'  # match arms
                ],
                'indent_size': 4,
                'brace_based': True,
                'comment_patterns': [r'//.*$', r'/\*.*?\*/', r'///.*$', r'//!.*$'],
                'string_patterns': [r'r#"[^"]*"#', r'"[^"]*"', r"'[^']*'"],
                'lambda_patterns': [r'\|[^|]*\|\s*', r'move\s*\|[^|]*\|\s*']
            },
            'php': {
                'function_patterns': [
                    r'^\s*(public|private|protected)?\s*(static\s+)?function\s+(\w+)\s*\(',
                    r'^\s*function\s+(\w+)\s*\('
                ],
                'class_patterns': [
                    r'^\s*(abstract\s+|final\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?',
                    r'^\s*interface\s+(\w+)',
                    r'^\s*trait\s+(\w+)'
                ],
                'control_flow': [
                    r'^\s*(if|elseif|else|for|foreach|while|do|switch|case|try|catch|finally)\s*',
                    r'.*\?\s*.*:\s*.*'  # ternary operator
                ],
                'indent_size': 4,
                'brace_based': True,
                'comment_patterns': [r'//.*$', r'/\*.*?\*/', r'#.*$'],
                'string_patterns': [r'"[^"]*"', r"'[^']*'"],
                'lambda_patterns': [r'function\s*\([^)]*\)\s*use\s*\([^)]*\)', r'fn\s*\([^)]*\)\s*=>']
            }
        }

    @traceable(
        name="static_complexity_analysis",
        metadata={"component": "static_analysis", "agent": "complexity"}
    )
    def _run_static_analysis(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Perform comprehensive static complexity analysis with detailed tracing"""
        if language not in self.language_patterns:
            print(f"[COMPLEXITY] Unsupported language: {language}")
            return []
        
        issues = []
        lines = code.split("\n")
        lang_config = self.language_patterns[language]
        
        analysis_metadata = {
            "total_lines": len(lines),
            "language": language,
            "analysis_types": [],
            "lang_config": {
                "indent_size": lang_config["indent_size"],
                "brace_based": lang_config["brace_based"]
            }
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

        # Cyclomatic complexity analysis
        complexity_issues = self._analyze_cyclomatic_complexity(lines, language)
        issues.extend(complexity_issues)
        if complexity_issues:
            analysis_metadata["analysis_types"].append("cyclomatic_complexity")

        # Add metadata to each issue
        for issue in issues:
            issue["static_analysis"] = True
            issue["analysis_metadata"] = analysis_metadata

        return issues

    @traceable(name="analyze_function_length")
    def _analyze_function_length(self, lines: List[str], language: str) -> List[Dict[str, Any]]:
        """Analyze function length with language-specific patterns"""
        issues = []
        lang_config = self.language_patterns[language]
        
        functions = self._extract_functions(lines, language)
        
        for func_info in functions:
            func_length = func_info['end_line'] - func_info['start_line'] + 1
            if func_length > self.agent_metadata["complexity_thresholds"]["function_length"]:
                severity = "high" if func_length > 100 else "medium"
                issues.append({
                    "severity": severity,
                    "title": f"Function '{func_info['name']}' too long ({func_length} lines)",
                    "description": f"Function exceeds recommended length of {self.agent_metadata['complexity_thresholds']['function_length']} lines. Current length: {func_length} lines.",
                    "line_number": func_info['start_line'],
                    "category": "complexity",
                    "complexity_type": "function_length",
                    "suggestion": "Consider breaking this function into smaller, more focused functions using the Single Responsibility Principle.",
                    "evidence": f"Function spans {func_length} lines",
                    "metrics": {
                        "actual_length": func_length,
                        "threshold": self.agent_metadata["complexity_thresholds"]["function_length"],
                        "severity_ratio": func_length / self.agent_metadata["complexity_thresholds"]["function_length"],
                        "function_type": func_info.get('type', 'function')
                    }
                })
        
        return issues

    def _extract_functions(self, lines: List[str], language: str) -> List[Dict[str, Any]]:
        """Extract function information using language-specific patterns"""
        functions = []
        lang_config = self.language_patterns[language]
        
        current_func = None
        brace_count = 0
        indent_stack = []
        
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or self._is_comment_line(stripped, language):
                continue
            
            # Check for function start
            for pattern in lang_config['function_patterns']:
                match = re.search(pattern, line)
                if match:
                    func_name = self._extract_function_name(match, pattern)
                    if func_name:
                        # Close previous function if needed
                        if current_func:
                            current_func['end_line'] = i - 1
                            functions.append(current_func)
                        
                        current_func = {
                            'name': func_name,
                            'start_line': i,
                            'end_line': len(lines),  # Will be updated
                            'type': 'function'
                        }
                        
                        if lang_config['brace_based']:
                            brace_count = line.count('{') - line.count('}')
                        else:
                            indent_stack = [len(line) - len(line.lstrip())]
                        break
            
            # Track function end
            if current_func:
                if lang_config['brace_based']:
                    brace_count += line.count('{') - line.count('}')
                    if brace_count <= 0:
                        current_func['end_line'] = i
                        functions.append(current_func)
                        current_func = None
                        brace_count = 0
                else:
                    # For indentation-based languages
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= indent_stack[0] and stripped and not line.startswith(' '):
                        if not any(re.match(pattern, line) for pattern in lang_config['function_patterns']):
                            current_func['end_line'] = i - 1
                            functions.append(current_func)
                            current_func = None
                            indent_stack = []
        
        # Handle last function
        if current_func:
            current_func['end_line'] = len(lines)
            functions.append(current_func)
        
        return functions

    def _extract_function_name(self, match: re.Match, pattern: str) -> Optional[str]:
        """Extract function name from regex match"""
        # Try different group indices based on pattern
        for i in range(1, min(len(match.groups()) + 1, 5)):
            try:
                name = match.group(i)
                if name and name.isidentifier():
                    return name
            except (IndexError, AttributeError):
                continue
        return None

    def _is_comment_line(self, line: str, language: str) -> bool:
        """Check if line is a comment using language-specific patterns"""
        lang_config = self.language_patterns[language]
        for comment_pattern in lang_config['comment_patterns']:
            if re.match(comment_pattern, line.strip()):
                return True
        return False

    @traceable(name="analyze_nesting_depth")
    def _analyze_nesting_depth(self, lines: List[str], language: str) -> List[Dict[str, Any]]:
        """Analyze nesting depth with language-specific patterns"""
        issues = []
        lang_config = self.language_patterns[language]
        max_depth = 0
        current_depth = 0
        deepest_line = 0
        depth_stack = []
        
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or self._is_comment_line(stripped, language):
                continue
            
            if lang_config['brace_based']:
                # Brace-based nesting
                open_braces = line.count('{')
                close_braces = line.count('}')
                
                # Check for control flow structures
                is_control_flow = any(re.match(pattern, stripped) for pattern in lang_config['control_flow'])
                
                if is_control_flow or open_braces > 0:
                    current_depth += open_braces
                    if is_control_flow and open_braces == 0:
                        current_depth += 1  # Implicit nesting for control structures
                
                current_depth -= close_braces
                current_depth = max(0, current_depth)
            else:
                # Indentation-based nesting (Python, etc.)
                indent = len(line) - len(line.lstrip())
                depth = indent // lang_config['indent_size']
                
                # Check for control flow structures
                is_control_flow = any(re.match(pattern, stripped) for pattern in lang_config['control_flow'])
                if is_control_flow:
                    depth += 1
                
                current_depth = depth
            
            if current_depth > max_depth:
                max_depth = current_depth
                deepest_line = i
        
        threshold = self.agent_metadata["complexity_thresholds"]["nesting_depth"]
        if max_depth > threshold:
            severity = "high" if max_depth > threshold + 2 else "medium"
            issues.append({
                "severity": severity,
                "title": f"Excessive nesting depth ({max_depth} levels)",
                "description": f"Code has {max_depth} levels of nesting, exceeding recommended maximum of {threshold}.",
                "line_number": deepest_line,
                "category": "complexity",
                "complexity_type": "nesting_depth",
                "suggestion": "Consider extracting nested logic into separate functions, using early returns, or guard clauses to reduce nesting.",
                "evidence": f"Maximum nesting depth: {max_depth}",
                "metrics": {
                    "max_depth": max_depth,
                    "threshold": threshold,
                    "deepest_line": deepest_line,
                    "language_type": "brace_based" if lang_config['brace_based'] else "indent_based"
                }
            })
        
        return issues

    @traceable(name="analyze_parameter_count")
    def _analyze_parameter_count(self, lines: List[str], language: str) -> List[Dict[str, Any]]:
        """Analyze function parameter counts with language-specific patterns"""
        issues = []
        lang_config = self.language_patterns[language]
        threshold = self.agent_metadata["complexity_thresholds"]["parameter_count"]
        
        for i, line in enumerate(lines, start=1):
            for pattern in lang_config['function_patterns']:
                match = re.search(pattern, line)
                if match:
                    func_name = self._extract_function_name(match, pattern)
                    if not func_name:
                        continue
                    
                    # Extract parameter section
                    param_match = re.search(r'\(([^)]*)\)', line)
                    if param_match:
                        params_str = param_match.group(1).strip()
                        
                        if params_str:
                            # Language-specific parameter parsing
                            params = self._parse_parameters(params_str, language)
                            param_count = len(params)
                            
                            if param_count > threshold:
                                severity = "medium" if param_count <= threshold + 2 else "high"
                                issues.append({
                                    "severity": severity,
                                    "title": f"Function '{func_name}' has too many parameters ({param_count})",
                                    "description": f"Function has {param_count} parameters, exceeding recommended maximum of {threshold}.",
                                    "line_number": i,
                                    "category": "complexity",
                                    "complexity_type": "parameter_count",
                                    "suggestion": "Consider using a parameter object, builder pattern, or breaking the function into smaller parts.",
                                    "evidence": f"Parameter count: {param_count}",
                                    "metrics": {
                                        "parameter_count": param_count,
                                        "threshold": threshold,
                                        "parameters": params[:3],  # Show first 3 params
                                        "excess_count": param_count - threshold
                                    }
                                })
                    break
        
        return issues

    def _parse_parameters(self, params_str: str, language: str) -> List[str]:
        """Parse function parameters based on language syntax"""
        if not params_str.strip():
            return []
        
        # Remove generic type parameters for languages that support them
        if language in ['java', 'csharp', 'typescript', 'cpp']:
            # Simple generic removal (may not handle nested generics perfectly)
            params_str = re.sub(r'<[^<>]*>', '', params_str)
        
        # Split by comma, but be careful with nested structures
        params = []
        current_param = ""
        paren_depth = 0
        bracket_depth = 0
        
        for char in params_str:
            if char in '([{':
                paren_depth += 1
            elif char in ')]}':
                paren_depth -= 1
            elif char == ',' and paren_depth == 0:
                if current_param.strip():
                    params.append(current_param.strip())
                current_param = ""
                continue
            
            current_param += char
        
        if current_param.strip():
            params.append(current_param.strip())
        
        # Filter out empty parameters and self/this parameters
        filtered_params = []
        for param in params:
            param = param.strip()
            if param and not param.startswith('self') and not param.startswith('this'):
                # Extract just the parameter name (remove type annotations)
                if language == 'python':
                    param_name = param.split(':')[0].split('=')[0].strip()
                elif language in ['typescript', 'java', 'csharp']:
                    # For typed languages, parameter name is usually last
                    parts = param.split()
                    param_name = parts[-1] if parts else param
                else:
                    param_name = param.split('=')[0].strip()  # Remove default values
                
                if param_name:
                    filtered_params.append(param_name)
        
        return filtered_params

    @traceable(name="analyze_cyclomatic_complexity")
    def _analyze_cyclomatic_complexity(self, lines: List[str], language: str) -> List[Dict[str, Any]]:
        """Analyze cyclomatic complexity with language-specific patterns"""
        issues = []
        lang_config = self.language_patterns[language]
        threshold = self.agent_metadata["complexity_thresholds"]["cyclomatic_complexity"]
        
        functions = self._extract_functions(lines, language)
        
        for func_info in functions:
            complexity = self._calculate_cyclomatic_complexity(
                lines[func_info['start_line']-1:func_info['end_line']], 
                language
            )
            
            if complexity > threshold:
                severity = "high" if complexity > threshold + 5 else "medium"
                issues.append({
                    "severity": severity,
                    "title": f"Function '{func_info['name']}' has high cyclomatic complexity ({complexity})",
                    "description": f"Function has cyclomatic complexity of {complexity}, exceeding recommended maximum of {threshold}.",
                    "line_number": func_info['start_line'],
                    "category": "complexity",
                    "complexity_type": "cyclomatic_complexity",
                    "suggestion": "Consider breaking this function into smaller functions, reducing conditional logic, or using polymorphism.",
                    "evidence": f"Cyclomatic complexity: {complexity}",
                    "metrics": {
                        "complexity": complexity,
                        "threshold": threshold,
                        "excess_complexity": complexity - threshold,
                        "function_length": func_info['end_line'] - func_info['start_line'] + 1
                    }
                })
        
        return issues

    def _calculate_cyclomatic_complexity(self, function_lines: List[str], language: str) -> int:
        """Calculate cyclomatic complexity for a function"""
        lang_config = self.language_patterns[language]
        complexity = 1  # Base complexity
        
        # Complexity-increasing constructs
        complexity_patterns = {
            'if_else': [r'\b(if|elif)\b'],
            'loops': [r'\b(for|while|do)\b'],
            'switch_case': [r'\b(switch|case)\b'],
            'try_catch': [r'\b(catch|except)\b'],
            'ternary': [r'\?.*:'],
            'logical_ops': [r'(\|\||&&|\band\b|\bor\b)'],
        }
        
        # Language-specific patterns
        if language == 'python':
            complexity_patterns['comprehensions'] = [r'\b(for\s+\w+\s+in\s+.*if\b)']
            complexity_patterns['with_stmt'] = [r'\bwith\b']
        elif language == 'go':
            complexity_patterns['select'] = [r'\bselect\b']
            complexity_patterns['go_stmt'] = [r'\bgo\b']
        elif language == 'rust':
            complexity_patterns['match_arms'] = [r'\w+\s*=>\s*']
        elif language in ['java', 'csharp']:
            complexity_patterns['foreach'] = [r'\bforeach\b']
        
        for line in function_lines:
            stripped = line.strip()
            if not stripped or self._is_comment_line(stripped, language):
                continue
            
            for pattern_type, patterns in complexity_patterns.items():
                for pattern in patterns:
                    matches = len(re.findall(pattern, line, re.IGNORECASE))
                    if pattern_type == 'logical_ops':
                        complexity += matches  # Each logical operator adds 1
                    elif matches > 0:
                        complexity += 1  # Each construct adds 1
                        break  # Don't double-count same line
        
        return complexity

    @traceable(name="analyze_line_length")
    def _analyze_line_length(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Analyze line length violations"""
        issues = []
        threshold = self.agent_metadata["complexity_thresholds"]["line_length"]
        
        for i, line in enumerate(lines, start=1):
            # Skip comment-only lines for line length (they're often documentation)
            stripped = line.strip()
            if len(line) > threshold and not (stripped.startswith('#') or stripped.startswith('//')):
                severity = "low" if len(line) < threshold + 20 else "medium"
                issues.append({
                    "severity": severity,
                    "title": f"Line {i} exceeds maximum length ({len(line)} characters)",
                    "description": f"Line exceeds recommended maximum of {threshold} characters.",
                    "line_number": i,
                    "category": "complexity",
                    "complexity_type": "line_length",
                    "suggestion": "Consider breaking this line into multiple lines for better readability.",
                    "evidence": f"Line length: {len(line)} characters",
                    "metrics": {
                        "line_length": len(line),
                        "threshold": threshold,
                        "excess_chars": len(line) - threshold,
                        "line_preview": line[:50] + "..." if len(line) > 50 else line
                    }
                })
        
        return issues

    def get_system_prompt(self, language: str) -> str:
        """Enhanced system prompt with language-specific static analysis integration"""
        lang_features = ""
        if language in self.language_patterns:
            config = self.language_patterns[language]
            lang_features = f"""
LANGUAGE-SPECIFIC ANALYSIS FOR {language.upper()}:
- Indentation: {"Tab/brace-based" if config['brace_based'] else f"{config['indent_size']} spaces"}
- Function patterns: {len(config['function_patterns'])} patterns detected
- Control flow constructs: {len(config['control_flow'])} patterns analyzed
- Comment styles: {config['comment_patterns']}
"""

        return f"""You are a COMPLEXITY analysis specialist for {language} code with access to comprehensive static analysis results.

Your role is to enhance and validate static analysis findings with contextual insights.

{lang_features}

STATIC ANALYSIS INTEGRATION:
- You will receive pre-computed complexity metrics from multi-language static analysis
- Validate these findings against the actual code context
- Provide enhanced explanations and specific refactoring suggestions
- Flag any false positives from static analysis
- Focus only on confirmed complexity issues

COMPLEXITY FOCUS AREAS:
1. Function length and single responsibility principle
2. Nesting depth and control flow complexity  
3. Parameter count and function interfaces
4. Line length and readability
5. Cyclomatic complexity and decision points

VALIDATION CRITERIA:
- Confirm each static finding represents a real maintainability issue
- Consider language idioms and best practices
- Evaluate impact on code readability and maintenance
- Prioritize issues by actual development impact

ENHANCEMENT GUIDELINES:
- Provide specific, actionable refactoring suggestions
- Explain the maintainability impact of each issue
- Consider the broader codebase context and patterns
- Suggest language-specific solutions and patterns
- Include code examples where helpful

RESPONSE FORMAT (valid JSON only):
{{
  "issues": [
    {{
      "severity": "high|medium|low",
      "title": "Enhanced issue title with specific context",
      "description": "Detailed explanation of complexity impact and why it matters",
      "line_number": 123,
      "suggestion": "Specific refactoring recommendation with {language} best practices",
      "category": "complexity",
      "complexity_type": "function_length|nesting_depth|parameter_count|line_length|cyclomatic_complexity",
      "evidence": "Relevant code pattern description",
      "confidence_adjustment": 0.1,
      "refactoring_priority": "high|medium|low"
    }}
  ],
  "metrics": {{
    "complexity_score": 0.7, 
    "static_analysis_validated": true,
    "language_specific_issues": true
  }},
  "confidence": 0.85
}}

CRITICAL REQUIREMENTS:
- Only validate and enhance confirmed complexity issues
- Don't create new issues unless static analysis missed obvious problems  
- Focus on maintainability impact, not style preferences
- Provide concrete refactoring strategies
- Consider {language}-specific patterns and idioms"""

    @traceable(
        name="complexity_agent_analyze",
        metadata={"agent_type": "complexity", "component": "main_analysis"}
    )
    async def analyze(self, code: str, file_path: str, language: str) -> Dict[str, Any]:
        """Run static analysis + LLM enhancement with comprehensive tracing"""
        
        # Validate language support
        if language not in self.language_patterns:
            return {
                'agent': 'complexity',
                'language': language,
                'file_path': file_path,
                'issues': [],
                'metrics': {
                    'complexity_score': 0.0,
                    'unsupported_language': True,
                    'supported_languages': list(self.language_patterns.keys())
                },
                'confidence': 0.0,
                'tokens_used': 0,
                'processing_time': 0.01,
                'llm_calls': 0,
                'status': 'unsupported_language'
            }
        
        # Static analysis first
        static_issues = self._run_static_analysis(code, language)
        
        static_metadata = {
            "static_issues_found": len(static_issues),
            "issue_types": list(set(issue.get("complexity_type", "unknown") for issue in static_issues)),
            "severity_distribution": {},
            "language_config": self.language_patterns[language]['brace_based']
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

        print(f"[COMPLEXITY] Found {len(static_issues)} static issues for {language}, enhancing with LLM...")
        
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
        
        # Summarize static findings by type
        issue_summary = {}
        for issue in static_issues:
            issue_type = issue.get('complexity_type', 'unknown')
            if issue_type not in issue_summary:
                issue_summary[issue_type] = []
            issue_summary[issue_type].append(f"Line {issue['line_number']}: {issue['title']}")
        
        static_summary = []
        for issue_type, issues in issue_summary.items():
            static_summary.append(f"\n{issue_type.upper()} ISSUES:")
            static_summary.extend([f"  {issue}" for issue in issues])
        
        system_prompt = self.get_system_prompt(language)
        
        # Create numbered code for better line reference
        numbered_lines = []
        code_lines = code.split('\n')
        for i, line in enumerate(code_lines, 1):
            numbered_lines.append(f"{i:4d}: {line}")
        numbered_code = '\n'.join(numbered_lines)
        total_lines = len(code_lines)

        enhanced_prompt = f"""
{system_prompt}

STATIC ANALYSIS RESULTS ({len(static_issues)} issues found):
{''.join(static_summary)}

CODE TO ANALYZE:
File: {file_path}
Language: {language}
Total Lines: {total_lines}

Code (with line numbers):
```{language}
{numbered_code}
```

VALIDATION TASK:
1. Review each static analysis finding for accuracy and relevance
2. Provide enhanced explanations with {language}-specific context
3. Suggest concrete refactoring approaches using {language} best practices
4. Assess the maintainability impact of each issue
5. Flag any false positives or low-priority issues

Focus on providing actionable insights that help developers understand WHY these complexity patterns are problematic in {language} and HOW to fix them effectively.
"""
        return enhanced_prompt

    @traceable(name="run_llm_enhancement")
    async def _run_llm_enhancement(self, prompt: str, file_path: str, language: str) -> Dict[str, Any]:
        """Run LLM enhancement with tracing"""
        try:
            response = await self.llm.generate(prompt, max_tokens=2000)
            return self._parse_and_validate_response(response, "", file_path)
        except Exception as e:
            print(f"[COMPLEXITY] LLM enhancement failed: {e}")
            return {
                "issues": [], 
                "metrics": {
                    "confidence": 0.0, 
                    "llm_failed": True,
                    "error": str(e)
                }
            }

    @traceable(name="merge_static_llm_results")
    def _merge_static_and_llm_results(self, static_issues: List[Dict],
                                     llm_result: Dict[str, Any],
                                     static_metadata: Dict[str, Any],
                                     file_path: str,
                                     language: str) -> Dict[str, Any]:
        """Merge static analysis and LLM results with enhanced validation"""
        
        # Start with validated static issues
        final_issues = []
        llm_issues = llm_result.get("issues", [])
        
        # Create a mapping of LLM issues by line number for matching
        llm_by_line = {}
        for llm_issue in llm_issues:
            line_num = llm_issue.get("line_number", 0)
            if line_num > 0:
                llm_by_line[line_num] = llm_issue
        
        # Enhance static issues with LLM insights
        for static_issue in static_issues:
            static_line = static_issue["line_number"]
            enhanced_issue = static_issue.copy()
            
            # Try to find matching LLM enhancement (within 2 lines)
            llm_match = None
            for line_offset in range(-2, 3):  # Check lines within ±2
                check_line = static_line + line_offset
                if check_line in llm_by_line:
                    llm_match = llm_by_line[check_line]
                    break
            
            if llm_match:
                # Enhance with LLM insights
                enhanced_issue.update({
                    "description": llm_match.get("description", static_issue["description"]),
                    "suggestion": llm_match.get("suggestion", static_issue["suggestion"]),
                    "llm_enhanced": True,
                    "refactoring_priority": llm_match.get("refactoring_priority", "medium"),
                    "confidence_adjustment": llm_match.get("confidence_adjustment", 0.0)
                })
                
                # Remove from llm_by_line to avoid double-processing
                if check_line in llm_by_line:
                    del llm_by_line[check_line]
            
            final_issues.append(enhanced_issue)
        
        # Add any remaining LLM-only issues (that weren't matched)
        for remaining_llm_issue in llm_by_line.values():
            if remaining_llm_issue.get("line_number", 0) > 0:
                remaining_llm_issue["llm_only"] = True
                remaining_llm_issue["static_analysis"] = False
                final_issues.append(remaining_llm_issue)

        # Calculate comprehensive metrics
        complexity_score = min(1.0, len(final_issues) * 0.15)  # Adjust scoring
        high_severity_count = sum(1 for issue in final_issues if issue.get("severity") == "high")
        
        return {
            'agent': 'complexity',
            'language': language,
            'file_path': file_path,
            'issues': final_issues,
            'metrics': {
                'complexity_score': complexity_score,
                'static_analysis_completed': True,
                'llm_enhancement_completed': True,
                'static_metadata': static_metadata,
                'confidence': llm_result.get("confidence", 0.8),
                'high_severity_issues': high_severity_count,
                'total_issues': len(final_issues),
                'enhancement_ratio': len([i for i in final_issues if i.get("llm_enhanced")]) / max(len(final_issues), 1)
            },
            'confidence': llm_result.get("confidence", 0.8),
            'tokens_used': len(str(final_issues)) // 4,  # Rough estimate
            'processing_time': 0.8,
            'llm_calls': 1,
            'status': 'completed_enhanced'
        }