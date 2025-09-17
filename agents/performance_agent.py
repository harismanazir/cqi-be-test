"""Enhanced Performance Agent with comprehensive LangSmith tracing."""

from .base_agent import BaseLLMAgent, traceable
import re
from typing import Dict, List, Any

class PerformanceAgent(BaseLLMAgent):
    """Enhanced Performance Agent with detailed tracing and pattern analysis"""
    
    def __init__(self, rag_analyzer=None):
        super().__init__('performance', rag_analyzer)
        self.performance_patterns = self._init_performance_patterns()
        
        # Performance-specific metadata
        self.agent_metadata.update({
            "specialization": "performance_optimization",
            "pattern_categories": list(self.performance_patterns.keys()),
            "analysis_methods": ["pattern_matching", "algorithmic_analysis", "llm_enhancement"]
        })
    
    @traceable(
        name="init_performance_patterns",
        metadata={"component": "pattern_initialization", "agent": "performance"}
    )
    def _init_performance_patterns(self):
        """Initialize performance anti-patterns with detailed tracing"""
        patterns = {
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
            ],
            'nested_loops': [
                r'for.*:.*for.*:',
                r'while.*:.*for.*:',
                r'for.*:.*while.*:',
            ],
            'io_in_loops': [
                r'for.*:.*open\s*\(',
                r'while.*:.*file\.',
                r'for.*:.*print\s*\(',
            ]
        }
        return patterns
    
    @traceable(
        name="pre_validate_performance",
        metadata={"component": "static_analysis", "agent": "performance"}
    )
    def _pre_validate_performance_issues(self, code: str) -> Dict[str, Any]:
        """Enhanced pre-validation with detailed pattern analysis"""
        found_patterns = {}
        total_matches = 0
        severity_scores = {}
        
        for category, patterns in self.performance_patterns.items():
            category_matches = []
            category_severity = 0
            
            for pattern in patterns:
                matches = list(re.finditer(pattern, code, re.IGNORECASE))
                if matches:
                    for match in matches:
                        line_num = code[:match.start()].count('\n') + 1
                        match_info = {
                            'pattern': pattern,
                            'line': line_num,
                            'match': match.group(),
                            'start_pos': match.start(),
                            'end_pos': match.end()
                        }
                        category_matches.append(match_info)
                        
                        # Calculate severity based on pattern type
                        if category in ['nested_loops', 'database_issues']:
                            category_severity += 3  # High impact
                        elif category in ['inefficient_loops', 'string_concatenation']:
                            category_severity += 2  # Medium impact
                        else:
                            category_severity += 1  # Low impact
            
            if category_matches:
                found_patterns[category] = category_matches
                severity_scores[category] = category_severity
                total_matches += len(category_matches)
        
        # Calculate overall performance risk score
        risk_score = sum(severity_scores.values()) / max(1, len(code.split('\n'))) * 100
        
        validation_result = {
            'has_issues': total_matches > 0,
            'pattern_matches': found_patterns,
            'total_matches': total_matches,
            'categories_affected': list(found_patterns.keys()),
            'severity_scores': severity_scores,
            'risk_score': risk_score,
            'high_risk': risk_score > 10
        }
        
        return validation_result
    
    def get_system_prompt(self, language: str) -> str:
        return f"""You are a PERFORMANCE OPTIMIZATION specialist for {language} code.

CRITICAL ACCURACY REQUIREMENTS:
- ONLY report performance issues that actually exist and can be measured
- Verify each issue by examining the specific code patterns
- If no performance problems exist, return empty issues array []
- Focus on issues that would have measurable impact on execution time or memory

PERFORMANCE SCOPE - Analyze these specific issues:
- Inefficient algorithms (O(n²) when O(n) exists, nested loops)
- Inefficient data structure usage (wrong data type for operations)
- Memory management issues (leaks, unnecessary allocations)
- String concatenation in loops (especially in Python)
- Database query inefficiencies (N+1 queries, missing optimizations)
- Unnecessary repeated calculations in loops
- I/O operations in tight loops
- Inefficient iteration patterns (range(len()) instead of enumerate)

VALIDATION REQUIREMENTS:
- For algorithm inefficiencies: Show actual nested loops or suboptimal patterns
- For data structures: Show clear misuse with performance implications
- For database issues: Show actual query patterns that cause performance problems
- Each issue must include specific performance impact explanation

RESPONSE FORMAT - Valid JSON only:
{{
  "issues": [
    {{
      "severity": "high|medium|low",
      "title": "Specific performance issue",
      "description": "Explanation of performance impact with time/space complexity",
      "line_number": 123,
      "suggestion": "Specific optimization with expected improvement",
      "category": "performance",
      "performance_type": "algorithm|data_structure|memory|io|database",
      "complexity_impact": "O(n) vs O(n²)",
      "evidence": "Brief description of the inefficient pattern"
    }}
  ],
  "metrics": {{"performance_score": 0.7, "risk_assessment": "medium"}},
  "confidence": 0.85
}}

REMEMBER: Focus on real bottlenecks that would impact production performance."""
    
    @traceable(
        name="performance_agent_analyze",
        metadata={"agent_type": "performance", "component": "main_analysis"}
    )
    async def analyze(self, code: str, file_path: str, language: str) -> Dict[str, Any]:
        """Enhanced performance analysis with comprehensive tracing"""
        
        # Pre-validation with detailed pattern analysis
        validation_result = self._pre_validate_performance_issues(code)
        
        # Create detailed metadata for tracing
        validation_metadata = {
            "static_analysis_completed": True,
            "patterns_found": validation_result['total_matches'],
            "performance_categories": validation_result['categories_affected'],
            "risk_score": validation_result['risk_score'],
            "high_risk_code": validation_result['high_risk'],
            "requires_llm_analysis": validation_result['has_issues']
        }
        
        if not validation_result['has_issues']:
            print(f"[PERFORMANCE] No performance anti-patterns detected in {file_path}")
            
            result = {
                'agent': 'performance',
                'language': language,
                'file_path': file_path,
                'issues': [],
                'metrics': {
                    'performance_score': 1.0, 
                    'confidence': 0.9,
                    'risk_assessment': 'low',
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
        
        # Run enhanced LLM analysis with pattern context
        print(f"[PERFORMANCE] Running enhanced analysis - found {validation_result['total_matches']} potential issues")
        print(f"[PERFORMANCE] Risk score: {validation_result['risk_score']:.1f}")
        
        # Create enhanced prompt with pattern context
        enhanced_prompt = self._create_performance_prompt(code, file_path, language, validation_result)
        
        # Run LLM analysis
        result = await self._run_performance_analysis(enhanced_prompt, file_path, language)
        
        # Enhance result with static analysis findings
        result['metrics']['static_analysis'] = validation_result
        result['validation_metadata'] = validation_metadata
        
        # Add performance-specific insights to issues
        for issue in result.get('issues', []):
            issue['pattern_analysis'] = self._analyze_issue_patterns(
                issue, validation_result['pattern_matches']
            )
            issue['performance_impact'] = self._estimate_performance_impact(issue)
        
        return result
    
    @traceable(name="create_performance_prompt")
    def _create_performance_prompt(self, code: str, file_path: str, language: str, 
                                  validation_result: Dict[str, Any]) -> str:
        """Create enhanced prompt with performance pattern context"""
        
        # Summarize found patterns
        pattern_summary = []
        for category, matches in validation_result['pattern_matches'].items():
            pattern_summary.append(f"- {category}: {len(matches)} occurrences")
        
        system_prompt = self.get_system_prompt(language)
        
        enhanced_prompt = f"""
{system_prompt}

STATIC ANALYSIS RESULTS:
Risk Score: {validation_result['risk_score']:.1f}/100
Patterns Found:
{chr(10).join(pattern_summary)}

CODE TO ANALYZE:
File: {file_path}
Language: {language}

```{language}
{code}
```

PERFORMANCE CONTEXT:
- Focus on the identified pattern categories: {', '.join(validation_result['categories_affected'])}
- Provide specific complexity analysis (Big O notation where applicable)
- Suggest concrete optimization strategies with expected performance gains
- Consider the impact on both time and space complexity

Analyze the code for performance bottlenecks and provide actionable optimization recommendations.
"""
        return enhanced_prompt
    
    @traceable(name="run_performance_analysis")
    async def _run_performance_analysis(self, prompt: str, file_path: str, language: str) -> Dict[str, Any]:
        """Run performance analysis with enhanced context"""
        try:
            response = await self.llm.generate(prompt, max_tokens=1500)
            result = self._parse_and_validate_response(response, "", file_path)
            
            # Add performance-specific metadata
            result['analysis_type'] = 'performance_optimization'
            result['status'] = 'completed_enhanced'
            
            return result
            
        except Exception as e:
            print(f"[PERFORMANCE] LLM analysis failed: {e}")
            return {
                "issues": [],
                "metrics": {"confidence": 0.0, "llm_failed": True, "error": str(e)},
                "confidence": 0.0
            }
    
    @traceable(name="analyze_issue_patterns")
    def _analyze_issue_patterns(self, issue: Dict[str, Any], 
                               pattern_matches: Dict[str, List]) -> Dict[str, Any]:
        """Analyze which patterns match this issue"""
        issue_line = issue.get('line_number', 0)
        matching_patterns = []
        
        for category, matches in pattern_matches.items():
            for match in matches:
                if abs(match['line'] - issue_line) <= 3:  # Within 3 lines
                    matching_patterns.append({
                        'category': category,
                        'pattern': match['pattern'],
                        'line_distance': abs(match['line'] - issue_line)
                    })
        
        return {
            'matching_patterns': matching_patterns,
            'pattern_confirmed': len(matching_patterns) > 0,
            'confidence_boost': 0.2 if matching_patterns else 0.0
        }
    
    @traceable(name="estimate_performance_impact")
    def _estimate_performance_impact(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate the performance impact of an issue"""
        severity = issue.get('severity', 'low')
        performance_type = issue.get('performance_type', 'unknown')
        
        impact_map = {
            'algorithm': {'high': 'Major', 'medium': 'Moderate', 'low': 'Minor'},
            'data_structure': {'high': 'Significant', 'medium': 'Moderate', 'low': 'Minor'},
            'database': {'high': 'Critical', 'medium': 'Significant', 'low': 'Moderate'},
            'memory': {'high': 'Significant', 'medium': 'Moderate', 'low': 'Minor'},
            'io': {'high': 'Major', 'medium': 'Moderate', 'low': 'Minor'}
        }
        
        impact_level = impact_map.get(performance_type, {}).get(severity, 'Unknown')
        
        return {
            'impact_level': impact_level,
            'optimization_priority': severity,
            'expected_improvement': self._get_expected_improvement(performance_type, severity)
        }
    
    def _get_expected_improvement(self, performance_type: str, severity: str) -> str:
        """Get expected improvement description"""
        improvements = {
            ('algorithm', 'high'): '10x-100x faster execution',
            ('algorithm', 'medium'): '2x-10x faster execution',
            ('database', 'high'): '50-90% reduction in query time',
            ('memory', 'high'): '50-80% memory usage reduction',
            ('io', 'medium'): '20-50% I/O operation reduction'
        }
        
        return improvements.get((performance_type, severity), 'Measurable improvement expected')