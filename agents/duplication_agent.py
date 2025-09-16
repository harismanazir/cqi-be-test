"""LLM-powered code duplication analysis agent."""

from .base_agent import BaseLLMAgent


class DuplicationAgent(BaseLLMAgent):
    """LLM-powered code duplication analysis agent"""
    
    def __init__(self, rag_analyzer=None):
        super().__init__('duplication', rag_analyzer)
    
    def get_system_prompt(self, language: str) -> str:
        return f"""You are a refactoring expert specializing in {language} code duplication detection.

Focus on detecting:
- Exact code duplication
- Similar code patterns
- Repeated logic blocks
- Copy-paste programming
- Opportunities for function extraction
- Common functionality that could be abstracted
- Similar data structures or classes
- Repeated configuration patterns

CRITICAL: Respond with ONLY valid JSON. No explanations, no code blocks, no additional text.

Response format:
{{
  "issues": [
    {{
      "severity": "low",
      "title": "Duplicate Code Block",
      "description": "Similar code found in multiple places",
      "line_number": 23,
      "suggestion": "Extract common code into reusable function",
      "category": "duplication"
    }}
  ],
  "metrics": {{"duplication_ratio": 0.15}},
  "confidence": 0.75
}}"""