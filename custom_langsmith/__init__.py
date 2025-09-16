"""
LangSmith Integration Package
Optional prompt management and enhancement for code analysis agents.

Safe fallback design - if LangSmith fails, everything works with existing prompts.
"""

from .prompt_manager import PromptManager, get_enhanced_prompt, get_langsmith_status
from .hub_client import LangSmithHubClient, create_hub_client

# Version info
__version__ = "1.0.0"
__author__ = "Code Quality Insights"

# Main exports
__all__ = [
    'PromptManager',
    'get_enhanced_prompt',
    'get_langsmith_status', 
    'LangSmithHubClient',
    'create_hub_client'
]

# Check if LangSmith is available
try:
    import sys
    # Temporarily remove current directory from path to avoid circular import
    current_dir = sys.path[0] if sys.path[0] else ''
    if current_dir:
        sys.path.remove(current_dir)

    from langsmith import Client
    LANGSMITH_AVAILABLE = True
    print("[LANGSMITH] LangSmith package loaded - enhanced prompts available")

    # Restore path
    if current_dir:
        sys.path.insert(0, current_dir)
except ImportError:
    LANGSMITH_AVAILABLE = False
    print("[LANGSMITH] LangSmith not installed - using fallback prompts")

    # Restore path in case of error
    if 'current_dir' in locals() and current_dir and current_dir not in sys.path:
        sys.path.insert(0, current_dir)

# Export availability status
__all__.append('LANGSMITH_AVAILABLE')