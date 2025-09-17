"""
LangGraph Workflow Package
Intelligent multi-agent orchestration for code analysis
Updated to remove duplication agent and add LangSmith tracing
"""

from .state import (
    WorkflowState, 
    AgentResult, 
    CodeAnalysisInput, 
    AGENT_DEPENDENCIES,
    get_ready_agents,
    should_continue_workflow,
    add_agent_insights,
    get_cross_agent_context,
    initialize_workflow_state,
    create_langsmith_metadata
)
from .graph import LangGraphMultiAgentAnalyzer

# Version info
__version__ = "2.0.0"
__author__ = "Code Quality Insights"

# Configuration
SUPPORTED_AGENTS = ['security', 'complexity', 'performance', 'documentation']
REMOVED_AGENTS = ['duplication']  # Agents that have been removed

__all__ = [
    'WorkflowState',
    'AgentResult', 
    'CodeAnalysisInput',
    'AGENT_DEPENDENCIES',
    'LangGraphMultiAgentAnalyzer',
    'get_ready_agents',
    'should_continue_workflow',
    'add_agent_insights',
    'get_cross_agent_context',
    'initialize_workflow_state',
    'create_langsmith_metadata',
    'SUPPORTED_AGENTS',
    'REMOVED_AGENTS'
]

# Check for LangSmith availability
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
    print("[WORKFLOW] LangSmith tracing available")
except ImportError:
    LANGSMITH_AVAILABLE = False
    print("[WORKFLOW] LangSmith tracing not available")

def get_workflow_info():
    """Get information about the workflow package"""
    return {
        'version': __version__,
        'supported_agents': SUPPORTED_AGENTS,
        'removed_agents': REMOVED_AGENTS,
        'langsmith_available': LANGSMITH_AVAILABLE,
        'features': [
            'Dependency-aware agent execution',
            'LangSmith tracing integration',
            'Intelligent caching',
            'RAG-enhanced analysis',
            'Cross-agent insights sharing'
        ]
    }