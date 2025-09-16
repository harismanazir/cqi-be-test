"""
LangGraph State Management for Multi-Agent Code Analysis
Defines the shared state and agent dependencies for the workflow
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass
import operator

# Agent dependency configuration as mentioned in README
AGENT_DEPENDENCIES = {
    'security': [],                              # Priority 1 - No dependencies
    'complexity': [],                            # Priority 2 - No dependencies  
    'performance': ['complexity'],               # Depends on complexity insights
# Remove testing agent dependencies for now
    'documentation': ['complexity'],             # Depends on complexity analysis
    'duplication': ['complexity']                # Depends on complexity analysis
}

@dataclass
class AgentResult:
    """Individual agent analysis result"""
    agent_name: str
    issues: List[Dict[str, Any]]
    processing_time: float
    tokens_used: int
    confidence: float
    status: str  # 'completed', 'failed', 'skipped'
    error_message: Optional[str] = None
    llm_calls: int = 0  # Track actual LLM API calls

@dataclass 
class CodeAnalysisInput:
    """Input data for code analysis"""
    file_path: str
    code_content: str
    language: str
    file_size: int
    selected_agents: List[str]
    enable_rag: bool = True

class WorkflowState(TypedDict):
    """LangGraph workflow state - shared across all agents"""
    
    # Input data
    analysis_input: CodeAnalysisInput
    
    # Agent execution tracking
    completed_agents: Annotated[List[str], operator.add]
    failed_agents: Annotated[List[str], operator.add]
    pending_agents: List[str]
    
    # Agent results
    agent_results: Annotated[Dict[str, AgentResult], operator.or_]
    
    # Cross-agent insights (agents can share findings)
    security_insights: Annotated[List[str], operator.add]
    complexity_insights: Annotated[List[str], operator.add]
    performance_insights: Annotated[List[str], operator.add]
    
    # Workflow control
    workflow_status: str  # 'initializing', 'analyzing', 'aggregating', 'completed', 'failed'
    total_processing_time: float
    total_tokens: int
    total_llm_calls: int
    
    # Final aggregated results
    all_issues: List[Dict[str, Any]]
    issues_by_severity: Dict[str, int]
    agent_performance: Dict[str, Dict[str, Any]]
    
    # RAG context (if enabled)
    rag_chunks: Optional[List[Dict[str, Any]]]
    rag_enabled: bool

def get_ready_agents(state: WorkflowState) -> List[str]:
    """
    Determine which agents are ready to run based on dependencies
    Returns agents whose dependencies have been completed
    """
    completed = set(state['completed_agents'])
    pending = state['pending_agents']
    ready = []
    
    for agent in pending:
        dependencies = AGENT_DEPENDENCIES.get(agent, [])
        if all(dep in completed for dep in dependencies):
            ready.append(agent)
    
    return ready

def should_continue_workflow(state: WorkflowState) -> bool:
    """Check if workflow should continue or terminate"""
    return (
        state['workflow_status'] not in ['completed', 'failed'] and
        len(state['pending_agents']) > 0
    )

def calculate_agent_priority(agent_name: str) -> int:
    """Calculate agent execution priority (lower = higher priority)"""
    priority_map = {
        'security': 1,      # Highest priority
        'complexity': 2,    # Second priority
        'performance': 3,   # Third priority (depends on complexity)
        'documentation': 4, # Fourth priority (depends on complexity)
        'duplication': 5    # Fifth priority (depends on complexity)
    }
    return priority_map.get(agent_name, 10)

def initialize_workflow_state(analysis_input: CodeAnalysisInput) -> WorkflowState:
    """Initialize the workflow state"""
    
    # Sort agents by priority and dependencies
    selected_agents = sorted(
        analysis_input.selected_agents, 
        key=calculate_agent_priority
    )
    
    return WorkflowState(
        analysis_input=analysis_input,
        completed_agents=[],
        failed_agents=[],
        pending_agents=selected_agents,
        agent_results={},
        security_insights=[],
        complexity_insights=[],
        performance_insights=[],
        workflow_status='initializing',
        total_processing_time=0.0,
        total_tokens=0,
        total_llm_calls=0,
        all_issues=[],
        issues_by_severity={'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
        agent_performance={},
        rag_chunks=None,
        rag_enabled=analysis_input.enable_rag
    )

def add_agent_insights(state: WorkflowState, agent_name: str, insights: List[str]) -> None:
    """Add insights from an agent to be shared with other agents"""
    if agent_name == 'security':
        state['security_insights'].extend(insights)
    elif agent_name == 'complexity':
        state['complexity_insights'].extend(insights)
    elif agent_name == 'performance':
        state['performance_insights'].extend(insights)

def get_cross_agent_context(state: WorkflowState, requesting_agent: str) -> str:
    """Get contextual insights from other agents for the requesting agent"""
    context_parts = []
    
    # Security context for other agents
    if requesting_agent != 'security' and state['security_insights']:
        context_parts.append(f"Security Analysis Insights:\n" + 
                           "\n".join(f"- {insight}" for insight in state['security_insights']))
    
    # Complexity context for dependent agents
    if (requesting_agent in ['performance', 'documentation', 'duplication'] 
        and state['complexity_insights']):
        context_parts.append(f"Complexity Analysis Insights:\n" + 
                           "\n".join(f"- {insight}" for insight in state['complexity_insights']))
    
# Remove testing agent context for now
    
    return "\n\n".join(context_parts) if context_parts else ""