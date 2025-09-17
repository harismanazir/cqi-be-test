"""
LangGraph State Management for Multi-Agent Code Analysis
Defines the shared state and agent dependencies for the workflow
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass
import operator

# Agent dependency configuration - duplication agent removed
AGENT_DEPENDENCIES = {
    'security': [],                              # Priority 1 - No dependencies
    'complexity': [],                            # Priority 2 - No dependencies  
    'performance': ['complexity'],               # Depends on complexity insights
    'documentation': ['complexity'],             # Depends on complexity analysis
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
    # Add LangSmith tracing fields
    trace_id: Optional[str] = None
    run_id: Optional[str] = None
    langsmith_url: Optional[str] = None

@dataclass 
class CodeAnalysisInput:
    """Input data for code analysis"""
    file_path: str
    code_content: str
    language: str
    file_size: int
    selected_agents: List[str]
    enable_rag: bool = True
    # Add LangSmith tracing
    trace_metadata: Optional[Dict[str, Any]] = None

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
    
    # Cross-agent insights (agents can share findings) - removed duplication_insights
    security_insights: Annotated[List[str], operator.add]
    complexity_insights: Annotated[List[str], operator.add]
    performance_insights: Annotated[List[str], operator.add]
    documentation_insights: Annotated[List[str], operator.add]
    
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
    
    # LangSmith tracing
    langsmith_session_id: Optional[str]
    langsmith_project: str
    trace_metadata: Dict[str, Any]

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
        # Removed duplication agent
    }
    return priority_map.get(agent_name, 10)

def initialize_workflow_state(analysis_input: CodeAnalysisInput) -> WorkflowState:
    """Initialize the workflow state with LangSmith tracing"""
    import uuid
    import time
    
    # Sort agents by priority and dependencies (excluding duplication)
    valid_agents = ['security', 'complexity', 'performance', 'documentation']
    selected_agents = [agent for agent in analysis_input.selected_agents if agent in valid_agents]
    selected_agents = sorted(selected_agents, key=calculate_agent_priority)
    
    # Generate session ID for LangSmith tracing
    session_id = str(uuid.uuid4())
    
    return WorkflowState(
        analysis_input=analysis_input,
        completed_agents=[],
        failed_agents=[],
        pending_agents=selected_agents,
        agent_results={},
        security_insights=[],
        complexity_insights=[],
        performance_insights=[],
        documentation_insights=[],
        workflow_status='initializing',
        total_processing_time=0.0,
        total_tokens=0,
        total_llm_calls=0,
        all_issues=[],
        issues_by_severity={'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
        agent_performance={},
        rag_chunks=None,
        rag_enabled=analysis_input.enable_rag,
        langsmith_session_id=session_id,
        langsmith_project="code-analysis-workflow",
        trace_metadata={
            'workflow_version': '2.0',
            'file_path': analysis_input.file_path,
            'language': analysis_input.language,
            'agents_count': len(selected_agents),
            'started_at': time.time(),
            'session_id': session_id
        }
    )

def add_agent_insights(state: WorkflowState, agent_name: str, insights: List[str]) -> None:
    """Add insights from an agent to be shared with other agents"""
    if agent_name == 'security':
        state['security_insights'].extend(insights)
    elif agent_name == 'complexity':
        state['complexity_insights'].extend(insights)
    elif agent_name == 'performance':
        state['performance_insights'].extend(insights)
    elif agent_name == 'documentation':
        state['documentation_insights'].extend(insights)

def get_cross_agent_context(state: WorkflowState, requesting_agent: str) -> str:
    """Get contextual insights from other agents for the requesting agent"""
    context_parts = []
    
    # Security context for other agents
    if requesting_agent != 'security' and state['security_insights']:
        context_parts.append(f"Security Analysis Insights:\n" + 
                           "\n".join(f"- {insight}" for insight in state['security_insights']))
    
    # Complexity context for dependent agents
    if (requesting_agent in ['performance', 'documentation'] 
        and state['complexity_insights']):
        context_parts.append(f"Complexity Analysis Insights:\n" + 
                           "\n".join(f"- {insight}" for insight in state['complexity_insights']))
    
    # Performance context for documentation agent
    if requesting_agent == 'documentation' and state['performance_insights']:
        context_parts.append(f"Performance Analysis Insights:\n" + 
                           "\n".join(f"- {insight}" for insight in state['performance_insights']))
    
    return "\n\n".join(context_parts) if context_parts else ""

def create_langsmith_metadata(agent_name: str, state: WorkflowState) -> Dict[str, Any]:
    """Create LangSmith metadata for agent execution"""
    return {
        'agent_name': agent_name,
        'session_id': state['langsmith_session_id'],
        'file_path': state['analysis_input'].file_path,
        'language': state['analysis_input'].language,
        'completed_agents': state['completed_agents'],
        'workflow_status': state['workflow_status'],
        'total_tokens_so_far': state['total_tokens'],
        'cross_agent_context_available': bool(get_cross_agent_context(state, agent_name))
    }