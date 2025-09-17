"""
Fixed LangGraph Multi-Agent Analyzer with Proper Graph Visualization in LangSmith
"""

import os
import time
import asyncio
import uuid
from typing import Dict, List, Any, Optional
from pathlib import Path

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.pregel import Pregel
except ImportError:
    raise ImportError("Install LangGraph: pip install langgraph")

# LangSmith tracing
LANGSMITH_AVAILABLE = False
try:
    from langsmith import traceable, trace
    from langsmith.client import Client as LangSmithClient
    from langsmith.run_helpers import get_current_run_tree
    LANGSMITH_AVAILABLE = True
except ImportError:
    def traceable(name: str = None, **kwargs):
        def decorator(func):
            return func
        return decorator
    def trace(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Workflow components
from workflow.state import (
    WorkflowState, AgentResult, CodeAnalysisInput,
    get_ready_agents, should_continue_workflow,
    add_agent_insights, get_cross_agent_context,
    initialize_workflow_state, create_langsmith_metadata
)

# Agents
from agents import SecurityAgent, PerformanceAgent, ComplexityAgent, DocumentationAgent


class LangGraphMultiAgentAnalyzer:
    """Multi-agent analyzer with proper LangSmith graph visualization"""
    
    def __init__(self, enable_rag: bool = True, enable_cache: bool = True, 
                 langsmith_project: str = "code-analysis-workflow"):
        self.enable_rag = enable_rag
        self.enable_cache = enable_cache
        self.langsmith_project = langsmith_project
        
        # Initialize LangSmith
        self.langsmith_client = None
        if LANGSMITH_AVAILABLE:
            try:
                self.langsmith_client = LangSmithClient()
                print(f"[WORKFLOW] LangSmith Project: {langsmith_project}")
                print(f"[WORKFLOW] Dashboard: https://smith.langchain.com/projects/{langsmith_project}")
            except Exception as e:
                print(f"[WORKFLOW] LangSmith setup failed: {e}")
        
        # Initialize other components
        self._setup_components()
        
        # Build workflow with proper graph structure
        self.workflow = self._build_workflow()
        self._print_workflow_info()
    
    def _setup_components(self):
        """Setup cache, RAG, and agents"""
        # Cache setup
        self.cache = None
        self.cache_enabled = False
        if self.enable_cache:
            try:
                from agents.intelligent_cache import get_cache
                self.cache = get_cache()
                self.cache_enabled = True
            except ImportError:
                pass
        
        # RAG setup
        self.rag_analyzer = None
        if self.enable_rag:
            try:
                from agents.rag_agent import RAGCodeAnalyzer
                self.rag_analyzer = RAGCodeAnalyzer()
            except ImportError:
                pass
        
        # Agents setup
        self.agents = {
            'security': SecurityAgent(rag_analyzer=self.rag_analyzer),
            'performance': PerformanceAgent(rag_analyzer=self.rag_analyzer),
            'complexity': ComplexityAgent(rag_analyzer=self.rag_analyzer),
            'documentation': DocumentationAgent(rag_analyzer=self.rag_analyzer)
        }
    
    def _build_workflow(self) -> Pregel:
        """Build workflow with proper structure for LangSmith visualization"""
        
        # Create state graph with proper naming
        workflow = StateGraph(WorkflowState)
        
        # Add nodes with clear, simple names (important for visualization)
        workflow.add_node("initialize", self._initialize)
        workflow.add_node("setup_rag", self._setup_rag)
        workflow.add_node("route_agents", self._route_agents)
        workflow.add_node("security_agent", self._run_security_agent)
        workflow.add_node("complexity_agent", self._run_complexity_agent)
        workflow.add_node("performance_agent", self._run_performance_agent)
        workflow.add_node("documentation_agent", self._run_documentation_agent)
        workflow.add_node("aggregate_results", self._aggregate_results)
        workflow.add_node("finalize", self._finalize)
        
        # Set entry point
        workflow.set_entry_point("initialize")
        
        # Linear flow for initialization
        workflow.add_edge("initialize", "setup_rag")
        workflow.add_edge("setup_rag", "route_agents")
        
        # Conditional routing to agents
        workflow.add_conditional_edges(
            "route_agents",
            self._determine_next_agent,
            {
                "security": "security_agent",
                "complexity": "complexity_agent",
                "performance": "performance_agent", 
                "documentation": "documentation_agent",
                "aggregate": "aggregate_results"
            }
        )
        
        # Agent completion routing back to router or aggregation
        for agent_node in ["security_agent", "complexity_agent", "performance_agent", "documentation_agent"]:
            workflow.add_conditional_edges(
                agent_node,
                self._agent_completion_router,
                {
                    "continue": "route_agents",
                    "aggregate": "aggregate_results"
                }
            )
        
        # Final edges
        workflow.add_edge("aggregate_results", "finalize")
        workflow.add_edge("finalize", END)
        
        # Compile with checkpointer for proper state tracking
        compiled_workflow = workflow.compile(
            checkpointer=MemorySaver() if MemorySaver else None
        )
        
        return compiled_workflow
    
    def _print_workflow_info(self):
        """Print workflow structure information"""
        print(f"[WORKFLOW] Workflow initialized")
        print(f"[WORKFLOW] Nodes: initialize -> setup_rag -> route_agents -> [agents] -> aggregate_results -> finalize")
        print(f"[WORKFLOW] Available agents: {', '.join(self.agents.keys())}")
        
        # Print the workflow graph structure for debugging
        if hasattr(self.workflow, 'get_graph'):
            try:
                graph_structure = self.workflow.get_graph()
                print(f"[WORKFLOW] Graph nodes: {list(graph_structure.nodes.keys())}")
                print(f"[WORKFLOW] Graph edges: {len(graph_structure.edges)} connections")
            except Exception as e:
                print(f"[WORKFLOW] Could not retrieve graph structure: {e}")
    
    # ============ WORKFLOW NODES ============
    
    async def _initialize(self, state: WorkflowState) -> Dict[str, Any]:
        """Initialize the analysis workflow"""
        file_info = state['analysis_input']
        print(f"[INIT] Starting analysis: {Path(file_info.file_path).name}")
        print(f"[INIT] Language: {file_info.language}")
        print(f"[INIT] Selected agents: {', '.join(file_info.selected_agents)}")
        
        return {
            "workflow_status": "analyzing",
            "total_processing_time": time.time()
        }
    
    async def _setup_rag(self, state: WorkflowState) -> Dict[str, Any]:
        """Setup RAG context if enabled"""
        if state['rag_enabled'] and self.rag_analyzer:
            print("[RAG] Setting up context...")
            try:
                from agents.base_agent import LanguageDetector
                self.rag_analyzer.index_codebase(
                    [state['analysis_input'].file_path], 
                    LanguageDetector()
                )
                print("[RAG] Context ready")
            except Exception as e:
                print(f"[RAG] Setup failed: {e}")
        else:
            print("[RAG] Disabled, skipping...")
        return {}
    
    async def _route_agents(self, state: WorkflowState) -> Dict[str, Any]:
        """Route to the next available agent"""
        completed = state['completed_agents']
        pending = state['pending_agents']
        ready = get_ready_agents(state)
        
        print(f"[ROUTER]  Routing decision")
        print(f"[ROUTER]  Completed: {completed}")
        print(f"[ROUTER]  Pending: {pending}")
        print(f"[ROUTER]  Ready: {ready}")
        
        return {}
    
    async def _run_security_agent(self, state: WorkflowState) -> Dict[str, Any]:
        """Run security analysis"""
        return await self._execute_agent('security', '[SEC]', state)
    
    async def _run_complexity_agent(self, state: WorkflowState) -> Dict[str, Any]:
        """Run complexity analysis"""
        return await self._execute_agent('complexity', '[COMP]', state)
    
    async def _run_performance_agent(self, state: WorkflowState) -> Dict[str, Any]:
        """Run performance analysis"""
        return await self._execute_agent('performance', '[PERF]', state)
    
    async def _run_documentation_agent(self, state: WorkflowState) -> Dict[str, Any]:
        """Run documentation analysis"""
        return await self._execute_agent('documentation', '[DOC]', state)
    
    async def _execute_agent(self, agent_name: str, emoji: str, state: WorkflowState) -> Dict[str, Any]:
        """Execute individual agent with proper tracing"""
        print(f"[{agent_name.upper()}] {emoji} Starting analysis...")
        
        # Rate limiting
        if len(state['completed_agents']) > 0:
            await asyncio.sleep(5)
        
        start_time = time.time()
        agent = self.agents[agent_name]
        analysis_input = state['analysis_input']
        
        # Execute agent with retry logic
        for attempt in range(3):
            try:
                result = await agent.analyze(
                    analysis_input.code_content,
                    analysis_input.file_path,
                    analysis_input.language
                )
                break
            except Exception as e:
                if "rate_limit" in str(e) and attempt < 2:
                    wait_time = 20 * (2 ** attempt)
                    print(f"[{agent_name.upper()}]  Rate limit, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                raise e
        
        processing_time = time.time() - start_time
        
        # Create agent result
        agent_result = AgentResult(
            agent_name=agent_name,
            issues=result.get('issues', []),
            processing_time=processing_time,
            tokens_used=result.get('tokens_used', 0),
            confidence=result.get('confidence', 0.8),
            status='completed',
            llm_calls=result.get('llm_calls', 0),
            trace_id=str(uuid.uuid4()) if LANGSMITH_AVAILABLE else None
        )
        
        # Share insights with other agents
        insights = self._extract_insights(agent_result)
        add_agent_insights(state, agent_name, insights)
        
        print(f"[{agent_name.upper()}]  Completed: {len(agent_result.issues)} issues in {processing_time:.2f}s")
        
        return {
            'completed_agents': [agent_name],
            'agent_results': {agent_name: agent_result},
            'pending_agents': [a for a in state['pending_agents'] if a != agent_name],
            'total_tokens': state['total_tokens'] + agent_result.tokens_used,
            'total_llm_calls': state['total_llm_calls'] + agent_result.llm_calls
        }
    
    async def _aggregate_results(self, state: WorkflowState) -> Dict[str, Any]:
        """Aggregate results from all completed agents"""
        print("[AGGREGATE]  Combining results...")
        
        all_issues = []
        issues_by_severity = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        agent_performance = {}
        
        for agent_name, result in state['agent_results'].items():
            # Add issues
            for issue in result.issues:
                issue['agent'] = agent_name
                issue['file_path'] = state['analysis_input'].file_path
                all_issues.append(issue)
                
                severity = issue.get('severity', 'low')
                if severity in issues_by_severity:
                    issues_by_severity[severity] += 1
            
            # Track performance
            agent_performance[agent_name] = {
                'issues_found': len(result.issues),
                'processing_time': result.processing_time,
                'tokens_used': result.tokens_used,
                'confidence': result.confidence,
                'status': result.status
            }
        
        print(f"[AGGREGATE] [COMP] Total issues: {len(all_issues)}")
        print(f"[AGGREGATE]  Severity breakdown: {issues_by_severity}")
        
        return {
            'all_issues': all_issues,
            'issues_by_severity': issues_by_severity,
            'agent_performance': agent_performance
        }
    
    async def _finalize(self, state: WorkflowState) -> Dict[str, Any]:
        """Finalize the analysis"""
        total_time = time.time() - state['total_processing_time']
        
        print(f"[FINALIZE]  Analysis complete!")
        print(f"[FINALIZE]  Total time: {total_time:.2f}s")
        print(f"[FINALIZE]  Total issues: {len(state['all_issues'])}")
        print(f"[FINALIZE]  Agents completed: {len(state['completed_agents'])}")
        print(f"[FINALIZE]  Total tokens: {state['total_tokens']}")
        
        if LANGSMITH_AVAILABLE and state.get('langsmith_session_id'):
            session_url = f"https://smith.langchain.com/projects/{self.langsmith_project}/sessions/{state['langsmith_session_id']}"
            print(f"[FINALIZE]  View in LangSmith: {session_url}")
        
        return {
            'workflow_status': 'completed',
            'total_processing_time': total_time
        }
    
    # ============ ROUTING FUNCTIONS ============
    
    def _determine_next_agent(self, state: WorkflowState) -> str:
        """Determine which agent should run next"""
        ready_agents = get_ready_agents(state)
        
        if not ready_agents:
            print("[ROUTER]  No more agents ready -> aggregating")
            return "aggregate"
        
        # Priority order
        priority_order = ['security', 'complexity', 'performance', 'documentation']
        next_agent = min(ready_agents, key=lambda x: priority_order.index(x) if x in priority_order else 999)
        
        print(f"[ROUTER]  Next agent: {next_agent}")
        return next_agent
    
    def _agent_completion_router(self, state: WorkflowState) -> str:
        """Route after agent completion"""
        if should_continue_workflow(state):
            print("[ROUTER]  More agents to run -> continue")
            return "continue"
        else:
            print("[ROUTER]  All agents complete -> aggregate")
            return "aggregate"
    
    def _extract_insights(self, agent_result: AgentResult) -> List[str]:
        """Extract key insights to share with other agents"""
        insights = []
        for issue in agent_result.issues:
            if issue.get('severity') in ['critical', 'high']:
                title = issue.get('title', 'Unknown issue')
                suggestion = issue.get('suggestion', 'No suggestion')
                insights.append(f"{title} - {suggestion}")
        return insights[:3]
    
    # ============ PUBLIC API ============
    
    async def analyze_file(self, file_path: str, code_content: str, language: str,
                          selected_agents: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze a single file with proper LangSmith tracing"""
        
        # Filter valid agents
        valid_agents = ['security', 'complexity', 'performance', 'documentation']
        if selected_agents:
            selected_agents = [a for a in selected_agents if a in valid_agents]
        else:
            selected_agents = valid_agents
        
        # Check cache first
        if self.cache_enabled and self.cache:
            cached_result = self.cache.get_analysis_result(file_path, selected_agents)
            if cached_result:
                print(f"[CACHE]  Cache hit: {Path(file_path).name}")
                cached_result.update({
                    'workflow_engine': 'langgraph',
                    'cache_hit': True,
                    'processing_time': 0.01
                })
                return cached_result
        
        # Prepare analysis input
        analysis_input = CodeAnalysisInput(
            file_path=file_path,
            code_content=code_content,
            language=language,
            file_size=len(code_content),
            selected_agents=selected_agents,
            enable_rag=self.enable_rag
        )
        
        # Initialize workflow state
        initial_state = initialize_workflow_state(analysis_input)
        session_id = initial_state['langsmith_session_id']
        
        print(f"\n[WORKFLOW]  Starting LangGraph Analysis")
        print(f"[WORKFLOW]  File: {Path(file_path).name}")
        print(f"[WORKFLOW]  Session: {session_id}")
        print(f"[WORKFLOW]  Agents: {', '.join(selected_agents)}")
        print("=" * 70)
        
        # Configure execution with proper LangSmith integration
        config = {
            "configurable": {
                "thread_id": f"analysis_{hash(file_path)}",
                "session_id": session_id
            },
            "recursion_limit": 100
        }
        
        # Add LangSmith configuration
        if LANGSMITH_AVAILABLE:
            config.update({
                "tags": ["langgraph", "code-analysis", "multi-agent", language],
                "metadata": {
                    "project": self.langsmith_project,
                    "file_path": file_path,
                    "file_name": Path(file_path).name,
                    "language": language,
                    "agents": selected_agents,
                    "session_id": session_id
                }
            })
        
        # Execute workflow with optional LangSmith tracing
        try:
            # Simplified execution without trace decorator to avoid issues
            final_state = await self.workflow.ainvoke(initial_state, config=config)
                
        except Exception as e:
            print(f"[WORKFLOW]  Execution failed: {e}")
            return {
                'file_path': file_path,
                'language': language,
                'error': str(e),
                'workflow_engine': 'langgraph',
                'session_id': session_id,
                'processing_time': 0,
                'total_issues': 0,
                'completed_agents': [],
                'failed_agents': selected_agents
            }
        
        # Prepare comprehensive result
        result = {
            'file_path': file_path,
            'language': language,
            'total_lines': len(code_content.split('\n')),
            'total_issues': len(final_state.get('all_issues', [])),
            'issues_by_severity': final_state.get('issues_by_severity', {}),
            'agent_performance': final_state.get('agent_performance', {}),
            'all_issues': final_state.get('all_issues', []),
            'processing_time': final_state.get('total_processing_time', 0),
            'total_tokens': final_state.get('total_tokens', 0),
            'llm_calls': final_state.get('total_llm_calls', 0),
            'workflow_engine': 'langgraph',
            'completed_agents': final_state.get('completed_agents', []),
            'failed_agents': final_state.get('failed_agents', []),
            'cache_hit': False,
            'langsmith_project': self.langsmith_project,
            'langsmith_session_id': session_id,
            'langsmith_enabled': LANGSMITH_AVAILABLE
        }
        
        # Add LangSmith dashboard URLs
        if LANGSMITH_AVAILABLE:
            result.update({
                'langsmith_dashboard_url': f"https://smith.langchain.com/projects/{self.langsmith_project}",
                'langsmith_session_url': f"https://smith.langchain.com/projects/{self.langsmith_project}/sessions/{session_id}",
                'langsmith_runs_url': f"https://smith.langchain.com/projects/{self.langsmith_project}/runs"
            })
        
        # Cache the result
        if self.cache_enabled and self.cache and 'error' not in result:
            self.cache.save_analysis_result(file_path, selected_agents, result)
        
        # Print final summary with LangSmith links
        print("\n" + "=" * 70)
        print(f"[SUMMARY]  Analysis Complete!")
        print(f"[SUMMARY] [COMP] Issues found: {result['total_issues']}")
        print(f"[SUMMARY]  Time taken: {result['processing_time']:.2f}s")
        print(f"[SUMMARY]  Agents completed: {len(result['completed_agents'])}")
        
        if LANGSMITH_AVAILABLE:
            print(f"[SUMMARY]  LangSmith Dashboard:")
            print(f"[SUMMARY]   [COMP] Project: {result['langsmith_dashboard_url']}")
            print(f"[SUMMARY]    Session: {result['langsmith_session_url']}")
            print(f"[SUMMARY]    All Runs: {result['langsmith_runs_url']}")
        
        return result
    
    def get_workflow_graph_url(self) -> Optional[str]:
        """Get URL to view the workflow graph structure"""
        if LANGSMITH_AVAILABLE:
            return f"https://smith.langchain.com/projects/{self.langsmith_project}/runs?tab=graph"
        return None
    
    def print_workflow_structure(self):
        """Print the workflow structure for debugging"""
        print("\n[WORKFLOW STRUCTURE]")
        print("Nodes:")
        print("  1. initialize")
        print("  2. setup_rag") 
        print("  3. route_agents")
        print("  4. security_agent")
        print("  5. complexity_agent")
        print("  6. performance_agent")
        print("  7. documentation_agent")
        print("  8. aggregate_results")
        print("  9. finalize")
        print("\nFlow:")
        print("  initialize -> setup_rag -> route_agents -> [agents] -> aggregate_results -> finalize")
        print(f"\nLangSmith Project: {self.langsmith_project}")
        if LANGSMITH_AVAILABLE:
            print(f"Dashboard: https://smith.langchain.com/projects/{self.langsmith_project}")