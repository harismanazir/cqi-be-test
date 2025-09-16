"""
LangGraph Workflow Graph for Multi-Agent Code Analysis
Implements intelligent agent orchestration with dependency management
"""

import os
import time
import asyncio
from typing import Dict, List, Any, Optional
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_NEW = True
except ImportError:
    try:
        # Older LangGraph version compatibility
        from langgraph.graph import StateGraph
        from langgraph.graph.state import END
        MemorySaver = None
        LANGGRAPH_NEW = False
    except ImportError:
        raise ImportError("LangGraph is not installed. Please install with: pip install langgraph")
from workflow.state import (
    WorkflowState, 
    AgentResult, 
    CodeAnalysisInput,
    get_ready_agents, 
    should_continue_workflow,
    add_agent_insights,
    get_cross_agent_context,
    initialize_workflow_state
)

# Import existing agents
from agents import (
    SecurityAgent,
    PerformanceAgent, 
    ComplexityAgent,
    DocumentationAgent,
    DuplicationAgent
)
# Remove QA agent import for now - will use existing agents only

class LangGraphMultiAgentAnalyzer:
    """LangGraph-powered multi-agent code analyzer with intelligent orchestration"""
    
    def __init__(self, enable_rag: bool = True, enable_cache: bool = True):
        self.enable_rag = enable_rag
        self.enable_cache = enable_cache
        
        # Initialize intelligent caching system
        self.cache = None
        if enable_cache:
            try:
                from agents.intelligent_cache import get_cache
                self.cache = get_cache()
                self.cache.cleanup_old_cache()
                self.cache_enabled = True
                print("[LANGGRAPH] Intelligent caching system enabled")
            except ImportError:
                self.cache_enabled = False
                print("[LANGGRAPH] Caching system not available")
        else:
            self.cache_enabled = False
        
        # Initialize agents (reuse existing implementations)
        self.rag_analyzer = None
        if enable_rag:
            try:
                from agents.rag_agent import RAGCodeAnalyzer
                self.rag_analyzer = RAGCodeAnalyzer()
                print("[RAG] RAG system enabled for LangGraph workflow")
            except ImportError:
                print("[WARN] RAG system not available, continuing without RAG")
        
        self.agents = {
            'security': SecurityAgent(rag_analyzer=self.rag_analyzer),
            'performance': PerformanceAgent(rag_analyzer=self.rag_analyzer), 
            'complexity': ComplexityAgent(rag_analyzer=self.rag_analyzer),
            'documentation': DocumentationAgent(rag_analyzer=self.rag_analyzer),
            'duplication': DuplicationAgent(rag_analyzer=self.rag_analyzer),
            # Remove testing agent for now - focus on core analysis agents
        }
        
        # Memory for workflow state persistence (if available) - initialize first
        self.memory = MemorySaver() if MemorySaver else None
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
        
        print(f"[LANGGRAPH] Multi-agent workflow initialized with {len(self.agents)} agents")
        print(f"[LANGGRAPH] Dependency-aware execution enabled")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with intelligent routing"""
        
        # Create state graph
        workflow = StateGraph(WorkflowState)
        
        # Add workflow nodes
        workflow.add_node("initialize_analysis", self._initialize_analysis)
        workflow.add_node("setup_rag", self._setup_rag)
        workflow.add_node("route_agents", self._route_agents)
        workflow.add_node("run_security_agent", self._run_security_agent)
        workflow.add_node("run_complexity_agent", self._run_complexity_agent)
        workflow.add_node("run_performance_agent", self._run_performance_agent)
# Remove testing agent node for now
        workflow.add_node("run_documentation_agent", self._run_documentation_agent)
        workflow.add_node("run_duplication_agent", self._run_duplication_agent)
        workflow.add_node("aggregate_results", self._aggregate_results)
        workflow.add_node("finalize_analysis", self._finalize_analysis)
        
        # Define workflow edges with intelligent routing
        workflow.set_entry_point("initialize_analysis")
        
        # Linear initialization flow
        workflow.add_edge("initialize_analysis", "setup_rag")
        workflow.add_edge("setup_rag", "route_agents")
        
        # Conditional routing from route_agents to individual agents
        workflow.add_conditional_edges(
            "route_agents",
            self._determine_next_agents,
            {
                "security": "run_security_agent",
                "complexity": "run_complexity_agent", 
                "performance": "run_performance_agent",
# Remove testing route for now
                "documentation": "run_documentation_agent",
                "duplication": "run_duplication_agent",
                "aggregate": "aggregate_results"
            }
        )
        
        # Agent completion flows - route back to route_agents or aggregate
        for agent_name in ['security', 'complexity', 'performance', 'documentation', 'duplication']:
            workflow.add_conditional_edges(
                f"run_{agent_name}_agent",
                self._agent_completion_router,
                {
                    "continue": "route_agents",    # More agents to run
                    "aggregate": "aggregate_results"  # All done, aggregate
                }
            )
        
        # Final flow
        workflow.add_edge("aggregate_results", "finalize_analysis")
        workflow.add_edge("finalize_analysis", END)
        
        if self.memory:
            return workflow.compile(checkpointer=self.memory)
        else:
            return workflow.compile()
    
    # ============ WORKFLOW NODES ============
    
    async def _initialize_analysis(self, state: WorkflowState) -> Dict[str, Any]:
        """Initialize the analysis workflow"""
        print(f"[LANGGRAPH] Starting LangGraph Multi-Agent Analysis")
        print(f"[LANGGRAPH] File: {state['analysis_input'].file_path}")
        print(f"[LANGGRAPH] Language: {state['analysis_input'].language}")
        print(f"[LANGGRAPH] Selected Agents: {', '.join(state['analysis_input'].selected_agents)}")
        
        return {
            "workflow_status": "analyzing",
            "total_processing_time": time.time()
        }
    
    async def _setup_rag(self, state: WorkflowState) -> Dict[str, Any]:
        """Setup RAG system if enabled"""
        updates = {}
        
        if state['rag_enabled'] and self.enable_rag and self.rag_analyzer:
            print("[LANGGRAPH] Setting up RAG context...")
            
            # Index the current file for RAG context
            analysis_input = state['analysis_input']
            try:
                from agents.base_agent import LanguageDetector
                language_detector = LanguageDetector()
                
                # Index just this file for efficient RAG context
                files_to_index = [analysis_input.file_path]
                self.rag_analyzer.index_codebase(files_to_index, language_detector)
                
                print(f"[LANGGRAPH] RAG context ready for {analysis_input.file_path}")
            except Exception as e:
                print(f"[LANGGRAPH] RAG setup failed: {str(e)}")
        
        return updates
    
    # RAG Agent removed - agents handle RAG internally
    
    async def _route_agents(self, state: WorkflowState) -> Dict[str, Any]:
        """Route to next available agents based on dependencies"""
        print(f"[LANGGRAPH] Routing agents...")
        print(f"[LANGGRAPH] Completed: {state['completed_agents']}")
        print(f"[LANGGRAPH] Pending: {state['pending_agents']}")
        
        return {}
    
    # Individual agent execution nodes
    async def _run_security_agent(self, state: WorkflowState) -> Dict[str, Any]:
        """Run security analysis agent"""
        return await self._run_agent('security', state)
    
    async def _run_complexity_agent(self, state: WorkflowState) -> Dict[str, Any]:
        """Run complexity analysis agent"""
        return await self._run_agent('complexity', state)
    
    async def _run_performance_agent(self, state: WorkflowState) -> Dict[str, Any]:
        """Run performance analysis agent"""
        return await self._run_agent('performance', state)
    
# Remove testing agent method for now
    
    async def _run_documentation_agent(self, state: WorkflowState) -> Dict[str, Any]:
        """Run documentation analysis agent"""
        return await self._run_agent('documentation', state)
    
    async def _run_duplication_agent(self, state: WorkflowState) -> Dict[str, Any]:
        """Run duplication analysis agent"""
        return await self._run_agent('duplication', state)
    
    async def _run_agent(self, agent_name: str, state: WorkflowState) -> Dict[str, Any]:
        """Generic agent execution with cross-agent context"""
        print(f"[LANGGRAPH] Running {agent_name.title()} Agent...")
        
        # Smart rate limiting - only delay if we made recent API calls
        if len(state['completed_agents']) > 0 and state.get('total_llm_calls', 0) > 0:
            print(f"[LANGGRAPH] Rate limit protection: waiting 5s...")
            await asyncio.sleep(5)  # Increased delay to avoid rate limits
        
        start_time = time.time()
        agent = self.agents[agent_name]
        analysis_input = state['analysis_input']
        
        # Get cross-agent context for enhanced analysis
        cross_context = get_cross_agent_context(state, agent_name)
        
        # Retry logic for rate limits
        max_retries = 3
        retry_delay = 20  # seconds
        
        for attempt in range(max_retries):
            try:
                # Run the agent analysis
                result = await agent.analyze(
                    analysis_input.code_content, 
                    analysis_input.file_path, 
                    analysis_input.language
                )
                break  # Success, exit retry loop
                
            except Exception as e:
                error_msg = str(e)
                if "rate_limit_exceeded" in error_msg and attempt < max_retries - 1:
                    # Check if it's daily limit vs per-minute limit
                    if "tokens per day" in error_msg or "TPD:" in error_msg:
                        print(f"[LANGGRAPH] Daily token limit reached! Stopping analysis.")
                        print(f"[LANGGRAPH] Wait until tomorrow or upgrade Groq plan.")
                        # For daily limits, don't retry - fail immediately
                        raise e
                    else:
                        # For per-minute limits, retry with exponential backoff
                        print(f"[LANGGRAPH] Rate limit hit, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                else:
                    raise e  # Re-raise if not rate limit or max retries reached
        
        processing_time = time.time() - start_time
        
        # Create agent result
        agent_result = AgentResult(
            agent_name=agent_name,
            issues=result.get('issues', []),
            processing_time=processing_time,
            tokens_used=result.get('tokens_used', 0),
            confidence=result.get('confidence', 0.8),
            status='completed',
            llm_calls=result.get('llm_calls', 0)  # Include LLM call count
        )
        
        # Extract insights to share with other agents
        insights = self._extract_insights(agent_result)
        add_agent_insights(state, agent_name, insights)
        
        print(f"[LANGGRAPH] SUCCESS {agent_name.title()} Agent: {len(agent_result.issues)} issues in {processing_time:.2f}s")
        
        # Update state
        updates = {
            'completed_agents': [agent_name],
            'agent_results': {agent_name: agent_result},
            'pending_agents': [a for a in state['pending_agents'] if a != agent_name],
            'total_tokens': state['total_tokens'] + agent_result.tokens_used,
            'total_llm_calls': state['total_llm_calls'] + agent_result.llm_calls  # Use actual LLM call count
        }


        return updates
    
    async def _aggregate_results(self, state: WorkflowState) -> Dict[str, Any]:
        """Aggregate results from all completed agents"""
        print("[LANGGRAPH] Aggregating results from all agents...")
        
        all_issues = []
        issues_by_severity = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        agent_performance = {}
        
        # Aggregate from all agent results
        for agent_name, result in state['agent_results'].items():
            for issue in result.issues:
                issue['agent'] = agent_name
                issue['file_path'] = state['analysis_input'].file_path
                all_issues.append(issue)
                
                severity = issue.get('severity', 'low')
                if severity in issues_by_severity:
                    issues_by_severity[severity] += 1
            
            agent_performance[agent_name] = {
                'issues_found': len(result.issues),
                'processing_time': result.processing_time,
                'tokens_used': result.tokens_used,
                'confidence': result.confidence,
                'status': result.status
            }
        
        print(f"[LANGGRAPH] Aggregated {len(all_issues)} total issues")
        
        return {
            'all_issues': all_issues,
            'issues_by_severity': issues_by_severity,
            'agent_performance': agent_performance
        }
    
    async def _finalize_analysis(self, state: WorkflowState) -> Dict[str, Any]:
        """Finalize the analysis workflow"""
        total_time = time.time() - state['total_processing_time']
        
        print(f"[LANGGRAPH] Analysis completed in {total_time:.2f}s")
        print(f"[LANGGRAPH] Total issues found: {len(state['all_issues'])}")
        print(f"[LANGGRAPH] Successful agents: {len(state['completed_agents'])}")
        print(f"[LANGGRAPH] Failed agents: {len(state['failed_agents'])}")
        
        return {
            'workflow_status': 'completed',
            'total_processing_time': total_time
        }
    
    # ============ ROUTING FUNCTIONS ============
    
    def _determine_next_agents(self, state: WorkflowState) -> str:
        """Determine which agents should run next based on dependencies"""
        ready_agents = get_ready_agents(state)
        
        if not ready_agents:
            return "aggregate"  # No more agents ready, move to aggregation
        
        # Return the highest priority ready agent
        # In LangGraph, we can only return one route, but the agent will loop back
        priority_agent = min(ready_agents, key=lambda x: self._get_agent_priority(x))
        return priority_agent
    
    def _agent_completion_router(self, state: WorkflowState) -> str:
        """Route after agent completion - continue or aggregate"""
        if should_continue_workflow(state):
            return "continue"  # More agents to run
        else:
            return "aggregate"  # All done
    
    def _get_agent_priority(self, agent_name: str) -> int:
        """Get agent priority for routing"""
        priority_map = {
            'security': 1,
            'complexity': 2, 
            'performance': 3,
            'documentation': 4,
            'duplication': 5
        }
        return priority_map.get(agent_name, 10)
    
    def _extract_insights(self, agent_result: AgentResult) -> List[str]:
        """Extract key insights from agent results to share with other agents"""
        insights = []
        
        for issue in agent_result.issues:
            if issue.get('severity') in ['critical', 'high']:
                insight = f"{issue.get('title', 'Unknown issue')} - {issue.get('suggestion', 'No suggestion')}"
                insights.append(insight)
        
        return insights[:3]  # Limit to top 3 insights
    
    # ============ PUBLIC API ============
    
    async def analyze_file(self, file_path: str, code_content: str, language: str, 
                          selected_agents: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze a single file using LangGraph workflow"""
        
        # Prepare analysis input
        selected_agents = selected_agents or ['security', 'complexity', 'performance', 'documentation']
        
        # Check cache first - FAST PATH for cache hits
        if self.cache_enabled and self.cache:
            cached_result = self.cache.get_analysis_result(file_path, selected_agents)
            if cached_result:
                print(f"[LANGGRAPH-CACHE] Cache HIT: {os.path.basename(file_path)} (using cached analysis)")
                # Add LangGraph-specific metadata to cached result
                cached_result['workflow_engine'] = 'langgraph'
                cached_result['cache_hit'] = True
                # Update performance metrics for cache hit
                cached_result['processing_time'] = 0.01  # Near-instant for cache
                cached_result['total_tokens'] = cached_result.get('total_tokens', 0)  # Keep original token count
                cached_result['llm_calls'] = 0  # No LLM calls for cache hit
                # Return immediately - no workflow execution needed
                return cached_result
            else:
                print(f"[LANGGRAPH-CACHE] Cache MISS: {os.path.basename(file_path)} (running analysis...)")
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
        
        print(f"\n[LANGGRAPH] Starting LangGraph workflow for {file_path}")
        print(f"[LANGGRAPH] Agents: {', '.join(selected_agents)}")
        print("=" * 70)
        
        # Execute workflow with increased recursion limit
        config = {
            "configurable": {"thread_id": f"analysis_{hash(file_path)}"},
            "recursion_limit": 100  # Increase from default 25 to 100
        }
        final_state = await self.workflow.ainvoke(initial_state, config=config)
        
        # Prepare results in expected format
        result = {
            'file_path': file_path,
            'language': language,
            'total_lines': len(code_content.split('\n')),
            'total_issues': len(final_state['all_issues']),
            'issues_by_severity': final_state['issues_by_severity'],
            'agent_performance': final_state['agent_performance'],
            'all_issues': final_state['all_issues'],
            'processing_time': final_state['total_processing_time'],
            'total_tokens': final_state['total_tokens'],
            'llm_calls': final_state['total_llm_calls'],
            'workflow_engine': 'langgraph',
            'completed_agents': final_state['completed_agents'],
            'failed_agents': final_state['failed_agents'],
            'cache_hit': False
        }
        
        # Cache the result for future runs
        if self.cache_enabled and self.cache and 'error' not in final_state:
            self.cache.save_analysis_result(file_path, selected_agents, result)
            print(f"[LANGGRAPH-CACHE] Analysis cached for future runs")
        
        return result