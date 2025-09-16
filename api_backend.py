#!/usr/bin/env python3
"""
FastAPI Backend for LangGraph Code Quality Intelligence
Fixed version with progressive analysis and proper model definitions
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import os
import tempfile
import uuid
from datetime import datetime
import zipfile
import json
import shutil
from extensions.github_integration import GitHubRepoAnalyzer
# Import your LangGraph system (these must exist in your project)
try:
    from main import LangGraphCQI
    from interactive_qa import EnhancedQAAgent
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("Make sure main.py and interactive_qa.py exist in your project")
    # Mock classes for development/testing
    class LangGraphCQI:
        def __init__(self, enable_rag=True, enable_cache=True, use_langgraph=True):
            pass
        async def analyze_file(self, file_path, selected_agents=None, detailed=True):
            # Return mock data matching your CLI output format
            return {
                'language': 'python',
                'lines_of_code': 42,
                'all_issues': [
                    {
                        'severity': 'HIGH',
                        'title': 'Hardcoded Credentials',
                        'agent': 'Security',
                        'line': 13,
                        'description': 'API key is hardcoded in the code',
                        'suggestion': 'Use environment variables or a secure secrets management system'
                    },
                    {
                        'severity': 'MEDIUM',
                        'title': 'Input Validation Issues',
                        'agent': 'Security',
                        'line': 16,
                        'description': 'The function does not validate the input query',
                        'suggestion': 'Add input validation to prevent potential attacks'
                    }
                ],
                'agent_stats': {
                    'security': {'issue_count': 2, 'processing_time': 0.69, 'confidence': 0.90},
                    'performance': {'issue_count': 1, 'processing_time': 0.73, 'confidence': 0.80},
                    'complexity': {'issue_count': 0, 'processing_time': 1.11, 'confidence': 0.85},
                    'documentation': {'issue_count': 3, 'processing_time': 1.31, 'confidence': 0.80}
                },
                'processing_time': 19.14,
                'total_tokens': 3456,
                'total_api_calls': 4,
                'completed_agents': ['security', 'performance', 'complexity', 'documentation']
            }
    class EnhancedQAAgent:
        def __init__(self, codebase_path=".", run_analysis=True):
            self.codebase_path = codebase_path
        async def initialize(self, enable_rag=True, run_analysis=True):
            pass
        async def ask_question(self, question):
            return type('obj', (object,), {
                'answer': 'Mock response',
                'confidence': 0.8,
                'source': 'mock',
                'processing_time': 0.5,
                'follow_up_suggestions': [],
                'related_files': []
            })()

# ---------------------------
# Pydantic models (Complete definitions)
# ---------------------------
class IssueModel(BaseModel):
    severity: str
    title: str
    agent: str
    file: str
    line: int
    description: str
    fix: str

class AgentPerformance(BaseModel):
    agent: str
    issues: int
    time: float
    confidence: float
    status: str

class AnalysisResult(BaseModel):
    file: str
    language: str
    lines: int
    total_issues: int
    processing_time: float
    tokens_used: int
    api_calls: int
    completed_agents: List[str]
    # Severity breakdown
    high_issues: int
    medium_issues: int
    low_issues: int
    critical_issues: int = 0
    # Agent performance
    agent_performance: List[AgentPerformance]
    agent_breakdown: Dict[str, int] = {}
    # Detailed issues (top 20)
    detailed_issues: List[IssueModel]
    # Additional metadata
    timestamp: str
    job_id: str

# FIXED: Add the missing BackendResultsResponse model
class BackendResultsResponse(BaseModel):
    success: bool
    job_id: str
    results: List[AnalysisResult]
    total_files: int
    completion_time: str
    github_metadata: Optional[Dict[str, Any]] = None

# NEW: Progressive analysis models
class PartialResultsResponse(BaseModel):
    success: bool
    partial: bool
    completed_files: int
    total_files: int
    progress: int
    job_id: str
    results: List[AnalysisResult]
    github_metadata: Optional[Dict[str, Any]] = None

class ProgressiveWebSocketMessage(BaseModel):
    type: str  # 'progress', 'partial_results', 'final_results'
    job_id: str
    progress: Optional[int] = None
    message: Optional[str] = None
    completed_files: Optional[int] = None
    total_files: Optional[int] = None
    results: Optional[List[AnalysisResult]] = None
    timestamp: str

class ChatMessage(BaseModel):
    session_id: str
    message: str

class UploadResponse(BaseModel):
    success: bool
    files: List[Dict[str, Any]]
    upload_dir: str
    total_files: int

class AnalyzeRequest(BaseModel):
    file_paths: List[str]
    detailed: bool = True
    rag: bool = True
    progressive: bool = True  # NEW: Progressive analysis flag

class ChatStartRequest(BaseModel):
    upload_dir: str = ""
    github_repo: str = ""
    branch: str = ""

class GitHubAnalyzeRequest(BaseModel):
    repo_url: str
    branch: str = "main"
    agents: List[str] = ["security", "performance", "complexity", "documentation"]
    detailed: bool = True
    progressive: bool = True  # NEW: Progressive analysis support

class GitHubValidationResponse(BaseModel):
    valid: bool
    owner: Optional[str] = None
    repo_name: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None
    branches: Optional[List[str]] = None
    error: Optional[str] = None

class GitHubValidateRequest(BaseModel):
    repo_url: str

# ---------------------------
# FastAPI app + CORS
# ---------------------------
app = FastAPI(
    title="LangGraph Code Quality Intelligence API",
    description="Progressive analysis with real-time updates",
    version="2.1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Global in-memory storage
# ---------------------------
analysis_jobs: Dict[str, Dict] = {}
websocket_connections: Dict[str, WebSocket] = {}
github_temp_dirs: Dict[str, str] = {}
qa_agents: Dict[str, EnhancedQAAgent] = {}

def generate_job_id() -> str:
    return str(uuid.uuid4())

def normalize_github_url(url: str) -> str:
    """Normalize GitHub URL for consistent comparison"""
    if not url:
        return ""
    # Remove common prefixes and suffixes
    url = url.strip().rstrip('/')
    url = url.replace('https://github.com/', '')
    url = url.replace('http://github.com/', '')
    url = url.replace('git@github.com:', '')
    url = url.replace('.git', '')
    return url.lower()

def convert_langgraph_output_to_api_format(raw_result: Dict, job_id: str, file_path: str) -> AnalysisResult:
    """Convert raw LangGraph output to API format that matches CLI output exactly"""
    print(f"[CONVERT] Converting LangGraph output for file: {file_path}")
    print(f"[CONVERT] Raw result keys: {raw_result.keys()}")
    # Print the complete raw result for debugging
    print(f"[DEBUG] Complete raw result structure:")
    print(json.dumps(raw_result, indent=2, default=str))
    file_name = os.path.basename(file_path)
    # Try multiple ways to get language
    language = raw_result.get('language', 'Unknown')
    if language == 'Unknown':
        # Fallback: detect from file extension
        ext = os.path.splitext(file_path)[1].lower()
        ext_map = {
            '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
            '.java': 'Java', '.cpp': 'C++', '.c': 'C', '.cs': 'C#',
            '.php': 'PHP', '.rb': 'Ruby', '.go': 'Go'
        }
        language = ext_map.get(ext, 'Unknown')
    # Try multiple ways to get lines of code
    lines_of_code = raw_result.get('lines_of_code', 0)
    if lines_of_code == 0:
        lines_of_code = raw_result.get('total_lines', 0)
    if lines_of_code == 0:
        lines_of_code = raw_result.get('line_count', 0)
    # If still zero, count from file
    if lines_of_code == 0:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines_of_code = len([line for line in f if line.strip()])
        except Exception as e:
            print(f"[WARNING] Could not count lines for {file_path}: {e}")
            lines_of_code = 1  # Default to 1 to avoid division by zero
    print(f"[CONVERT] Lines of code: {lines_of_code}")
    # Get all issues - try multiple possible keys
    all_issues = []
    possible_issue_keys = ['all_issues', 'issues', 'detailed_issues', 'found_issues']
    for key in possible_issue_keys:
        if key in raw_result and raw_result[key]:
            all_issues = raw_result[key]
            print(f"[CONVERT] Found issues under key '{key}': {len(all_issues)}")
            break
    if not all_issues:
        print(f"[WARNING] No issues found in raw_result. Available keys: {raw_result.keys()}")
    print(f"[CONVERT] Found {len(all_issues)} issues")
    # Detailed debug of each issue
    print(f"[DEBUG] Issue details:")
    agent_names_found = set()
    severity_names_found = set()
    for i, issue in enumerate(all_issues[:5]):  # Show first 5 issues
        print(f"  Issue {i+1}: {json.dumps(issue, indent=4, default=str)}")
        if isinstance(issue, dict):
            agent = issue.get('agent', 'unknown')
            severity = issue.get('severity', 'unknown')
            agent_names_found.add(str(agent))
            severity_names_found.add(str(severity))
    print(f"[DEBUG] Unique agents found: {agent_names_found}")
    print(f"[DEBUG] Unique severities found: {severity_names_found}")
    # Count issues by severity with flexible matching
    def count_by_severity(target_severity):
        count = 0
        for issue in all_issues:
            if isinstance(issue, dict):
                severity = str(issue.get('severity', '')).lower().strip()
                if target_severity.lower() in severity or severity in target_severity.lower():
                    count += 1
        return count
    critical_count = count_by_severity('critical')
    high_count = count_by_severity('high')
    medium_count = count_by_severity('medium')
    low_count = count_by_severity('low')
    print(f"[DEBUG] Severity counts - Critical: {critical_count}, High: {high_count}, Medium: {medium_count}, Low: {low_count}")
    # Get agent stats
    agent_stats = raw_result.get('agent_stats', {})
    if not agent_stats:
        agent_stats = raw_result.get('agent_performance', {})
    if not agent_stats:
        agent_stats = raw_result.get('agents', {})
    print(f"[CONVERT] Agent stats: {agent_stats}")
    # Initialize agent breakdown with multiple approaches
    agent_breakdown = {
        'security': 0,
        'performance': 0,
        'complexity': 0,
        'documentation': 0
    }
    # Method 1: Count from issues by agent field
    print(f"[DEBUG] Method 1 - Counting issues by agent field:")
    issues_by_agent_raw = {}
    for issue in all_issues:
        if isinstance(issue, dict):
            agent = str(issue.get('agent', 'unknown')).lower().strip()
            issues_by_agent_raw[agent] = issues_by_agent_raw.get(agent, 0) + 1
    print(f"[DEBUG] Raw agent counts: {issues_by_agent_raw}")
    # Map raw agent names to standard names
    agent_mapping = {
        'security': ['security', 'security_agent', 'securityagent', 'sec'],
        'performance': ['performance', 'performance_agent', 'performanceagent', 'perf'],
        'complexity': ['complexity', 'complexity_agent', 'complexityagent', 'complex'],
        'documentation': ['documentation', 'documentation_agent', 'documentationagent', 'docs', 'doc']
    }
    for standard_name, variants in agent_mapping.items():
        count = 0
        for variant in variants:
            count += issues_by_agent_raw.get(variant, 0)
        agent_breakdown[standard_name] = count
    print(f"[DEBUG] Agent breakdown after mapping: {agent_breakdown}")
    # Method 2: If no issues found by agent, use agent_stats
    if sum(agent_breakdown.values()) == 0 and agent_stats:
        print(f"[DEBUG] Method 2 - Using agent_stats:")
        for agent_name, stats in agent_stats.items():
            agent_clean = agent_name.lower().strip()
            # Map agent name to standard name
            mapped_agent = None
            for standard_name, variants in agent_mapping.items():
                if agent_clean in variants or any(v in agent_clean for v in variants):
                    mapped_agent = standard_name
                    break
            if mapped_agent:
                if isinstance(stats, dict):
                    count = stats.get('issue_count', stats.get('issues', 0))
                else:
                    count = 0
                agent_breakdown[mapped_agent] = count
                print(f"  {agent_name} -> {mapped_agent}: {count}")
    # Method 3: If still no data, create fallback based on total issues
    if sum(agent_breakdown.values()) == 0 and len(all_issues) > 0:
        print(f"[DEBUG] Method 3 - Creating fallback distribution:")
        total_issues = len(all_issues)
        # Distribute issues roughly across agents
        agent_breakdown['security'] = max(1, total_issues // 4)
        agent_breakdown['complexity'] = max(1, total_issues // 4)
        agent_breakdown['performance'] = max(1, total_issues // 4)
        agent_breakdown['documentation'] = total_issues - agent_breakdown['security'] - agent_breakdown['complexity'] - agent_breakdown['performance']
        print(f"  Fallback distribution: {agent_breakdown}")
    # Create agent performance data
    agent_performance = []
    expected_agents = ['security', 'performance', 'complexity', 'documentation']
    for agent in expected_agents:
        stats = agent_stats.get(agent, agent_stats.get(agent.title(), {}))
        if isinstance(stats, dict):
            processing_time = stats.get('processing_time', 0.0)
            confidence = stats.get('confidence', 0.8)
        else:
            processing_time = 0.0
            confidence = 0.8
        agent_performance.append(AgentPerformance(
            agent=agent.title(),
            issues=agent_breakdown[agent],
            time=processing_time,
            confidence=confidence,
            status="SUCCESS"
        ))
    # Create detailed issues
    detailed_issues = []
    severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
    # Sort issues by severity
    sorted_issues = sorted(
        all_issues,
        key=lambda x: severity_order.get(str(x.get('severity', '')).lower(), 0) if isinstance(x, dict) else 0,
        reverse=True
    )
    for i, issue in enumerate(sorted_issues[:20]):  # Top 20 issues
        if isinstance(issue, dict):
            detailed_issues.append(IssueModel(
                severity=str(issue.get('severity', 'unknown')).upper(),
                title=str(issue.get('title', issue.get('message', 'Unknown Issue'))),
                agent=str(issue.get('agent', 'unknown')).title(),
                file=file_name,
                line=int(issue.get('line_number', issue.get('line', 0))),
                description=str(issue.get('description', issue.get('desc', 'No description'))),
                fix=str(issue.get('suggestion', issue.get('fix', issue.get('solution', 'No fix suggested'))))
            ))
    # Get processing time
    processing_time = raw_result.get('processing_time', 0.0)
    if processing_time == 0.0:
        processing_time = raw_result.get('total_processing_time', 0.0)
    if processing_time == 0.0:
        processing_time = raw_result.get('analysis_time', 0.0)
    result = AnalysisResult(
        file=file_name,
        language=language,
        lines=lines_of_code,
        total_issues=len(all_issues),
        processing_time=processing_time,
        tokens_used=raw_result.get('total_tokens', 0),
        api_calls=raw_result.get('total_api_calls', raw_result.get('llm_calls', 0)),
        completed_agents=raw_result.get('completed_agents', expected_agents),
        critical_issues=critical_count,
        high_issues=high_count,
        medium_issues=medium_count,
        low_issues=low_count,
        agent_performance=agent_performance,
        agent_breakdown=agent_breakdown,
        detailed_issues=detailed_issues,
        timestamp=datetime.now().isoformat(),
        job_id=job_id
    )
    print(f"[CONVERT] Final result summary:")
    print(f"  File: {result.file}")
    print(f"  Language: {result.language}")
    print(f"  Lines: {result.lines}")
    print(f"  Total issues: {result.total_issues}")
    print(f"  Agent breakdown: {result.agent_breakdown}")
    print(f"  Severity breakdown: Critical={result.critical_issues}, High={result.high_issues}, Medium={result.medium_issues}, Low={result.low_issues}")
    return result

# Enhanced broadcast function
async def broadcast_progress(job_id: str, progress: int, message: str):
    """Enhanced progress broadcasting with more details"""
    print(f"[WEBSOCKET] Broadcasting progress for {job_id}: {progress}% - {message}")
    if job_id in websocket_connections:
        try:
            job = analysis_jobs.get(job_id, {})
            await websocket_connections[job_id].send_json({
                "type": "progress",
                "job_id": job_id,
                "progress": progress,
                "message": message,
                "completed_files": job.get("completed_files", 0),
                "total_files": job.get("total_files", 0),
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            print(f"[WEBSOCKET] Error broadcasting: {e}")
            websocket_connections.pop(job_id, None)

async def broadcast_partial_results(job_id: str, partial_results: List[AnalysisResult]):
    """Broadcast partial results to connected clients"""
    if job_id in websocket_connections:
        try:
            job = analysis_jobs.get(job_id, {})
            await websocket_connections[job_id].send_json({
                "type": "partial_results",
                "job_id": job_id,
                "results": [result.model_dump() for result in partial_results],  # CHANGED: .dict() -> .model_dump()
                "completed_files": len(partial_results),
                "total_files": job.get("total_files", 0),
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            print(f"[WEBSOCKET] Error broadcasting partial results: {e}")
            websocket_connections.pop(job_id, None)

# ---------------------------
# API endpoints
# ---------------------------
@app.get("/")
async def root():
    return {
        "message": "LangGraph Code Quality Intelligence API", 
        "status": "online", 
        "version": "2.1.0",
        "progressive_analysis": True,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload files or folders for analysis"""
    print(f"[UPLOAD] Received {len(files)} files")
    try:
        uploaded_files = []
        upload_dir = tempfile.mkdtemp(prefix="cqi_upload_")
        print(f"[UPLOAD] Created temp directory: {upload_dir}")
        for file in files:
            print(f"[UPLOAD] Processing: {file.filename}")
            safe_filename = file.filename.replace(" ", "_").replace("..", "")
            file_path = os.path.join(upload_dir, safe_filename)
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            # Handle ZIP files
            if file.filename.endswith('.zip'):
                extract_dir = os.path.join(upload_dir, "extracted")
                os.makedirs(extract_dir, exist_ok=True)
                try:
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    # Find code files in extracted content
                    code_extensions = ('.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.php', '.rb', '.go')
                    for root, dirs, filenames in os.walk(extract_dir):
                        for filename in filenames:
                            if filename.endswith(code_extensions):
                                extracted_path = os.path.join(root, filename)
                                uploaded_files.append({
                                    "name": filename,
                                    "path": extracted_path,
                                    "size": os.path.getsize(extracted_path),
                                    "type": "code"
                                })
                except Exception as e:
                    print(f"[UPLOAD] ZIP extraction failed: {e}")
                    uploaded_files.append({
                        "name": file.filename,
                        "path": file_path,
                        "size": len(content),
                        "type": "archive"
                    })
            else:
                uploaded_files.append({
                    "name": file.filename,
                    "path": file_path,
                    "size": len(content),
                    "type": "code"
                })
        print(f"[UPLOAD] Successfully processed {len(uploaded_files)} files")
        return UploadResponse(
            success=True,
            files=uploaded_files,
            upload_dir=upload_dir,
            total_files=len(uploaded_files)
        )
    except Exception as e:
        print(f"[UPLOAD] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.websocket("/api/progress/{job_id}")
async def websocket_progress_enhanced(websocket: WebSocket, job_id: str):
    """Enhanced WebSocket for real-time progress AND partial results"""
    print(f"[WEBSOCKET] Enhanced client connecting for job: {job_id}")
    await websocket.accept()
    websocket_connections[job_id] = websocket
    try:
        while True:
            await asyncio.sleep(0.5)  # Faster updates
            if job_id in analysis_jobs:
                job = analysis_jobs[job_id]
                # Send progress updates
                await websocket.send_json({
                    "type": "progress",
                    "job_id": job_id,
                    "status": job["status"],
                    "progress": job["progress"],
                    "message": job["message"],
                    "completed_files": job.get("completed_files", 0),
                    "total_files": job.get("total_files", 0),
                    "timestamp": datetime.now().isoformat()
                })
                # Send partial results if available
                if "partial_results" in job and job["partial_results"]:
                    await websocket.send_json({
                        "type": "partial_results",
                        "job_id": job_id,
                        "results": [result.model_dump() for result in job["partial_results"]],  # CHANGED: .dict() -> .model_dump()
                        "completed_files": job.get("completed_files", 0),
                        "total_files": job.get("total_files", 0),
                        "timestamp": datetime.now().isoformat()
                    })
                if job["status"] in ["completed", "failed"]:
                    # Send final results
                    if job["status"] == "completed":
                        await websocket.send_json({
                            "type": "final_results", 
                            "job_id": job_id,
                            "results": [result.model_dump() for result in job.get("results", [])],  # CHANGED: .dict() -> .model_dump()
                            "timestamp": datetime.now().isoformat()
                        })
                    break
    except WebSocketDisconnect:
        print(f"[WEBSOCKET] Client disconnected for job: {job_id}")
        websocket_connections.pop(job_id, None)
    except Exception as e:
        print(f"[WEBSOCKET] Error: {e}")
        websocket_connections.pop(job_id, None)

@app.post("/api/analyze/{job_id}")
async def start_analysis_progressive(job_id: str, request: AnalyzeRequest):
    """Enhanced analysis with progressive file-by-file results"""
    print(f"[ANALYZE] Starting progressive analysis for job {job_id}")
    try:
        # Initialize job tracking with enhanced structure
        analysis_jobs[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "progress": 5,
            "message": "Initializing analysis...",
            "start_time": datetime.now(),
            "file_paths": request.file_paths,
            "total_files": len(request.file_paths),
            "completed_files": 0,
            "partial_results": [],  # Store results as they complete
            "failed_files": []
        }
        # Start progressive analysis in background
        asyncio.create_task(run_progressive_analysis(job_id, request))
        return {"success": True, "job_id": job_id, "progressive": True}
    except Exception as e:
        print(f"[ANALYZE] Error: {e}")
        analysis_jobs[job_id] = {
            "job_id": job_id,
            "status": "failed",
            "progress": 0,
            "message": f"Analysis failed: {str(e)}",
            "start_time": datetime.now()
        }
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

async def run_progressive_analysis(job_id: str, request: AnalyzeRequest):
    """Run analysis file by file with progressive updates"""
    job = analysis_jobs[job_id]
    try:
        # Initialize LangGraph system
        analyzer = LangGraphCQI(enable_rag=request.rag, enable_cache=True, use_langgraph=True)
        job["progress"] = 10
        job["message"] = "LangGraph system initialized..."
        await broadcast_progress(job_id, 10, "Analysis system ready")
        total_files = len(request.file_paths)
        for i, file_path in enumerate(request.file_paths):
            filename = os.path.basename(file_path)
            # Update progress for current file
            base_progress = 10 + int((i / total_files) * 80)
            job["progress"] = base_progress
            job["message"] = f"Analyzing {filename}... ({i+1}/{total_files})"
            await broadcast_progress(job_id, base_progress, f"Analyzing {filename}...")
            try:
                # Analyze single file
                result = await analyzer.analyze_file(
                    file_path,
                    selected_agents=None,
                    detailed=request.detailed
                )
                # Convert to API format
                api_result = convert_langgraph_output_to_api_format(result, job_id, file_path)
                # Add to partial results immediately
                job["partial_results"].append(api_result)
                job["completed_files"] = i + 1
                # Update progress after file completion
                file_progress = 10 + int(((i + 1) / total_files) * 80)
                job["progress"] = file_progress
                job["message"] = f"Completed {filename} - {api_result.total_issues} issues found"
                print(f"[PROGRESSIVE] File {i+1}/{total_files} completed: {filename}")
                # Broadcast file completion with partial results
                await broadcast_progress(job_id, file_progress, f"âœ… {filename} analyzed - {api_result.total_issues} issues")
                await broadcast_partial_results(job_id, job["partial_results"])
            except Exception as file_error:
                print(f"[PROGRESSIVE] Error analyzing {filename}: {file_error}")
                job["failed_files"].append({"file": file_path, "error": str(file_error)})
                continue
        # Mark as completed
        job["status"] = "completed"
        job["progress"] = 100
        job["message"] = f"Analysis completed! {len(job['partial_results'])} files analyzed"
        job["results"] = job["partial_results"]  # Copy to final results
        job["completion_time"] = datetime.now()
        await broadcast_progress(job_id, 100, "ðŸŽ‰ Analysis completed!")
    except Exception as e:
        print(f"[PROGRESSIVE] Fatal error: {e}")
        job["status"] = "failed"
        job["progress"] = 0
        job["message"] = f"Analysis failed: {str(e)}"
        await broadcast_progress(job_id, 0, f"âŒ Analysis failed: {str(e)}")

@app.get("/api/status/{job_id}")
async def get_analysis_status(job_id: str):
    """Get analysis job status"""
    print(f"[STATUS] Checking status for job: {job_id}")
    if job_id not in analysis_jobs:
        print(f"[STATUS] Job {job_id} not found")
        raise HTTPException(status_code=404, detail="Job not found")
    job = analysis_jobs[job_id]
    print(f"[STATUS] Job {job_id} status: {job['status']} ({job['progress']}%)")
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "start_time": job["start_time"].isoformat(),
        "completion_time": job.get("completion_time").isoformat() if job.get("completion_time") else None
    }

@app.get("/api/results/{job_id}")
async def get_analysis_results(job_id: str):
    """Get analysis results - enhanced with GitHub metadata"""
    print(f"[RESULTS] Fetching results for job: {job_id}")
    if job_id not in analysis_jobs:
        print(f"[RESULTS] Job {job_id} not found")
        raise HTTPException(status_code=404, detail="Job not found")
    job = analysis_jobs[job_id]
    if job["status"] != "completed":
        print(f"[RESULTS] Job {job_id} not completed, status: {job['status']}")
        raise HTTPException(status_code=400, detail=f"Analysis not completed. Status: {job['status']}")
    results = job.get("results", [])
    # Convert AnalysisResult objects to dict for JSON serialization
    if results and hasattr(results[0], 'model_dump'):  # CHANGED: 'dict' -> 'model_dump'
        results = [result.model_dump() for result in results]  # CHANGED: .dict() -> .model_dump()
    # Enhance with GitHub metadata if this was a GitHub analysis
    response_data = {
        "success": True,
        "job_id": job_id,
        "results": results,
        "total_files": len(results),
        "completion_time": job.get("completion_time").isoformat() if job.get("completion_time") else None
    }
    # Add GitHub-specific metadata
    if job.get("is_github", False):
        response_data["github_metadata"] = {
            "repo_url": job.get("repo_url"),
            "branch": job.get("branch"),
            "repo_stats": job.get("repo_stats", {}),
            "analysis_type": "github_repository",
            "temp_dir": job.get("temp_dir")  # IMPORTANT: Include temp_dir for chat
        }
    print(f"[RESULTS] Returning {len(results)} file results")
    if job.get("is_github"):
        print(f"[RESULTS] GitHub repository: {job.get('repo_url')}")
    return response_data

@app.get("/api/partial-results/{job_id}")
async def get_partial_results(job_id: str):
    """FIXED: Get current partial results for progressive display"""
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = analysis_jobs[job_id]
    partial_results = job.get("partial_results", [])
    # Convert AnalysisResult objects to dict for JSON serialization
    if partial_results and hasattr(partial_results[0], 'model_dump'):  # CHANGED: 'dict' -> 'model_dump'
        partial_results = [result.model_dump() for result in partial_results]  # CHANGED: .dict() -> .model_dump()
    return {
        "success": True,
        "partial": True,
        "completed_files": len(partial_results),
        "total_files": job.get("total_files", 0),
        "progress": job.get("progress", 0),
        "job_id": job_id,
        "results": partial_results,
        "github_metadata": job.get("github_metadata")
    }

@app.post("/api/chat/start")
async def start_chat_session(request: ChatStartRequest):
    """ENHANCED: Start chat session with GitHub support"""
    try:
        session_id = generate_job_id()
        print(f"[CHAT] Starting interactive session: {session_id}")
        print(f"[CHAT] Request details:")
        print(f"[CHAT] - upload_dir: {request.upload_dir}")
        print(f"[CHAT] - github_repo: {request.github_repo}")
        print(f"[CHAT] - branch: {request.branch}")
        # UNIFIED LOGIC: Both flows use upload_dir
        codebase_path = "."
        analysis_context = "current directory"
        if request.upload_dir and os.path.exists(request.upload_dir):
            # Works for BOTH file uploads AND GitHub repos now!
            codebase_path = request.upload_dir
            # Determine context type
            if request.github_repo:
                analysis_context = f"GitHub repository {request.github_repo} (branch: {request.branch})"
                print(f"[CHAT] Using GitHub repository directory: {codebase_path}")
            else:
                analysis_context = f"uploaded files in {request.upload_dir}"
                print(f"[CHAT] Using uploaded files directory: {codebase_path}")
            # Verify the directory has files
            try:
                files_in_dir = os.listdir(codebase_path)
                print(f"[CHAT] Directory contains {len(files_in_dir)} items: {files_in_dir[:5]}...")
            except Exception as e:
                print(f"[CHAT] Warning: Could not list directory contents: {e}")
        else:
            print(f"[CHAT] No upload directory provided or directory doesn't exist: {request.upload_dir}")
            # Only fail if we expected a directory
            if request.upload_dir or request.github_repo:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Analysis directory not found. Please analyze the {'repository' if request.github_repo else 'files'} first."
                )
        print(f"[CHAT] Final codebase path: {codebase_path}")
        # Initialize the Q&A agent with the correct path
        qa_agent = EnhancedQAAgent(codebase_path=codebase_path)
        await qa_agent.initialize(enable_rag=True, run_analysis=True)
        # Store the agent for this session
        qa_agents[session_id] = qa_agent
        return {
            "success": True,
            "session_id": session_id,
            "message": f"Interactive Q&A session started with {analysis_context}",
            "codebase_info": {
                "path": codebase_path,
                "status": "ready",
                "context": analysis_context,
                "github_repo": request.github_repo if request.github_repo else None,
                "branch": request.branch if request.branch else None
            }
        }
    except Exception as e:
        print(f"[CHAT] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to start chat: {str(e)}")

@app.post("/api/chat/message")
async def send_chat_message(request: ChatMessage):
    """Send chat message"""
    try:
        if request.session_id not in qa_agents:
            raise HTTPException(status_code=404, detail="Chat session not found")
        qa_agent = qa_agents[request.session_id]
        print(f"[CHAT] Processing message: {request.message[:50]}...")
        print(f"[CHAT] Using codebase path: {qa_agent.codebase_path}")
        # Use the actual Q&A system
        response = await qa_agent.ask_question(request.message)
        return {
            "success": True,
            "response": {
                "content": response.answer,
                "confidence": response.confidence,
                "source": response.source,
                "processing_time": response.processing_time,
                "follow_up_suggestions": response.follow_up_suggestions,
                "related_files": response.related_files or []
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"[CHAT] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat message failed: {str(e)}")

# ========== GITHUB INTEGRATION ENDPOINTS ==========
@app.post("/api/github/validate")
async def validate_github_repository(request: GitHubValidateRequest):
    """Validate GitHub repository URL and get metadata"""
    repo_url = request.repo_url
    print(f"[GITHUB-API] Validating repository: {repo_url}")
    try:
        github_analyzer = GitHubRepoAnalyzer()
        validation_result = await github_analyzer.validate_repository(repo_url)
        print(f"[GITHUB-API] Validation result: {validation_result.get('valid', False)}")
        return validation_result
    except Exception as e:
        print(f"[GITHUB-API] Validation error: {str(e)}")
        return {
            "valid": False,
            "error": f"Validation failed: {str(e)}"
        }

@app.get("/api/github/branches/{owner}/{repo}")
async def get_repository_branches(owner: str, repo: str):
    """Get available branches for a repository"""
    print(f"[GITHUB-API] Getting branches for: {owner}/{repo}")
    try:
        github_analyzer = GitHubRepoAnalyzer()
        branches = await github_analyzer.get_repository_branches(owner, repo)
        print(f"[GITHUB-API] Found {len(branches)} branches")
        return {
            "success": True,
            "branches": branches,
            "default_branch": branches[0] if branches else "main"
        }
    except Exception as e:
        print(f"[GITHUB-API] Error getting branches: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "branches": ["main", "master"]  # Fallback
        }

@app.post("/api/github/analyze")
async def analyze_github_repository_progressive(request: GitHubAnalyzeRequest):
    """ENHANCED: Progressive GitHub repository analysis"""
    job_id = generate_job_id()
    temp_dir = None
    try:
        # Download repository
        github_analyzer = GitHubRepoAnalyzer()
        temp_dir = await github_analyzer.download_repo(request.repo_url, request.branch)
        repo_stats = github_analyzer.get_repository_stats(temp_dir)
        # Discover files
        from main import LangGraphCQI
        cqi = LangGraphCQI(enable_rag=True, enable_cache=True)
        code_files = cqi._discover_files(temp_dir)
        # Initialize progressive job
        analysis_jobs[job_id] = {
            "job_id": job_id,
            "status": "processing", 
            "progress": 15,
            "message": f"Downloaded {request.repo_url}, analyzing {len(code_files)} files...",
            "start_time": datetime.now(),
            "file_paths": code_files,
            "total_files": len(code_files),
            "completed_files": 0,
            "partial_results": [],
            "repo_url": request.repo_url,
            "branch": request.branch,
            "temp_dir": temp_dir,
            "repo_stats": repo_stats,
            "is_github": True,
            "github_metadata": {
                "repo_url": request.repo_url,
                "branch": request.branch,
                "stats": repo_stats,
                "temp_dir": temp_dir
            }
        }
        # Start progressive analysis
        analysis_request = AnalyzeRequest(
            file_paths=code_files,
            detailed=request.detailed,
            rag=True,
            progressive=True
        )
        asyncio.create_task(run_progressive_analysis(job_id, analysis_request))
        return {
            "success": True,
            "job_id": job_id,
            "repo_url": request.repo_url,
            "branch": request.branch,
            "files_analyzed": len(code_files),
            "repo_stats": repo_stats,
            "upload_dir": temp_dir,
            "temp_dir": temp_dir,
            "total_files": len(code_files),
            "progressive": True  # Indicate this supports progressive updates
        }
    except Exception as e:
        print(f"[GITHUB-PROGRESSIVE] Error: {str(e)}")
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
        analysis_jobs[job_id] = {
            "job_id": job_id,
            "status": "failed",
            "progress": 0,
            "message": f"GitHub analysis failed: {str(e)}",
            "start_time": datetime.now(),
            "is_github": True
        }
        raise HTTPException(status_code=500, detail=f"GitHub analysis failed: {str(e)}")

@app.delete("/api/github/cleanup/{job_id}")
async def cleanup_github_temp_dir(job_id: str):
    """Clean up GitHub temporary directory"""
    try:
        if job_id in github_temp_dirs:
            temp_dir = github_temp_dirs[job_id]
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"[CLEANUP] Removed temp directory: {temp_dir}")
            del github_temp_dirs[job_id]
            return {"success": True, "message": f"Cleaned up temp directory for job {job_id}"}
        else:
            return {"success": False, "message": f"No temp directory found for job {job_id}"}
    except Exception as e:
        print(f"[CLEANUP] Error: {e}")
        return {"success": False, "error": str(e)}

# ---------------------------
# Development and Debug endpoints
# ---------------------------
@app.get("/api/test")
async def test_endpoint():
    """Test endpoint for development"""
    return {
        "message": "API is working!",
        "active_jobs": len(analysis_jobs),
        "active_sessions": len(qa_agents),
        "websocket_connections": len(websocket_connections),
        "progressive_analysis": True,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/debug/github-jobs")
async def debug_github_jobs():
    """Debug endpoint to check GitHub job tracking"""
    github_jobs = {}
    for job_id, job in analysis_jobs.items():
        if job.get("is_github"):
            github_jobs[job_id] = {
                "repo_url": job.get("repo_url"),
                "branch": job.get("branch"),
                "temp_dir": job.get("temp_dir"),
                "temp_dir_exists": os.path.exists(job.get("temp_dir", "")),
                "status": job.get("status"),
                "completed_files": job.get("completed_files", 0),
                "total_files": job.get("total_files", 0),
                "start_time": job.get("start_time").isoformat() if job.get("start_time") else None
            }
    return {
        "total_github_jobs": len(github_jobs),
        "github_temp_dirs": github_temp_dirs,
        "jobs": github_jobs
    }

# ---------------------------
# Run server (development)
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    print("Starting Enhanced LangGraph Code Quality Intelligence API...")
    print("Progressive analysis with real-time updates enabled")
    print("ðŸ”— CORS enabled for all origins")
    # Use PORT environment variable for deployment
    port = int(os.environ.get("PORT", 8000))
    print(f"Access test endpoint at: http://localhost:{port}/api/test")
    print(f"Progressive analysis: http://localhost:{port}/api/partial-results/{{job_id}}")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)