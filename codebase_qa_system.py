#!/usr/bin/env python3
"""
Standalone Modular Codebase Q&A System
Separate from the main LLM Multi-Agent Analyzer
"""

import os
import sys
import time
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.base_agent import GroqLLM, LanguageDetector
from agents.rag_agent import RAGCodeAnalyzer

@dataclass
class CodebaseInfo:
    """Store factual information about the codebase"""
    path: str
    files: List[str]
    languages: Dict[str, int]  # language -> file count
    total_lines: int
    file_sizes: Dict[str, int]  # filename -> size in bytes

class CodebaseAnalyzer:
    """Analyzes codebase structure and provides factual information"""
    
    def __init__(self, codebase_path: str):
        self.codebase_path = codebase_path
        self.info: Optional[CodebaseInfo] = None
        self.language_detector = LanguageDetector()
        
    def analyze_codebase_structure(self) -> CodebaseInfo:
        """Analyze the codebase structure and return factual information"""
        if not os.path.exists(self.codebase_path):
            raise FileNotFoundError(f"Path not found: {self.codebase_path}")
        
        files = self._discover_code_files()
        languages = self._analyze_languages(files)
        total_lines = self._count_total_lines(files)
        file_sizes = self._get_file_sizes(files)
        
        self.info = CodebaseInfo(
            path=self.codebase_path,
            files=files,
            languages=languages,
            total_lines=total_lines,
            file_sizes=file_sizes
        )
        
        return self.info
    
    def _discover_code_files(self) -> List[str]:
        """Discover actual code files in the codebase"""
        code_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.cs', '.go', '.rs', '.php', '.html', '.css'}
        files = []
        
        if os.path.isfile(self.codebase_path):
            return [self.codebase_path]
        
        try:
            for root, _, filenames in os.walk(self.codebase_path):
                # Skip hidden directories and common ignore patterns
                if any(ignore in root for ignore in ['.git', 'node_modules', '__pycache__', '.venv', 'venv', '.cache']):
                    continue
                    
                for filename in filenames:
                    if Path(filename).suffix.lower() in code_extensions:
                        full_path = os.path.join(root, filename)
                        files.append(full_path)
            
            return files[:100]  # Reasonable limit
            
        except PermissionError:
            print(f"[WARNING] Permission denied accessing: {self.codebase_path}")
            return []
    
    def _analyze_languages(self, files: List[str]) -> Dict[str, int]:
        """Count files by programming language"""
        languages = {}
        for file_path in files:
            language = self.language_detector.detect_language(file_path)
            languages[language] = languages.get(language, 0) + 1
        return languages
    
    def _count_total_lines(self, files: List[str]) -> int:
        """Count total lines of code"""
        total = 0
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    total += len(f.readlines())
            except:
                continue
        return total
    
    def _get_file_sizes(self, files: List[str]) -> Dict[str, int]:
        """Get file sizes in bytes"""
        sizes = {}
        for file_path in files:
            try:
                sizes[os.path.basename(file_path)] = os.path.getsize(file_path)
            except:
                continue
        return sizes

class FactualQAAgent:
    """Q&A agent that provides factual, verified answers about codebases"""
    
    def __init__(self, codebase_analyzer: CodebaseAnalyzer, rag_analyzer: Optional[RAGCodeAnalyzer] = None):
        self.codebase_analyzer = codebase_analyzer
        self.rag_analyzer = rag_analyzer
        self.llm = GroqLLM()
        self.conversation_history = []
    
    async def ask_question(self, question: str) -> Dict[str, Any]:
        """Answer questions with verified factual information"""
        start_time = time.time()
        
        # First check if it's a factual question we can answer directly
        factual_answer = self._answer_factual_question(question)
        if factual_answer:
            return {
                'answer': factual_answer['answer'],
                'confidence': factual_answer['confidence'],
                'source': 'factual_analysis',
                'processing_time': time.time() - start_time,
                'follow_up_suggestions': factual_answer.get('follow_ups', [])
            }
        
        # Otherwise use RAG + LLM with factual constraints
        return await self._answer_with_llm(question, start_time)
    
    def _answer_factual_question(self, question: str) -> Optional[Dict[str, Any]]:
        """Answer purely factual questions without LLM hallucination"""
        if not self.codebase_analyzer.info:
            self.codebase_analyzer.analyze_codebase_structure()
        
        info = self.codebase_analyzer.info
        question_lower = question.lower()
        
        # File count questions
        if any(phrase in question_lower for phrase in ['how many files', 'file count', 'number of files']):
            file_list = [os.path.basename(f) for f in info.files]
            return {
                'answer': f"I can see exactly {len(info.files)} files in the codebase: {', '.join(file_list)}",
                'confidence': 1.0,
                'follow_ups': ['What programming languages are used?', 'What is the largest file?', 'Show me file sizes']
            }
        
        # Language questions
        if any(phrase in question_lower for phrase in ['programming language', 'what language', 'languages used']):
            lang_summary = []
            for lang, count in info.languages.items():
                lang_name = lang.title() if lang != 'unknown' else 'Unknown'
                lang_summary.append(f"{lang_name} ({count} file{'s' if count > 1 else ''})")
            
            return {
                'answer': f"The codebase uses these programming languages: {', '.join(lang_summary)}",
                'confidence': 1.0,
                'follow_ups': ['What does each file do?', 'Are there any security issues?']
            }
        
        # File size questions
        if any(phrase in question_lower for phrase in ['file size', 'largest file', 'biggest file']):
            largest = max(info.file_sizes.items(), key=lambda x: x[1])
            size_kb = largest[1] / 1024
            return {
                'answer': f"The largest file is {largest[0]} at {size_kb:.1f} KB. Total files: {len(info.files)}",
                'confidence': 1.0,
                'follow_ups': ['What does this file contain?', 'Show me all file sizes']
            }
        
        # Line count questions
        if any(phrase in question_lower for phrase in ['how many lines', 'line count', 'total lines']):
            return {
                'answer': f"The codebase has {info.total_lines} total lines of code across {len(info.files)} files",
                'confidence': 1.0,
                'follow_ups': ['Which file has the most lines?', 'What is the code structure?']
            }
        
        return None  # Not a factual question, needs LLM
    
    async def _answer_with_llm(self, question: str, start_time: float) -> Dict[str, Any]:
        """Answer complex questions using LLM with factual constraints"""
        try:
            # Get factual context
            info = self.codebase_analyzer.info
            factual_context = self._build_factual_context(info)
            
            # Get code context from RAG if available
            code_context = ""
            if self.rag_analyzer:
                try:
                    similar_chunks = self.rag_analyzer.vector_store.search_similar(
                        query=question,
                        top_k=3
                    )
                    
                    if similar_chunks:
                        code_context = "\nRelevant code snippets:\n"
                        for i, (chunk, similarity) in enumerate(similar_chunks, 1):
                            if similarity > 0.2:
                                code_context += f"{i}. From {os.path.basename(chunk.file_path)}:\n{chunk.content[:300]}...\n\n"
                        
                except Exception:
                    pass  # Continue without RAG context
            
            # Create constrained prompt
            prompt = f"""You are a code analyst. Answer the question using ONLY the factual information provided. Do not make assumptions or add information not present in the context.

FACTUAL CODEBASE INFORMATION:
{factual_context}

{code_context}

Question: {question}

Provide a precise answer based only on the provided information. If you cannot answer with certainty, say so.

Format:
ANSWER: [Your factual answer]
CONFIDENCE: [0.0-1.0]
FOLLOW_UPS: [2 relevant questions, separated by |]
"""
            
            response = await self.llm.generate(prompt, max_tokens=500)
            structured_response = self._parse_response(response)
            
            return {
                'answer': structured_response['answer'],
                'confidence': structured_response['confidence'],
                'source': 'llm_with_constraints',
                'processing_time': time.time() - start_time,
                'follow_up_suggestions': structured_response.get('follow_ups', [])
            }
            
        except Exception as e:
            return {
                'answer': f"I encountered an error while analyzing your question: {str(e)}",
                'confidence': 0.1,
                'source': 'error',
                'processing_time': time.time() - start_time,
                'follow_up_suggestions': ['Could you rephrase your question?']
            }
    
    def _build_factual_context(self, info: CodebaseInfo) -> str:
        """Build factual context string"""
        file_list = [os.path.basename(f) for f in info.files]
        
        context = f"""Codebase Path: {info.path}
Total Files: {len(info.files)}
File Names: {', '.join(file_list)}
Total Lines: {info.total_lines}
Languages: {dict(info.languages)}
"""
        return context
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse structured LLM response"""
        try:
            lines = response.strip().split('\n')
            answer = ""
            confidence = 0.8
            follow_ups = []
            
            for line in lines:
                if line.startswith('ANSWER:'):
                    answer = line.replace('ANSWER:', '').strip()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.replace('CONFIDENCE:', '').strip())
                    except:
                        confidence = 0.8
                elif line.startswith('FOLLOW_UPS:'):
                    follow_up_text = line.replace('FOLLOW_UPS:', '').strip()
                    follow_ups = [q.strip() for q in follow_up_text.split('|') if q.strip()]
            
            if not answer:
                answer = response.strip()
            
            return {
                'answer': answer,
                'confidence': max(0.0, min(1.0, confidence)),
                'follow_ups': follow_ups[:2]
            }
            
        except Exception:
            return {
                'answer': response.strip(),
                'confidence': 0.7,
                'follow_ups': []
            }

class ModularQASession:
    """Modular Q&A session separate from main analyzer"""
    
    def __init__(self, codebase_path: str, enable_rag: bool = True):
        self.codebase_path = codebase_path
        self.enable_rag = enable_rag
        self.codebase_analyzer = CodebaseAnalyzer(codebase_path)
        self.rag_analyzer = None
        self.qa_agent = None
    
    async def initialize(self):
        """Initialize the modular Q&A system"""
        print("[QA] MODULAR CODEBASE Q&A SYSTEM")
        print("=" * 50)
        print(f"[DIR] Analyzing: {self.codebase_path}")
        
        # Analyze codebase structure
        info = self.codebase_analyzer.analyze_codebase_structure()
        print(f"[FILES] Found {len(info.files)} files")
        print(f"[LANGS] Languages: {list(info.languages.keys())}")
        
        # Initialize RAG if enabled
        if self.enable_rag and len(info.files) > 0:
            print("[RAG] Initializing RAG system...")
            try:
                self.rag_analyzer = RAGCodeAnalyzer()
                language_detector = LanguageDetector()
                self.rag_analyzer.index_codebase(info.files, language_detector)
                print("[OK] RAG indexing complete")
            except Exception as e:
                print(f"[WARNING] RAG initialization failed: {e}")
                self.rag_analyzer = None
        
        # Initialize Q&A agent
        self.qa_agent = FactualQAAgent(self.codebase_analyzer, self.rag_analyzer)
        print("[READY] Q&A system initialized")
        print()
    
    async def start_interactive_session(self):
        """Start interactive Q&A session"""
        print("[TIP] Ask questions about your codebase (type 'exit' to quit)")
        print("Examples: 'How many files?', 'What languages are used?', 'What does app.py do?'")
        print()
        
        while True:
            try:
                question = input("[Q] Ask about your codebase: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("[BYE] Goodbye!")
                    break
                
                if question.lower() in ['help', '?']:
                    self._show_help()
                    continue
                
                # Process question
                print("[THINKING]...", end="", flush=True)
                response = await self.qa_agent.ask_question(question)
                print("\r" + " " * 20 + "\r", end="")
                
                # Display answer
                print(f"[A] {response['answer']}")
                print(f"[CONFIDENCE] {response['confidence']:.1%} | [SOURCE] {response['source']} | [TIME] {response['processing_time']:.1f}s")
                
                if response.get('follow_up_suggestions'):
                    print("[FOLLOW-UPS]")
                    for i, suggestion in enumerate(response['follow_up_suggestions'], 1):
                        print(f"  {i}. {suggestion}")
                
                print()
                
            except KeyboardInterrupt:
                print("\n[BYE] Goodbye!")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
    
    def _show_help(self):
        """Show help information"""
        print("\n[HELP] AVAILABLE QUESTIONS:")
        print("  • How many files do you see?")
        print("  • What programming languages are used?")
        print("  • What does [filename] do?")
        print("  • Are there any security issues?")
        print("  • What is the largest file?")
        print("  • Show me the project structure")
        print("\n[COMMANDS]")
        print("  help, ?    - Show this help")
        print("  exit, quit - Exit the session")
        print()

async def main():
    """Main entry point for modular Q&A system"""
    if len(sys.argv) != 2:
        print("Usage: python codebase_qa_system.py <codebase_path>")
        print("Example: python codebase_qa_system.py C:/Users/haris/Downloads/democode")
        sys.exit(1)
    
    codebase_path = sys.argv[1]
    
    if not os.path.exists(codebase_path):
        print(f"[ERROR] Path not found: {codebase_path}")
        sys.exit(1)
    
    # Create and start modular Q&A session
    session = ModularQASession(codebase_path, enable_rag=True)
    
    try:
        await session.initialize()
        await session.start_interactive_session()
    except KeyboardInterrupt:
        print("\n[BYE] Goodbye!")
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())