#!/usr/bin/env python3
"""
Enhanced Interactive Q&A System for Codebase Analysis
Fixed version with comprehensive file context preservation
"""

import os
import sys
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.base_agent import GroqLLM, LanguageDetector
from agents.rag_agent import RAGCodeAnalyzer
from workflow.graph import LangGraphMultiAgentAnalyzer

@dataclass
class ConversationContext:
    """Store conversation context and memory"""
    codebase_path: str
    conversation_id: str
    start_time: datetime
    questions_asked: List[Dict[str, Any]]
    analysis_results: Optional[Dict[str, Any]] = None
    user_focus: Optional[str] = None
    last_files_discussed: List[str] = None
    conversation_summary: str = ""

@dataclass 
class QuestionResponse:
    """Structure for Q&A responses"""
    answer: str
    confidence: float
    source: str
    processing_time: float
    follow_up_suggestions: List[str]
    context_used: List[str]
    related_files: List[str] = None
    code_examples: List[str] = None

class EnhancedQAAgent:
    """Enhanced Q&A agent with comprehensive file context preservation"""
    
    def __init__(self, codebase_path: str):
        self.codebase_path = codebase_path
        self.llm = GroqLLM()
        self.rag_analyzer = None
        self.langgraph_analyzer = None
        self.conversation_context = None
        
        # Knowledge base from analysis
        self.codebase_facts = {}
        self.analysis_cache = {}
        
    async def initialize(self, enable_rag: bool = True, run_analysis: bool = True):
        """Initialize the enhanced Q&A system"""
        print("[QA] Enhanced Interactive Q&A System")
        print("=" * 50)
        print(f"[DIR] Codebase: {self.codebase_path}")
        
        # Initialize conversation context
        self.conversation_context = ConversationContext(
            codebase_path=self.codebase_path,
            conversation_id=f"qa_{int(time.time())}",
            start_time=datetime.now(),
            questions_asked=[],
            last_files_discussed=[]
        )
        
        # Discover and analyze codebase structure
        print("[SCAN] Analyzing codebase structure...")
        self.codebase_facts = await self._analyze_codebase_structure()
        print(f"[FILES] Found {len(self.codebase_facts['files'])} files")
        
        # Log discovered files for debugging
        print(f"[FILES] Discovered files:")
        for filename, info in self.codebase_facts.get('file_info', {}).items():
            print(f"  - {filename}: {info['language']}, {info['lines']} lines")
        
        # Initialize RAG system
        if enable_rag:
            print("[RAG] Setting up RAG system...")
            try:
                self.rag_analyzer = RAGCodeAnalyzer()
                language_detector = LanguageDetector()
                self.rag_analyzer.index_codebase(self.codebase_facts['files'], language_detector)
                print("[OK] RAG indexing complete")
            except Exception as e:
                print(f"[WARN] RAG setup failed: {e}")
                self.rag_analyzer = None
        
        # Run LangGraph analysis if requested
        if run_analysis and len(self.codebase_facts['files']) <= 10:
            print("[ANALYSIS] Running LangGraph analysis for enhanced context...")
            try:
                self.langgraph_analyzer = LangGraphMultiAgentAnalyzer(
                    enable_rag=enable_rag,
                    enable_cache=True
                )
                
                analysis_results = []
                for file_path in self.codebase_facts['files'][:5]:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            code_content = f.read()
                        language = LanguageDetector.detect_language(file_path)
                        
                        result = await self.langgraph_analyzer.analyze_file(
                            file_path=file_path,
                            code_content=code_content,
                            language=language
                        )
                        analysis_results.append(result)
                        
                    except Exception as e:
                        print(f"[WARN] Failed to analyze {file_path}: {e}")
                        continue
                
                self.conversation_context.analysis_results = {
                    'files_analyzed': len(analysis_results),
                    'total_issues': sum(r.get('total_issues', 0) for r in analysis_results),
                    'results': analysis_results
                }
                print(f"[OK] Analysis complete: {self.conversation_context.analysis_results['total_issues']} issues found")
                
            except Exception as e:
                print(f"[WARN] LangGraph analysis failed: {e}")
        
        print("[READY] Q&A system ready!")
        print()
    
    async def ask_question(self, question: str) -> QuestionResponse:
        """Process a question with conversational context"""
        start_time = time.time()
        
        # Add to conversation history
        self.conversation_context.questions_asked.append({
            'question': question,
            'timestamp': datetime.now(),
            'context': self._extract_question_context(question)
        })
        
        # Update user focus based on question patterns
        self._update_user_focus(question)
        
        # Enhanced Universal Question Handler with comprehensive file context
        response = await self._answer_universal_question(question)
        response.processing_time = time.time() - start_time
        return response
    
    async def _answer_universal_question(self, question: str) -> QuestionResponse:
        """Enhanced universal question handler with comprehensive file context"""
        print(f"[UNIVERSAL] Processing question: {question[:50]}...")
        
        # Check for file-specific questions first
        file_response = await self._answer_file_questions(question)
        if file_response:
            return file_response
        
        # Build comprehensive context for any question
        context_parts = []
        related_files = []
        confidence = 0.7
        source = "universal"
        
        # 1. FIRST: Add complete file inventory (before RAG chunks)
        facts = self.codebase_facts
        context_parts.append(f"\nCOMPLETE FILE INVENTORY:")
        context_parts.append(f"- Total Files: {facts['file_count']}")
        context_parts.append(f"- Languages: {', '.join(facts['languages'].keys())}")
        context_parts.append(f"- Total Lines: {facts['total_lines']}")
        
        # Add detailed file list with metadata
        if facts.get('file_info'):
            context_parts.append(f"\nDETAILED FILE LIST:")
            for filename, info in list(facts['file_info'].items())[:10]:  # Show first 10 files
                context_parts.append(f"  - {filename}: {info['language']}, {info['lines']} lines, {info['size']} bytes")
                related_files.append(filename)
            
            if len(facts['file_info']) > 10:
                context_parts.append(f"  ... and {len(facts['file_info']) - 10} more files")
        
        # 2. THEN: Add RAG results (with preserved file context)
        if self.rag_analyzer:
            try:
                similar_chunks = self.rag_analyzer.vector_store.search_similar(
                    query=question, 
                    top_k=6  # Reduce chunks to leave room for file context
                )
                
                if similar_chunks and similar_chunks[0][1] > 0.2:
                    rag_context = []
                    files_found = set()
                    
                    for chunk, similarity in similar_chunks:
                        if similarity > 0.2:
                            filename = os.path.basename(chunk.file_path)
                            files_found.add(filename)
                            rag_context.append({
                                'file': filename,
                                'content': chunk.content[:400],  # Smaller chunks
                                'similarity': similarity,
                                'full_path': chunk.file_path
                            })
                    
                    if rag_context:
                        context_parts.append(f"\nRELEVANT CODE CHUNKS FROM YOUR FILES:")
                        for i, chunk in enumerate(rag_context, 1):
                            context_parts.append(f"\n{i}. From '{chunk['file']}' (similarity: {chunk['similarity']:.2f}):")
                            context_parts.append(f"   Path: {chunk['full_path']}")
                            context_parts.append(f"```\n{chunk['content']}\n```")
                        
                        # Update related files with RAG findings
                        related_files.extend(list(files_found))
                        confidence = min(0.9, 0.5 + max(c['similarity'] for c in rag_context))
                        source = "rag_enhanced"
                        
            except Exception as e:
                print(f"[WARN] RAG search failed: {e}")
        
        # 3. Add analysis results if available
        if self.conversation_context.analysis_results:
            analysis = self.conversation_context.analysis_results
            context_parts.append(f"\nANALYSIS RESULTS SUMMARY:")
            context_parts.append(f"- Files Analyzed: {analysis['files_analyzed']}")
            context_parts.append(f"- Total Issues: {analysis['total_issues']}")
            
            # Add file-specific analysis results
            for result in analysis.get('results', [])[:5]:  # First 5 analyzed files
                file_name = os.path.basename(result.get('file_path', 'unknown'))
                issues = result.get('total_issues', 0)
                context_parts.append(f"  - {file_name}: {issues} issues found")
        
        # 4. Add conversation context
        conversation_summary = self._get_conversation_summary()
        
        # 5. Build the enhanced prompt
        context_text = "\n".join(context_parts)
        
        prompt = f"""You are an expert code analyst with comprehensive knowledge of this codebase. Answer the user's question using the provided context.

CODEBASE: {self.codebase_path}

{context_text}

CONVERSATION HISTORY:
{conversation_summary}

USER QUESTION: {question}

FORMATTING INSTRUCTIONS:
- Use clear headers with ## for main sections, ### for subsections
- Use **bold text** for important terms, file names, and key concepts
- Use bullet points with - for lists
- Use numbered lists 1. 2. 3. for sequential steps
- Use `code` for inline code, function names, and technical terms
- Use ```language code blocks for multi-line code examples
- Add proper spacing between sections
- Keep paragraphs concise and readable

IMPORTANT INSTRUCTIONS:
- You have access to COMPLETE file information including names, counts, and paths
- When discussing files, always mention specific filenames from the FILE LIST above
- If asked about file counts or names, refer to the COMPLETE FILE INVENTORY section
- Use the RELEVANT CODE CHUNKS to provide specific code examples with their source files
- Be specific and reference actual files by name
- Include practical suggestions with file-specific context

Answer the question comprehensively using both the complete file inventory and the relevant code chunks:"""
        
        try:
            response_text = await self.llm.generate(prompt, max_tokens=1000)
            follow_ups = self._extract_follow_ups(response_text)
            
            return QuestionResponse(
                answer=response_text,
                confidence=confidence,
                source=source,
                processing_time=0,
                follow_up_suggestions=follow_ups,
                context_used=["complete_inventory", "rag", "analysis", "conversation"],
                related_files=list(set(related_files)),  # Remove duplicates
                code_examples=[chunk['content'] for chunk in (rag_context[:2] if 'rag_context' in locals() else [])]
            )
            
        except Exception as e:
            print(f"[ERROR] Universal handler failed: {e}")
            return self._create_fallback_response(question, str(e))
    
    async def _answer_file_questions(self, question: str) -> Optional[QuestionResponse]:
        """Handle questions specifically about files, counts, and names"""
        question_lower = question.lower()
        
        # Check if this is asking about files, counts, or names
        if any(phrase in question_lower for phrase in [
            'how many files', 'file count', 'list files', 'what files', 
            'file names', 'show files', 'which files', 'files are', 'files do'
        ]):
            facts = self.codebase_facts
            
            # Build comprehensive file response
            answer_parts = []
            
            # File count and summary
            answer_parts.append(f"## File Analysis Summary")
            answer_parts.append(f"Your codebase contains **{facts['file_count']} files** with **{facts['total_lines']:,} total lines** of code.")
            
            # Language breakdown
            answer_parts.append(f"\n### Programming Languages:")
            for lang, count in facts['languages'].items():
                answer_parts.append(f"- **{lang.title()}**: {count} files")
            
            # Detailed file list
            answer_parts.append(f"\n### Complete File List:")
            if facts.get('file_info'):
                for filename, info in facts['file_info'].items():
                    answer_parts.append(f"- **{filename}** ({info['language']}, {info['lines']} lines, {info['size']:,} bytes)")
            
            # Analysis results if available
            if self.conversation_context.analysis_results:
                analysis = self.conversation_context.analysis_results
                answer_parts.append(f"\n### Analysis Status:")
                answer_parts.append(f"- **Files Analyzed**: {analysis['files_analyzed']}")
                answer_parts.append(f"- **Issues Found**: {analysis['total_issues']}")
            
            answer = "\n".join(answer_parts)
            
            return QuestionResponse(
                answer=answer,
                confidence=1.0,
                source="file_inventory",
                processing_time=0,
                follow_up_suggestions=[
                    "Which file has the most issues?",
                    "Show me details about a specific file",
                    "What are the main problems in these files?"
                ],
                context_used=["file_discovery", "analysis_results"],
                related_files=list(facts.get('file_info', {}).keys())
            )
        
        return None
    
    async def _analyze_codebase_structure(self) -> Dict[str, Any]:
        """Analyze codebase structure and build comprehensive knowledge base"""
        if not os.path.exists(self.codebase_path):
            raise FileNotFoundError(f"Path not found: {self.codebase_path}")
        
        # Discover files
        files = self._discover_files()
        
        # Analyze languages and build file info
        languages = {}
        total_lines = 0
        file_info = {}
        
        for file_path in files:
            try:
                # Get file info
                filename = os.path.basename(file_path)
                size = os.path.getsize(file_path)
                language = LanguageDetector.detect_language(file_path)
                
                # Count lines
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = len(f.readlines())
                    total_lines += lines
                
                languages[language] = languages.get(language, 0) + 1
                
                file_info[filename] = {
                    'path': file_path,
                    'size': size,
                    'lines': lines,
                    'language': language
                }
                
            except Exception as e:
                print(f"⚠️ Error analyzing {file_path}: {e}")
                continue
        
        return {
            'path': self.codebase_path,
            'files': files,
            'file_count': len(files),
            'languages': languages,
            'total_lines': total_lines,
            'file_info': file_info,
            'analyzed_at': datetime.now()
        }
    
    def _discover_files(self) -> List[str]:
        """Discover code files in the codebase"""
        extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', 
                     '.cs', '.go', '.rs', '.php', '.html', '.css', '.sql', '.sh'}
        files = []
        
        if os.path.isfile(self.codebase_path):
            return [self.codebase_path]
        
        # Handle directory - check if it exists
        if not os.path.exists(self.codebase_path):
            print(f"[WARNING] Path does not exist: {self.codebase_path}")
            return []
        
        ignore_dirs = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 
                      '.cache', 'build', 'dist', '.analysis_cache'}
        
        try:
            for root, dirs, filenames in os.walk(self.codebase_path):
                # Filter out ignored directories
                dirs[:] = [d for d in dirs if d not in ignore_dirs]
                
                for filename in filenames:
                    if Path(filename).suffix.lower() in extensions:
                        full_path = os.path.join(root, filename)
                        files.append(full_path)
            
            print(f"[DISCOVER] Found {len(files)} code files in {self.codebase_path}")
            return files[:50]  # Reasonable limit
            
        except Exception as e:
            print(f"[ERROR] Failed to discover files: {e}")
            return []
    
    def _extract_question_context(self, question: str) -> Dict[str, Any]:
        """Extract context clues from the question"""
        question_lower = question.lower()
        
        context = {
            'type': 'general',
            'mentions_files': [],
            'mentions_languages': [],
            'intent': 'info'
        }
        
        # Extract mentioned files
        for filename in self.codebase_facts.get('file_info', {}):
            if filename.lower() in question_lower:
                context['mentions_files'].append(filename)
        
        # Extract mentioned languages
        for language in self.codebase_facts.get('languages', {}):
            if language.lower() in question_lower:
                context['mentions_languages'].append(language)
        
        # Determine intent
        if any(word in question_lower for word in ['security', 'vulnerability', 'exploit', 'attack']):
            context['intent'] = 'security'
        elif any(word in question_lower for word in ['performance', 'slow', 'optimize', 'speed']):
            context['intent'] = 'performance'
        elif any(word in question_lower for word in ['bug', 'error', 'fix', 'debug', 'problem']):
            context['intent'] = 'debug'
        elif any(word in question_lower for word in ['how', 'what does', 'explain', 'understand']):
            context['intent'] = 'explanation'
        elif any(word in question_lower for word in ['show', 'code', 'function', 'class']):
            context['intent'] = 'code'
        elif any(word in question_lower for word in ['files', 'count', 'list', 'names']):
            context['intent'] = 'files'
        
        return context
    
    def _update_user_focus(self, question: str):
        """Update understanding of what the user is most interested in"""
        context = self._extract_question_context(question)
        
        # Track user's primary interests
        if not hasattr(self.conversation_context, 'interests'):
            self.conversation_context.interests = {}
        
        intent = context['intent']
        self.conversation_context.interests[intent] = self.conversation_context.interests.get(intent, 0) + 1
        
        # Update current focus
        most_common = max(self.conversation_context.interests.items(), key=lambda x: x[1])
        self.conversation_context.user_focus = most_common[0]
    
    def _get_conversation_summary(self) -> str:
        """Get a summary of the conversation so far"""
        if not self.conversation_context.questions_asked:
            return "This is the start of our conversation."
        
        recent_questions = self.conversation_context.questions_asked[-3:]
        summary = f"Previous questions asked: "
        
        for i, q_info in enumerate(recent_questions, 1):
            summary += f"{i}. {q_info['question'][:50]}... "
        
        if self.conversation_context.user_focus:
            summary += f"\nUser seems interested in: {self.conversation_context.user_focus}"
        
        return summary
    
    def _extract_follow_ups(self, response: str) -> List[str]:
        """Extract follow-up questions from LLM response"""
        lines = response.split('\n')
        follow_ups = []
        
        for line in lines:
            line = line.strip()
            if line.endswith('?') and len(line) > 10 and len(line) < 100:
                # Remove common prefixes
                for prefix in ['Follow-up:', 'You could ask:', 'Next:', 'Also:', '-', '•', '*']:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                follow_ups.append(line)
        
        return follow_ups[:3]
    
    def _create_fallback_response(self, question: str, error: str) -> QuestionResponse:
        """Create fallback response for errors"""
        return QuestionResponse(
            answer=f"I apologize, but I encountered an error while processing your question about the codebase. The error was: {str(error)}. Please try rephrasing your question or ask about something more specific.",
            confidence=0.3,
            source="error_fallback",
            processing_time=0,
            follow_up_suggestions=[
                "What files are in this codebase?",
                "Are there any security issues?", 
                "Can you summarize the main files?"
            ],
            context_used=["error_handling"],
            related_files=[]
        )

class InteractiveQASession:
    """Main interactive Q&A session manager"""
    
    def __init__(self, codebase_path: str, enable_rag: bool = True, run_analysis: bool = True):
        self.codebase_path = codebase_path
        self.enable_rag = enable_rag
        self.run_analysis = run_analysis
        self.qa_agent = None
        
    async def initialize(self):
        """Initialize the Q&A session"""
        self.qa_agent = EnhancedQAAgent(self.codebase_path)
        await self.qa_agent.initialize(
            enable_rag=self.enable_rag,
            run_analysis=self.run_analysis
        )
    
    async def start_session(self):
        """Start the interactive session"""
        print("[CHAT] **Interactive Q&A Session Started**")
        print("Ask me anything about your codebase!")
        print("Type 'help' for examples, 'exit' to quit")
        print()
        
        # Check if we're in an interactive environment
        if not self._is_interactive_environment():
            print("[INFO] Running in non-interactive mode. Running demo questions...")
            await self._run_demo_questions()
            return
        
        while True:
            try:
                try:
                    question = input("[Q] Your question: ").strip()
                except EOFError:
                    print("\n[EOF] End of input detected. Running demo questions...")
                    await self._run_demo_questions()
                    break
                except KeyboardInterrupt:
                    print("\n[BYE] Session ended by user. Goodbye!")
                    break
                
                if not question:
                    continue
                
                # Handle special commands
                if question.lower() in ['exit', 'quit', 'q']:
                    await self._end_session()
                    break
                
                if question.lower() in ['help', '?']:
                    self._show_help()
                    continue
                
                # Process question
                print("[THINKING] *Processing...*", end="", flush=True)
                response = await self.qa_agent.ask_question(question)
                print("\r" + " " * 25 + "\r", end="")
                
                # Display response
                self._display_response(response)
                print()
                
            except KeyboardInterrupt:
                print("\n[BYE] Session ended. Goodbye!")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
                print("Please try rephrasing your question.")
                
    def _is_interactive_environment(self) -> bool:
        """Check if we're in an interactive environment"""
        import sys
        return sys.stdin.isatty() and hasattr(sys.stdin, 'readline')
    
    async def _run_demo_questions(self):
        """Run demo questions when interactive input is not available"""
        demo_questions = [
            "How many files are in this codebase?",
            "What programming languages are used?",
            "Find all security vulnerabilities",
            "What are the main files I should look at?"
        ]
        
        print("\n[DEMO] Running sample questions:")
        print("=" * 60)
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n[Q{i}] {question}")
            print("-" * 40)
            
            try:
                response = await self.qa_agent.ask_question(question)
                self._display_response(response)
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"[ERROR] Failed to answer question: {e}")
        
        print(f"\n[DEMO] Demo completed!")
    
    async def _end_session(self):
        """End the Q&A session"""
        context = self.qa_agent.conversation_context
        print(f"\n[SESSION] Session Summary:")
        print(f"- Duration: {datetime.now() - context.start_time}")
        print(f"- Questions asked: {len(context.questions_asked)}")
        if context.analysis_results:
            print(f"- Issues analyzed: {context.analysis_results['total_issues']}")
        print("\n[BYE] Thank you for using the Q&A system! Goodbye!")
    
    def _display_response(self, response: QuestionResponse):
        """Display the response in a nice format"""
        print("[A] **Answer:**")
        print(f"{response.answer}")
        
        print(f"\n[STATS] Confidence: {response.confidence:.1%} | Source: {response.source} | Time: {response.processing_time:.1f}s")
        
        if response.related_files:
            print(f"[FILES] Related files: {', '.join(response.related_files[:3])}")
        
        if response.follow_up_suggestions:
            print("\n[FOLLOW-UP] Questions you might ask:")
            for i, suggestion in enumerate(response.follow_up_suggestions, 1):
                print(f"   {i}. {suggestion}")
    
    def _show_help(self):
        """Show help information"""
        print("\n[HELP] **What you can ask me:**")
        print("**Files & Structure:**")
        print("  - How many files are there?")
        print("  - What programming languages are used?")
        print("  - List all files in the codebase")
        print()
        print("**Code Analysis:**")
        print("  - Are there any security issues?")
        print("  - What performance problems exist?")
        print("  - How complex is this code?")
        print("  - What does [filename] do?")
        print()
        print("**Commands:**")
        print("  - help - Show this help")
        print("  - exit - End session")
        print()

async def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python interactive_qa.py <codebase_path>")
        print("Example: python interactive_qa.py C:/Users/haris/Downloads/democode")
        return
    
    codebase_path = sys.argv[1]
    
    if not os.path.exists(codebase_path):
        print(f"⚠️ Path not found: {codebase_path}")
        return
    
    # Parse options
    enable_rag = "--no-rag" not in sys.argv
    run_analysis = "--no-analysis" not in sys.argv
    
    session = InteractiveQASession(codebase_path, enable_rag, run_analysis)
    
    try:
        await session.initialize()
        await session.start_session()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())