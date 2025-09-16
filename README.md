# 🤖 Code Quality Intelligence (CQI)

> **LangGraph Multi-Agent Code Analysis System** - AI-powered code quality assessment with intelligent workflow orchestration

An advanced code analysis platform that uses specialized AI agents working together through LangGraph workflows to provide comprehensive code quality insights.

## 🚀 Quick Setup

### Prerequisites

You need Python 3.10+ and an internet connection. That's it!

### 1. Install uv (Ultra-fast Python Package Manager)

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (using pip):**
```bash
pip install uv
```

### 2. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd cqi

# Create virtual environment and install dependencies
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install PyTorch CPU-only first (prevents dependency conflicts)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Then install remaining dependencies
uv pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file with your API keys:

```bash
# Required: Groq API key for LLM inference
GROQ_API_KEY=your_groq_api_key_here

# Optional: For GitHub repository analysis
GITHUB_TOKEN=your_github_token_here
```

**Get your Groq API key:** Visit [console.groq.com](https://console.groq.com) and create a free account.

## 🎯 Usage

### Using CLI Tool

#### Basic Analysis Commands

```bash
# Analyze local file or directory
python atlan-code-analyze.py --path="/path/to/code"

# Detailed analysis with comprehensive reports
python atlan-code-analyze.py --path="/path/to/code" --detailed

# Analyze GitHub repository
python atlan-code-analyze.py --repo="https://github.com/user/repo"

# Analyze specific branch
python atlan-code-analyze.py --repo="https://github.com/user/repo" --branch="develop"
```

#### Interactive Q&A Mode

```bash
# Start Q&A session for local code
python atlan-code-analyze.py --qa --path="/path/to/code"

# Interactive mode with GitHub repository
python atlan-code-analyze.py --qa --repo="https://github.com/user/repo"
```

#### Advanced Options

```bash
# Use specific agents only
python atlan-code-analyze.py --path="." --agents="security,performance,complexity"

# All available agents: security,performance,complexity,documentation,testing,duplication
python atlan-code-analyze.py --path="." --agents="security,performance,complexity,documentation,testing,duplication"

# Limit number of files analyzed
python atlan-code-analyze.py --path="." --max-files=50

# Disable RAG system for small codebases
python atlan-code-analyze.py --path="." --no-rag

# Disable caching for fresh analysis
python atlan-code-analyze.py --path="." --no-cache

# Combined advanced analysis
python atlan-code-analyze.py --path="." --agents="security,performance" --detailed --max-files=100
```

#### CLI Help & Version

```bash
# Show help and all available options
python atlan-code-analyze.py --help

# Show version
python atlan-code-analyze.py --version
```

### Basic Analysis (Alternative if CLI tool doesn't work)

```bash
# Analyze a single file
python main.py analyze <file_path>
python main.py analyze app.py

# Analyze entire directory (. = current directory)
python main.py analyze <directory_path>
python main.py analyze .

# Detailed analysis with top 20 issues
python main.py analyze <path> --detailed
python main.py analyze . --detailed
```

### Advanced Options

```bash
# Specific agents only
python main.py analyze <path> --agents security,performance
python main.py analyze . --agents security,performance

# Enable RAG for large codebases
python main.py analyze <path> --rag
python main.py analyze . --rag

# Limit analysis scope
python main.py analyze <path> --max-files 20 --parallel 2
python main.py analyze . --max-files 20 --parallel 2

# Legacy mode (without LangGraph)
python main.py analyze <path> --legacy
python main.py analyze . --legacy
```

### Interactive Q&A Mode

```bash
# Start interactive session
python main.py interactive <codebase_path>
python main.py interactive .

# Example queries:
# "What are the main security vulnerabilities?"
# "Show me the most complex functions"
# "Which files need better documentation?"
```


## 🤖 Available Agents

| Agent | Specialization | Key Features |
|-------|----------------|--------------|
| 🛡️ **Security** | Vulnerability Detection | SQL injection, XSS, hardcoded secrets, insecure practices |
| 🔧 **Complexity** | Code Structure | Cyclomatic complexity, SOLID principles, maintainability |
| ⚡ **Performance** | Optimization | Algorithm efficiency, bottlenecks, resource usage |
| 📚 **Documentation** | Code Documentation | Missing docstrings, API docs, code comments |


## 📊 Features

### 🧠 LangGraph Workflow Engine
- **Smart Agent Orchestration**: Dependency-aware execution
- **State Management**: Cross-agent insight sharing
- **Dynamic Routing**: Conditional workflow paths
- **Error Recovery**: Built-in retry mechanisms

### 🚀 Analysis Capabilities
- **Multi-Language Support**: Python, JavaScript, TypeScript, Java, C++, Go, Rust
- **RAG Integration**: Automatic activation for large codebases (>8K tokens)
- **Intelligent Caching**: Faster subsequent analyses
- **Parallel Processing**: Configurable concurrent analysis

### 📈 Output & Reporting
- **Severity-Based Grouping**: Critical, High, Medium, Low
- **Agent Performance Metrics**: Processing time, confidence scores
- **Detailed Issue Reports**: Line-by-line findings with suggestions
- **Language Statistics**: Multi-language project insights

## 🔧 Configuration Options

```bash
# View workflow architecture
python main.py workflow

# List available agents
python main.py agents

# All available options
python main.py --help
```

## 🐳 Docker Support

```bash
# Build and run with Docker
docker build -t cqi .
docker run -v $(pwd):/app -e GROQ_API_KEY=your_key cqi
# Note: Docker runs the FastAPI backend by default
```

## 🌐 Web API Mode

```bash
# Start FastAPI server
uvicorn api_backend:app --host 0.0.0.0 --port 8000

# API endpoints available at http://localhost:8000/docs
```

## 📁 Project Structure

```
cqi/
├── main.py                 # Main CLI entry point
├── atlan-code-analyze.py   # Standalone CLI tool
├── api_backend.py          # FastAPI web service
├── agents/                 # AI agent implementations
├── workflow/               # LangGraph workflow definitions
├── requirements.txt        # Dependencies
└── .env                    # Configuration (create this)
```

## 🤝 Example Output

```
[ANALYSIS SUMMARY] (LangGraph Engine)
======================================
[FILES] Files Processed: 15
[LINES] Total Lines: 2,847
[LANGUAGES] Languages: 2
   - Python: 12 files, 8 issues
   - JavaScript: 3 files, 2 issues
[ISSUES] Total Issues: 10
[TIME] Processing Time: 12.34s
[TOKENS] LLM Tokens Used: 15,420

[SEVERITY] ISSUES BY SEVERITY:
   [HIGH] High: 2
   [MEDIUM] Medium: 5
   [LOW] Low: 3

[SUCCESS] Completed Agents: security, complexity, performance, documentation
```

## 🛠️ Troubleshooting

**Common Issues:**

1. **Missing API Key**: Ensure `GROQ_API_KEY` is set in your `.env` file
2. **Rate Limits**: Use `--parallel 1` to reduce concurrent requests
3. **Large Codebases**: Enable `--rag` for better handling of large projects
4. **Memory Issues**: Use `--max-files N` to limit analysis scope

**Performance Tips:**
- Use `--agents security,complexity` for focused analysis
- Intelligent caching is enabled by default for faster subsequent runs
- Set `--parallel 2` for balanced performance/resource usage
- Use `--max-files N` to limit analysis scope for large codebases

## 📄 License

MIT License - see LICENSE file for details.

---

**🚀 Ready to analyze your code?** Start with: `python main.py analyze . --detailed`