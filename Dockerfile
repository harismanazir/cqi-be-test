# Ultra-fast Dockerfile using uv for lightning-fast builds
FROM python:3.11-slim

# Install uv - the blazing fast Python package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy dependency files
COPY requirements.txt .
COPY pyproject.toml* ./

# Install dependencies using uv (10x faster than pip) with ML support
# LangSmith/LangChain versions used:
# - langsmith>=0.4.0,<0.5.0 (for enhanced prompt management with v0.4.x API)
# - langchain-groq>=0.1.9 (for Groq LLM integration)
# - langchain-core>=0.2.26 (core LangChain functionality)
# - langgraph>=0.1.1 (for workflow orchestration)

# Install PyTorch CPU-only first to reduce memory usage
RUN uv pip install --system --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Verify critical imports including LangSmith/LangChain
RUN python -c "import fastapi, uvicorn, git, github; print('✅ Core dependencies verified')" && \
    python -c "import langsmith, langchain_groq, langgraph; print('✅ LangSmith/LangChain dependencies verified')" && \
    python -c "import langsmith; print(f'📦 LangSmith version: {langsmith.__version__}')" && \
    python -c "import langchain_core; print(f'📦 LangChain Core version: {langchain_core.__version__}')"

# Copy application code
COPY . .

# Remove unnecessary files that cause deployment issues
RUN rm -rf \
    .analysis_cache/ \
    .rag_cache/ \
    __pycache__/ \
    .git/ \
    .vscode/ \
    .idea/ \
    *.pkl \
    temp/ \
    tmp/ \
    .pytest_cache/ \
    && find . -name "*.pyc" -delete \
    && find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Create app user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/test || exit 1

# Start application
CMD ["python", "api_backend.py"]