#!/bin/bash
# Ultra-fast deployment script for DigitalOcean

set -e  # Exit on any error

echo "🚀 STREAMLINED CQI DEPLOYMENT"
echo "=============================="

# Check if we're in a git repo
if [ ! -d ".git" ]; then
    echo "❌ Not in a git repository. Run: git init && git add . && git commit -m 'Initial commit'"
    exit 1
fi

# Check for required files
echo "📋 Checking deployment files..."
required_files=("requirements.txt" "api_backend.py" "Dockerfile" ".do/app.yaml")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing required file: $file"
        exit 1
    fi
    echo "✅ Found: $file"
done

# Check if requirements.txt is lightweight
if grep -q "sentence-transformers\|scikit-learn" requirements.txt && ! grep -q "^#.*sentence-transformers\|^#.*scikit-learn" requirements.txt; then
    echo "⚠️  WARNING: Heavy ML dependencies detected in requirements.txt"
    echo "   This will cause deployment failures due to disk space limits"
    echo "   Consider using the lightweight version"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Show deployment size estimate
echo "📊 Deployment footprint estimate:"
if grep -q "sentence-transformers" requirements.txt && ! grep -q "^#.*sentence-transformers" requirements.txt; then
    echo "   Size: ~2-3 GB (WITH heavy ML)"
    echo "   Build time: 10-15 minutes"
    echo "   Cost: $98+/month (requires professional plan)"
else
    echo "   Size: ~200-300 MB (lightweight)"
    echo "   Build time: 2-3 minutes"
    echo "   Cost: $5-12/month (basic plan works)"
fi

echo ""
echo "🔧 DEPLOYMENT OPTIONS:"
echo "1. DigitalOcean App Platform (Recommended)"
echo "2. Docker deployment"
echo "3. Local testing first"
echo ""

read -p "Choose option (1-3): " choice

case $choice in
    1)
        echo ""
        echo "🌊 DIGITALOCEAN APP PLATFORM DEPLOYMENT"
        echo "========================================"
        echo ""
        echo "📝 Steps to deploy:"
        echo "1. Go to: https://cloud.digitalocean.com/apps"
        echo "2. Click 'Create App'"
        echo "3. Connect your GitHub repository"
        echo "4. DigitalOcean will auto-detect the .do/app.yaml"
        echo "5. Set environment variables:"
        echo "   - GROQ_API_KEY=your_actual_groq_key"
        echo "6. Choose instance size:"
        if grep -q "sentence-transformers" requirements.txt && ! grep -q "^#.*sentence-transformers" requirements.txt; then
            echo "   - Professional ($98/month) - Required for ML"
        else
            echo "   - Basic ($5/month) - Sufficient for lightweight version"
        fi
        echo "7. Deploy!"
        echo ""
        echo "🔗 Your API will be at: https://your-app-name.ondigitalocean.app"
        ;;
    2)
        echo ""
        echo "🐳 DOCKER DEPLOYMENT"
        echo "==================="
        echo ""
        echo "📝 Testing Docker build locally:"
        echo "docker build -t cqi-backend ."
        echo "docker run -p 8000:8000 -e GROQ_API_KEY=your_key cqi-backend"
        echo ""
        echo "📤 For deployment to any Docker platform:"
        echo "1. Build: docker build -t your-registry/cqi-backend ."
        echo "2. Push: docker push your-registry/cqi-backend"
        echo "3. Deploy on your preferred platform"
        ;;
    3)
        echo ""
        echo "🧪 LOCAL TESTING"
        echo "==============="
        echo ""
        if command -v uv >/dev/null 2>&1; then
            echo "✅ uv detected - using ultra-fast installation"
            echo "uv pip install -r requirements.txt"
            echo "python api_backend.py"
        else
            echo "📦 Installing uv for faster package management..."
            echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
            echo "source ~/.bashrc"
            echo "uv pip install -r requirements.txt"
            echo "python api_backend.py"
        fi
        echo ""
        echo "🌐 Test at: http://localhost:8000/api/test"
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo ""
echo "✨ Deployment configuration complete!"
echo "📚 See DIGITALOCEAN_DEPLOYMENT.md for detailed instructions"