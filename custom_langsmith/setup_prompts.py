#!/usr/bin/env python3
"""
Setup Script: Push Agent Prompts to LangSmith Hub

This script extracts your current agent prompts and pushes enhanced versions
to LangSmith Hub for improved performance and context awareness.

Usage:
    python -m langsmith.setup_prompts
    python langsmith/setup_prompts.py
"""

import os
import sys
from typing import Dict, Any

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def main():
    """Main setup function"""
    print("🚀 LangSmith Prompts Setup")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        return
    
    # Initialize hub client
    try:
        from langsmith.hub_client import LangSmithHubClient
        hub_client = LangSmithHubClient()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the project root directory")
        return
    
    # Check hub client status
    status = hub_client.get_hub_status()
    print(f"[STATUS] LangSmith Available: {status['langsmith_available']}")
    print(f"[STATUS] API Key Configured: {status['api_key_configured']}")
    print(f"[STATUS] Can Push Prompts: {status['can_push']}")
    print(f"[STATUS] Can Import Agents: {status['can_import_agents']}")
    
    if not status['can_push']:
        print("\n❌ Cannot push prompts. Please check:")
        print("  1. LangSmith is installed: pip install langsmith langchain")
        print("  2. API key is set: export LANGSMITH_API_KEY=your_key_here")
        print("  3. Running from project root directory")
        return
    
    # Extract current prompts
    print("\n📖 Extracting current agent prompts...")
    current_prompts = hub_client.extract_current_prompts()
    
    if not current_prompts:
        print("❌ No prompts extracted. Check agent imports.")
        return
    
    print(f"✅ Extracted {len(current_prompts)} agent prompts:")
    for prompt_name in current_prompts.keys():
        print(f"  - {prompt_name}")
    
    # Create enhanced versions
    print("\n🔧 Creating enhanced prompts...")
    enhanced_prompts = hub_client.create_enhanced_prompts(current_prompts)
    
    # Show preview
    print("\n📋 Enhanced Prompt Preview:")
    for name, data in enhanced_prompts.items():
        print(f"  - {name}: {data['description']}")
        print(f"    Tags: {', '.join(data['tags'])}")
        print(f"    Length: {len(data['template'])} characters")
    
    # Confirm before pushing
    print("\n⚠️  Ready to push prompts to LangSmith Hub")
    print("This will create/update prompts in the 'code-analysis' namespace")
    
    if not confirm_action("Push prompts to hub"):
        print("❌ Cancelled by user")
        return
    
    # Push to hub
    print("\n🚀 Pushing prompts to LangSmith Hub...")
    results = hub_client.push_prompts_to_hub(enhanced_prompts)
    
    # Report results
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    print(f"\n📊 Push Results: {successful}/{total} successful")
    
    for prompt_name, success in results.items():
        status_emoji = "✅" if success else "❌"
        print(f"  {status_emoji} {prompt_name}")
    
    if successful == total:
        print("\n🎉 All prompts successfully pushed to LangSmith Hub!")
        print("\n📝 Next Steps:")
        print("  1. Your agents will now use enhanced prompts automatically")
        print("  2. Run tests to verify everything works: python -m langsmith.test_integration")
        print("  3. Check prompts in LangSmith Hub web interface")
        print("  4. Monitor performance improvements in your analysis results")
        
        # Test the integration
        if confirm_action("Run integration test now"):
            print("\n🧪 Running integration test...")
            run_integration_test()
    else:
        print(f"\n⚠️  {total - successful} prompts failed to push")
        print("Check the error messages above and try again")

def check_environment() -> bool:
    """Check if environment is properly configured"""
    print("[ENV] Checking environment...")
    
    # Check Python path
    if not os.path.exists('agents'):
        print("❌ 'agents' folder not found. Run from project root directory.")
        return False
    
    # Check API key
    api_key = os.getenv('LANGSMITH_API_KEY')
    if not api_key:
        print("❌ LANGSMITH_API_KEY not set")
        print("Set your API key:")
        print("  export LANGSMITH_API_KEY=your_api_key_here")
        return False
    
    # Check LangSmith installation
    try:
        import langsmith
        import langchain
        print("✅ LangSmith and LangChain installed")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install langsmith langchain")
        return False
    
    print("✅ Environment check passed")
    return True

def confirm_action(action: str) -> bool:
    """Get user confirmation for an action"""
    try:
        response = input(f"{action}? [y/N]: ").strip().lower()
        return response in ['y', 'yes']
    except (EOFError, KeyboardInterrupt):
        return False

def run_integration_test():
    """Run a quick integration test"""
    try:
        from langsmith.test_integration import run_basic_test
        run_basic_test()
    except ImportError:
        print("⚠️  Integration test module not found")
    except Exception as e:
        print(f"⚠️  Integration test failed: {e}")

def show_usage():
    """Show usage instructions"""
    print("Usage:")
    print("  python -m langsmith.setup_prompts")
    print("  python langsmith/setup_prompts.py")
    print()
    print("Prerequisites:")
    print("  1. Set LANGSMITH_API_KEY environment variable")
    print("  2. Install: pip install langsmith langchain") 
    print("  3. Run from project root directory (where 'agents' folder is)")

if __name__ == "__main__":
    # Handle help
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_usage()
        sys.exit(0)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)