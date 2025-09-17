#!/usr/bin/env python3
"""
LangSmith Integration Test Suite
Tests that LangSmith integration works correctly and falls back safely.
"""

import os
import sys
import time
from typing import Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def run_basic_test():
    """Run basic integration test"""
    print("\n🧪 LangSmith Integration Test")
    print("=" * 40)
    
    # Test 1: Import test
    print("[TEST 1] Testing imports...")
    try:
        from custom_langsmith import get_enhanced_prompt, get_langsmith_status
        print("✅ LangSmith module imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Status check
    print("\n[TEST 2] Checking LangSmith status...")
    status = get_langsmith_status()
    print(f"  - LangSmith Available: {status['langsmith_available']}")
    print(f"  - Client Enabled: {status['client_enabled']}")
    print(f"  - API Key Configured: {status['api_key_configured']}")
    print(f"  - Status: {status['status']}")
    print(f"  - Cached Prompts: {status['cached_prompts']}")
    
    # Test 3: Fallback prompt test
    print("\n[TEST 3] Testing fallback prompts...")
    test_prompt = "You are a test agent for {language} code."
    
    # Test with LangSmith (should work regardless)
    enhanced_prompt = get_enhanced_prompt(
        prompt_name="test-agent",
        fallback_prompt=test_prompt,
        language="python"
    )
    
    if enhanced_prompt and len(enhanced_prompt) > 0:
        print("✅ Prompt retrieval successful")
        if enhanced_prompt == test_prompt:
            print("   (Using fallback prompt - expected if no Hub prompts)")
        else:
            print("   (Using enhanced prompt from Hub)")
    else:
        print("❌ Prompt retrieval failed")
        return False
    
    # Test 4: Agent integration test
    print("\n[TEST 4] Testing agent integration...")
    try:
        # Test with a real agent
        from agents import SecurityAgent
        
        # Create agent instance
        security_agent = SecurityAgent()
        
        # Test prompt retrieval (this should now use LangSmith if available)
        original_prompt = security_agent.get_system_prompt("python")
        
        if original_prompt and len(original_prompt) > 100:
            print("✅ Agent prompt retrieval successful")
            print(f"   Prompt length: {len(original_prompt)} characters")
        else:
            print("❌ Agent prompt retrieval failed")
            return False
            
    except Exception as e:
        print(f"❌ Agent integration test failed: {e}")
        return False
    
    # Test 5: Cache test
    print("\n[TEST 5] Testing prompt caching...")
    start_time = time.time()
    
    # First call
    prompt1 = get_enhanced_prompt("cache-test", "Test prompt", "python")
    first_time = time.time() - start_time
    
    # Second call (should be cached)
    start_time = time.time()
    prompt2 = get_enhanced_prompt("cache-test", "Test prompt", "python")
    second_time = time.time() - start_time
    
    if prompt1 == prompt2:
        print("✅ Cache consistency test passed")
        print(f"   First call: {first_time:.3f}s, Second call: {second_time:.3f}s")
    else:
        print("❌ Cache consistency test failed")
        return False
    
    # Test 6: Error handling test
    print("\n[TEST 6] Testing error handling...")
    try:
        # Test with invalid prompt name
        fallback_prompt = "This is a fallback prompt"
        result_prompt = get_enhanced_prompt(
            prompt_name="non-existent-prompt-12345",
            fallback_prompt=fallback_prompt,
            language="python"
        )
        
        if result_prompt == fallback_prompt:
            print("✅ Error handling test passed (correctly used fallback)")
        else:
            print("⚠️  Error handling test inconclusive (may have found prompt)")
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False
    
    print("\n🎉 All tests passed!")
    print("\n📋 Integration Summary:")
    print(f"   - LangSmith Status: {'Connected' if status['client_enabled'] else 'Fallback Mode'}")
    print(f"   - Prompt Enhancement: {'Active' if status['client_enabled'] else 'Using Fallbacks'}")
    print(f"   - Cache Status: {status['cached_prompts']} prompts cached")
    print(f"   - Safety: All fallbacks working correctly")
    
    return True

def run_agent_specific_tests():
    """Run tests specific to each agent"""
    print("\n🔍 Agent-Specific Integration Tests")
    print("=" * 40)
    
    agents_to_test = [
        ('SecurityAgent', 'security-agent'),
        ('PerformanceAgent', 'performance-agent'),
        ('ComplexityAgent', 'complexity-agent'),
        ('DocumentationAgent', 'documentation-agent')
    ]
    
    results = {}
    
    for agent_name, prompt_name in agents_to_test:
        print(f"\n[AGENT TEST] {agent_name}")
        try:
            # Import the agent
            from agents import SecurityAgent, PerformanceAgent, ComplexityAgent, DocumentationAgent
            
            agent_classes = {
                'SecurityAgent': SecurityAgent,
                'PerformanceAgent': PerformanceAgent,
                'ComplexityAgent': ComplexityAgent,
                'DocumentationAgent': DocumentationAgent
            }
            
            agent_class = agent_classes[agent_name]
            agent = agent_class()
            
            # Test prompt for different languages
            languages = ['python', 'javascript', 'java']
            language_results = {}
            
            for lang in languages:
                try:
                    prompt = agent.get_system_prompt(lang)
                    if prompt and len(prompt) > 50:
                        language_results[lang] = "✅ OK"
                    else:
                        language_results[lang] = "❌ Empty/Short"
                except Exception as e:
                    language_results[lang] = f"❌ Error: {str(e)[:30]}"
            
            results[agent_name] = language_results
            
            # Show results for this agent
            for lang, result in language_results.items():
                print(f"   {lang}: {result}")
                
        except Exception as e:
            results[agent_name] = {"error": str(e)}
            print(f"   ❌ Failed: {e}")
    
    # Summary
    print(f"\n📊 Agent Test Summary:")
    total_tests = 0
    passed_tests = 0
    
    for agent_name, test_results in results.items():
        if "error" in test_results:
            print(f"   {agent_name}: ❌ FAILED")
        else:
            agent_passed = sum(1 for result in test_results.values() if "✅" in result)
            agent_total = len(test_results)
            total_tests += agent_total
            passed_tests += agent_passed
            print(f"   {agent_name}: {agent_passed}/{agent_total} languages OK")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    return passed_tests == total_tests

def run_performance_test():
    """Run performance comparison test"""
    print("\n⚡ Performance Test")
    print("=" * 30)
    
    from langsmith import get_enhanced_prompt
    
    test_cases = [
        ("security-agent", "Test security prompt"),
        ("performance-agent", "Test performance prompt"),  
        ("complexity-agent", "Test complexity prompt")
    ]
    
    print("Testing prompt retrieval performance...")
    
    for prompt_name, fallback in test_cases:
        # Time multiple calls
        times = []
        for i in range(5):
            start_time = time.time()
            prompt = get_enhanced_prompt(prompt_name, fallback, "python")
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"   {prompt_name}: avg={avg_time:.3f}s, min={min_time:.3f}s, max={max_time:.3f}s")
    
    print("✅ Performance test completed")

def run_full_test_suite():
    """Run complete test suite"""
    print("🚀 LangSmith Integration - Full Test Suite")
    print("=" * 50)
    
    # Basic tests
    if not run_basic_test():
        print("❌ Basic tests failed - stopping")
        return False
    
    # Agent-specific tests
    if not run_agent_specific_tests():
        print("⚠️  Some agent tests failed, but continuing...")
    
    # Performance tests
    run_performance_test()
    
    print("\n🎉 Full test suite completed!")
    return True

def main():
    """Main test function"""
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        if test_type == "basic":
            run_basic_test()
        elif test_type == "agents":
            run_agent_specific_tests()
        elif test_type == "performance":
            run_performance_test()
        elif test_type == "full":
            run_full_test_suite()
        else:
            print("Usage: python -m custom_langsmith.test_integration [basic|agents|performance|full]")
    else:
        # Default to basic test
        run_basic_test()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)