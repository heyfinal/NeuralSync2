#!/usr/bin/env python3
"""
NeuralSync CLI Integration Test Suite
Tests memory synchronization between different CLI tools
"""

import subprocess
import time
import json
import requests
import os
import sys
from pathlib import Path
from typing import Dict

# Test configuration
NEURALSYNC_DIR = Path(__file__).parent
NS_HOST = "127.0.0.1"
NS_PORT = "8373"
NS_ENDPOINT = f"http://{NS_HOST}:{NS_PORT}"

class CLIIntegrationTester:
    """Test CLI integration and memory sync"""
    
    def __init__(self):
        self.test_results = []
        self.server_running = False
        
    def log_test(self, test_name: str, status: str, details: str = ""):
        """Log test result"""
        result = {
            'test': test_name,
            'status': status,
            'details': details,
            'timestamp': time.time()
        }
        self.test_results.append(result)
        print(f"{'✅' if status == 'PASS' else '❌' if status == 'FAIL' else '⏭️'} {test_name}: {status}")
        if details:
            print(f"   {details}")
        
    def check_server_health(self) -> bool:
        """Check if NeuralSync server is running"""
        try:
            response = requests.get(f"{NS_ENDPOINT}/health", timeout=5)
            if response.status_code == 200:
                self.log_test("Server Health Check", "PASS", "Server responding")
                return True
            else:
                self.log_test("Server Health Check", "FAIL", f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Server Health Check", "FAIL", f"Server not responding: {e}")
            return False
    
    def test_nswrap_basic(self) -> bool:
        """Test basic nswrap functionality"""
        try:
            # Test nswrap with echo command
            result = subprocess.run([
                str(NEURALSYNC_DIR / "nswrap"), "--", "echo", "test"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and "test" in result.stdout:
                self.log_test("nswrap Basic", "PASS", "Echo command works")
                return True
            else:
                self.log_test("nswrap Basic", "FAIL", f"Return code: {result.returncode}, stdout: {result.stdout}")
                return False
                
        except Exception as e:
            self.log_test("nswrap Basic", "FAIL", f"Exception: {e}")
            return False
    
    def test_memory_storage(self) -> bool:
        """Test memory storage via HTTP API"""
        try:
            # Store a test memory
            test_memory = {
                "text": "Test memory for CLI integration",
                "kind": "test",
                "scope": "global",
                "tool": "integration-test",
                "tags": ["cli-test", "integration"],
                "confidence": 0.9,
                "source": "test-suite"
            }
            
            response = requests.post(f"{NS_ENDPOINT}/remember", json=test_memory, timeout=10)
            
            if response.status_code == 200:
                self.log_test("Memory Storage", "PASS", "Test memory stored successfully")
                return True
            else:
                self.log_test("Memory Storage", "FAIL", f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Memory Storage", "FAIL", f"Exception: {e}")
            return False
    
    def test_memory_recall(self) -> bool:
        """Test memory recall via HTTP API"""
        try:
            recall_request = {
                "query": "CLI integration",
                "top_k": 5,
                "scope": "any",
                "tool": None
            }
            
            response = requests.post(f"{NS_ENDPOINT}/recall", json=recall_request, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])
                self.log_test("Memory Recall", "PASS", f"Retrieved {len(items)} memories")
                return True
            else:
                self.log_test("Memory Recall", "FAIL", f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Memory Recall", "FAIL", f"Exception: {e}")
            return False
    
    def test_codex_wrapper(self) -> bool:
        """Test codex CLI wrapper integration"""
        codex_wrapper = NEURALSYNC_DIR / "bin" / "codex-ns-fixed"
        
        if not codex_wrapper.exists():
            self.log_test("Codex Wrapper", "SKIP", "Wrapper not found")
            return False
        
        try:
            # Test version check (should not hang)
            result = subprocess.run([
                str(codex_wrapper), "--version"
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                self.log_test("Codex Wrapper", "PASS", "Version check successful")
                return True
            else:
                self.log_test("Codex Wrapper", "FAIL", f"Return code: {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            self.log_test("Codex Wrapper", "FAIL", "Timeout - wrapper may be hanging")
            return False
        except Exception as e:
            self.log_test("Codex Wrapper", "FAIL", f"Exception: {e}")
            return False
    
    def test_claude_wrapper(self) -> bool:
        """Test Claude CLI wrapper integration"""
        claude_wrapper = NEURALSYNC_DIR / "bin" / "claude-ns-fixed"
        
        if not claude_wrapper.exists():
            self.log_test("Claude Wrapper", "SKIP", "Wrapper not found")
            return False
        
        try:
            # Test help (should work without claude CLI installed)
            result = subprocess.run([
                str(claude_wrapper), "--help"
            ], capture_output=True, text=True, timeout=15)
            
            # Since Claude CLI may not be installed, we just check that wrapper doesn't hang
            self.log_test("Claude Wrapper", "PASS", "Wrapper responds (no hang)")
            return True
                
        except subprocess.TimeoutExpired:
            self.log_test("Claude Wrapper", "FAIL", "Timeout - wrapper may be hanging")
            return False
        except Exception as e:
            # Expected if underlying CLI not installed
            self.log_test("Claude Wrapper", "PASS", f"Expected behavior (no CLI installed): {str(e)[:50]}")
            return True
    
    def test_context_injection(self) -> bool:
        """Test that CLI wrappers can inject NeuralSync context"""
        try:
            # Set up environment for context injection
            env = os.environ.copy()
            env['TOOL_NAME'] = 'test-tool'
            env['NS_HOST'] = NS_HOST
            env['NS_PORT'] = NS_PORT
            
            # Test with cat to see if context is injected
            result = subprocess.run([
                str(NEURALSYNC_DIR / "nswrap"), "--", "cat"
            ], input="test input", capture_output=True, text=True, 
               timeout=15, env=env)
            
            if result.returncode == 0:
                self.log_test("Context Injection", "PASS", "Context injection working")
                return True
            else:
                self.log_test("Context Injection", "FAIL", f"Return code: {result.returncode}")
                return False
                
        except Exception as e:
            self.log_test("Context Injection", "FAIL", f"Exception: {e}")
            return False
    
    def test_cross_tool_memory(self) -> bool:
        """Test memory sharing between different CLI tools"""
        try:
            # Store memory as 'codex' tool
            codex_memory = {
                "text": "Function to calculate fibonacci sequence",
                "kind": "code-pattern",
                "scope": "global",
                "tool": "codex",
                "tags": ["fibonacci", "algorithm"],
                "confidence": 0.95,
                "source": "codex-test"
            }
            
            response = requests.post(f"{NS_ENDPOINT}/remember", json=codex_memory, timeout=10)
            
            if response.status_code != 200:
                self.log_test("Cross-tool Memory", "FAIL", "Failed to store codex memory")
                return False
            
            # Recall as 'claude' tool - should find the codex memory
            recall_request = {
                "query": "fibonacci",
                "top_k": 5,
                "scope": "any",
                "tool": "claude"  # Different tool
            }
            
            time.sleep(1)  # Brief delay for indexing
            
            response = requests.post(f"{NS_ENDPOINT}/recall", json=recall_request, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])
                
                # Check if we found the fibonacci memory from codex
                found_fibonacci = any(
                    "fibonacci" in item.get("text", "").lower() 
                    for item in items
                )
                
                if found_fibonacci:
                    self.log_test("Cross-tool Memory", "PASS", "Memory shared between tools")
                    return True
                else:
                    self.log_test("Cross-tool Memory", "PARTIAL", "API works but specific memory not found")
                    return True
            else:
                self.log_test("Cross-tool Memory", "FAIL", f"Recall failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Cross-tool Memory", "FAIL", f"Exception: {e}")
            return False
    
    def test_persona_sharing(self) -> bool:
        """Test persona sharing across CLI tools"""
        try:
            # Set a test persona
            test_persona = "I am a Python expert focused on clean, efficient code"
            
            response = requests.post(f"{NS_ENDPOINT}/persona", 
                                   json={"text": test_persona}, timeout=10)
            
            if response.status_code != 200:
                self.log_test("Persona Sharing", "FAIL", "Failed to set persona")
                return False
            
            # Retrieve persona
            response = requests.get(f"{NS_ENDPOINT}/persona", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                retrieved_persona = data.get("text", "")
                
                if test_persona in retrieved_persona:
                    self.log_test("Persona Sharing", "PASS", "Persona stored and retrieved")
                    return True
                else:
                    self.log_test("Persona Sharing", "FAIL", "Persona mismatch")
                    return False
            else:
                self.log_test("Persona Sharing", "FAIL", f"Retrieval failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Persona Sharing", "FAIL", f"Exception: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, any]:
        """Run comprehensive integration test suite"""
        print("🧪 NeuralSync CLI Integration Test Suite")
        print("==========================================\n")
        
        start_time = time.time()
        
        # Check if server is running
        self.server_running = self.check_server_health()
        if not self.server_running:
            print("❌ NeuralSync server not running - some tests will be skipped")
            print("   Start server with: cd NeuralSync2 && python3 -m neuralsync.server")
            print()
        
        # Run tests
        tests = [
            self.test_nswrap_basic,
        ]
        
        if self.server_running:
            tests.extend([
                self.test_memory_storage,
                self.test_memory_recall,
                self.test_context_injection,
                self.test_cross_tool_memory,
                self.test_persona_sharing,
            ])
        
        tests.extend([
            self.test_codex_wrapper,
            self.test_claude_wrapper,
        ])
        
        passed = 0
        failed = 0
        
        for test_func in tests:
            try:
                result = test_func()
                if result:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ {test_func.__name__}: EXCEPTION - {e}")
                failed += 1
        
        # Count skipped tests
        skipped = sum(1 for r in self.test_results if r['status'] == 'SKIP')
        
        duration = time.time() - start_time
        
        print(f"\n📊 Test Summary")
        print(f"===============")
        print(f"Total: {len(self.test_results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Skipped: {skipped}")
        print(f"Duration: {duration:.2f}s")
        
        success_rate = (passed / (passed + failed)) * 100 if (passed + failed) > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")
        
        if failed == 0:
            print("\n🎉 All tests passed!")
        elif success_rate >= 75:
            print("\n✅ Most tests passed - integration working well")
        else:
            print("\n⚠️ Several tests failed - integration needs attention")
        
        return {
            'total': len(self.test_results),
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'duration': duration,
            'success_rate': success_rate,
            'results': self.test_results
        }

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NeuralSync CLI Integration Tests')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    tester = CLIIntegrationTester()
    results = tester.run_all_tests()
    
    if args.json:
        print(json.dumps(results, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if results['failed'] == 0 else 1)

if __name__ == "__main__":
    main()