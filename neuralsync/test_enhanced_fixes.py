#!/usr/bin/env python3
"""
Comprehensive test suite for NeuralSync2 enhanced daemon management fixes
Tests all optimization modules and validates performance improvements
"""

import asyncio
import os
import sys
import time
import json
import logging
import subprocess
import psutil
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import concurrent.futures
import threading
import tempfile

# Add neuralsync to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuralsync.robust_service_detector import RobustServiceDetector, ServiceState
from neuralsync.smart_process_discovery import SmartProcessDiscovery, DiscoveryStrategy
from neuralsync.configuration_validator import ConfigurationValidator, ValidationSeverity
from neuralsync.performance_optimizer import PerformanceOptimizer, OptimizationLevel
from neuralsync.enhanced_daemon_manager import EnhancedDaemonManager

# Setup logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result with performance metrics"""
    test_name: str
    success: bool
    duration: float
    performance_metrics: Dict[str, float]
    error_message: Optional[str] = None
    warnings: List[str] = None
    recommendations: List[str] = None

@dataclass
class PerformanceBaseline:
    """Performance baseline metrics"""
    startup_time_baseline: float = 30.0  # Original startup time
    memory_usage_baseline: float = 200.0  # MB
    detection_time_baseline: float = 5.0   # seconds
    service_count: int = 2

class NeuralSyncTestSuite:
    """Comprehensive test suite for enhanced NeuralSync fixes"""
    
    def __init__(self, test_dir: Optional[Path] = None):
        self.test_dir = test_dir or Path.home() / ".neuralsync_test"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance baseline
        self.baseline = PerformanceBaseline()
        
        # Test results
        self.test_results: List[TestResult] = []
        
        # Temporary configuration for testing
        self.test_config_dir = self.test_dir / "test_config"
        self.test_config_dir.mkdir(exist_ok=True)
        
        # Initialize test modules with isolated configuration
        self.service_detector = RobustServiceDetector(self.test_config_dir)
        self.process_discovery = SmartProcessDiscovery(self.test_config_dir)
        self.config_validator = ConfigurationValidator(self.test_config_dir)
        self.performance_optimizer = PerformanceOptimizer(self.test_config_dir)
        
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests and return results"""
        logger.info("Starting comprehensive NeuralSync enhanced fixes test suite...")
        
        test_start_time = time.time()
        all_tests_passed = True
        
        # Test categories
        test_categories = [
            ("Service Detection Tests", self._test_service_detection),
            ("Process Discovery Tests", self._test_process_discovery),
            ("Configuration Validation Tests", self._test_configuration_validation),
            ("Performance Optimization Tests", self._test_performance_optimization),
            ("Integration Tests", self._test_integration),
            ("Performance Comparison Tests", self._test_performance_comparison),
            ("Stress Tests", self._test_stress_scenarios),
        ]
        
        for category_name, test_function in test_categories:
            logger.info(f"Running {category_name}...")
            
            try:
                category_results = await test_function()
                self.test_results.extend(category_results)
                
                # Check if any test in this category failed
                category_failed = any(not result.success for result in category_results)
                if category_failed:
                    all_tests_passed = False
                    logger.warning(f"{category_name} had failures")
                else:
                    logger.info(f"{category_name} passed")
                    
            except Exception as e:
                logger.error(f"Error in {category_name}: {e}")
                all_tests_passed = False
                
                self.test_results.append(TestResult(
                    test_name=f"{category_name}_exception",
                    success=False,
                    duration=0.0,
                    performance_metrics={},
                    error_message=str(e)
                ))
        
        total_duration = time.time() - test_start_time
        
        # Generate comprehensive test report
        report = self._generate_test_report(all_tests_passed, total_duration)
        
        logger.info(f"Test suite completed in {total_duration:.2f}s - {'PASSED' if all_tests_passed else 'FAILED'}")
        return report
    
    async def _test_service_detection(self) -> List[TestResult]:
        """Test robust service detection capabilities"""
        results = []
        
        # Test 1: Basic service detection
        start_time = time.time()
        try:
            detection_result = self.service_detector.detect_service_comprehensive(
                "test-service", 8373
            )
            
            results.append(TestResult(
                test_name="service_detection_basic",
                success=True,
                duration=time.time() - start_time,
                performance_metrics={
                    "detection_time": detection_result.detection_time,
                    "confidence_score": detection_result.confidence_score
                }
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="service_detection_basic",
                success=False,
                duration=time.time() - start_time,
                performance_metrics={},
                error_message=str(e)
            ))
        
        # Test 2: Fast service detection
        start_time = time.time()
        try:
            is_running = self.service_detector.is_service_running_fast("neuralsync-server")
            
            results.append(TestResult(
                test_name="service_detection_fast",
                success=True,
                duration=time.time() - start_time,
                performance_metrics={
                    "is_running": 1.0 if is_running else 0.0
                }
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="service_detection_fast",
                success=False,
                duration=time.time() - start_time,
                performance_metrics={},
                error_message=str(e)
            ))
        
        # Test 3: Stale resource cleanup
        start_time = time.time()
        try:
            cleanup_results = self.service_detector.cleanup_stale_resources()
            
            results.append(TestResult(
                test_name="service_detection_cleanup",
                success=True,
                duration=time.time() - start_time,
                performance_metrics={
                    "pid_files_cleaned": len(cleanup_results.get('pid_files_cleaned', [])),
                    "locks_cleaned": len(cleanup_results.get('locks_cleaned', []))
                }
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="service_detection_cleanup",
                success=False,
                duration=time.time() - start_time,
                performance_metrics={},
                error_message=str(e)
            ))
        
        return results
    
    async def _test_process_discovery(self) -> List[TestResult]:
        """Test smart process discovery capabilities"""
        results = []
        
        # Test 1: NeuralSync process discovery
        start_time = time.time()
        try:
            processes = await self.process_discovery.discover_neuralsync_processes(
                DiscoveryStrategy.THOROUGH
            )
            
            results.append(TestResult(
                test_name="process_discovery_neuralsync",
                success=True,
                duration=time.time() - start_time,
                performance_metrics={
                    "processes_found": len(processes),
                    "avg_confidence": sum(p.confidence for p in processes) / len(processes) if processes else 0
                }
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="process_discovery_neuralsync",
                success=False,
                duration=time.time() - start_time,
                performance_metrics={},
                error_message=str(e)
            ))
        
        # Test 2: Port usage discovery
        start_time = time.time()
        try:
            test_ports = [8373, 8374, 22, 80, 443]
            port_usage = await self.process_discovery.discover_port_usage(test_ports)
            
            results.append(TestResult(
                test_name="process_discovery_ports",
                success=True,
                duration=time.time() - start_time,
                performance_metrics={
                    "ports_checked": len(test_ports),
                    "ports_in_use": sum(1 for p in port_usage.values() if p.state != 'AVAILABLE'),
                    "available_ports": sum(1 for p in port_usage.values() if p.state == 'AVAILABLE')
                }
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="process_discovery_ports",
                success=False,
                duration=time.time() - start_time,
                performance_metrics={},
                error_message=str(e)
            ))
        
        # Test 3: Port conflict detection
        start_time = time.time()
        try:
            service_ports = {
                'neuralsync-server': 8373,
                'test-service': 22  # Likely to have conflict
            }
            conflicts = await self.process_discovery.detect_port_conflicts(service_ports)
            
            results.append(TestResult(
                test_name="process_discovery_conflicts",
                success=True,
                duration=time.time() - start_time,
                performance_metrics={
                    "conflicts_detected": len(conflicts),
                    "auto_resolvable": sum(1 for c in conflicts.values() if c.can_auto_resolve)
                }
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="process_discovery_conflicts",
                success=False,
                duration=time.time() - start_time,
                performance_metrics={},
                error_message=str(e)
            ))
        
        return results
    
    async def _test_configuration_validation(self) -> List[TestResult]:
        """Test configuration validation capabilities"""
        results = []
        
        # Test 1: Valid configuration validation
        start_time = time.time()
        try:
            valid_config = {
                'site_id': 'test-site-123',
                'bind_host': '127.0.0.1',
                'bind_port': 8373,
                'db_path': str(self.test_config_dir / 'test.db'),
                'oplog_path': str(self.test_config_dir / 'test.jsonl'),
                'vector_dim': 512
            }
            
            issues = self.config_validator.validate_configuration(valid_config, 'development')
            critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
            
            results.append(TestResult(
                test_name="config_validation_valid",
                success=len(critical_issues) == 0,
                duration=time.time() - start_time,
                performance_metrics={
                    "total_issues": len(issues),
                    "critical_issues": len(critical_issues),
                    "auto_fixable": sum(1 for i in issues if i.auto_fixable)
                }
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="config_validation_valid",
                success=False,
                duration=time.time() - start_time,
                performance_metrics={},
                error_message=str(e)
            ))
        
        # Test 2: Invalid configuration validation
        start_time = time.time()
        try:
            invalid_config = {
                'bind_host': 'invalid-host',
                'bind_port': 99999,  # Invalid port
                'vector_dim': -1     # Invalid dimension
            }
            
            issues = self.config_validator.validate_configuration(invalid_config)
            
            results.append(TestResult(
                test_name="config_validation_invalid",
                success=len(issues) > 0,  # Should detect issues
                duration=time.time() - start_time,
                performance_metrics={
                    "issues_detected": len(issues),
                    "error_issues": sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)
                }
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="config_validation_invalid",
                success=False,
                duration=time.time() - start_time,
                performance_metrics={},
                error_message=str(e)
            ))
        
        # Test 3: Auto-fix functionality
        start_time = time.time()
        try:
            broken_config = {
                'bind_port': 'not_a_number',
                'vector_dim': 'invalid'
            }
            
            issues = self.config_validator.validate_configuration(broken_config)
            fixed_config, fix_log = self.config_validator.auto_fix_issues(broken_config, issues)
            
            results.append(TestResult(
                test_name="config_validation_autofix",
                success=len(fix_log) > 0,
                duration=time.time() - start_time,
                performance_metrics={
                    "fixes_applied": len(fix_log),
                    "original_issues": len(issues)
                }
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="config_validation_autofix",
                success=False,
                duration=time.time() - start_time,
                performance_metrics={},
                error_message=str(e)
            ))
        
        return results
    
    async def _test_performance_optimization(self) -> List[TestResult]:
        """Test performance optimization capabilities"""
        results = []
        
        # Test 1: Profile selection
        start_time = time.time()
        try:
            profile = self.performance_optimizer.select_optimal_profile()
            
            results.append(TestResult(
                test_name="performance_profile_selection",
                success=profile is not None,
                duration=time.time() - start_time,
                performance_metrics={
                    "max_workers": profile.max_workers,
                    "cache_size_mb": profile.cache_size / (1024 * 1024),
                    "optimization_level": hash(profile.level.value) % 100  # Simple hash
                }
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="performance_profile_selection",
                success=False,
                duration=time.time() - start_time,
                performance_metrics={},
                error_message=str(e)
            ))
        
        # Test 2: Service startup optimization
        start_time = time.time()
        try:
            test_services = {
                'test-service-1': {'command': ['echo', 'test1'], 'expected_port': 8001},
                'test-service-2': {'command': ['echo', 'test2'], 'expected_port': 8002}
            }
            
            optimization_results = await self.performance_optimizer.optimize_service_startup(
                test_services, OptimizationLevel.BALANCED
            )
            
            success = any(result.success for result in optimization_results.values())
            
            results.append(TestResult(
                test_name="performance_startup_optimization",
                success=success,
                duration=time.time() - start_time,
                performance_metrics={
                    "services_optimized": len(optimization_results),
                    "successful_optimizations": sum(1 for r in optimization_results.values() if r.success),
                    "total_performance_gain": sum(r.performance_gain for r in optimization_results.values())
                }
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="performance_startup_optimization",
                success=False,
                duration=time.time() - start_time,
                performance_metrics={},
                error_message=str(e)
            ))
        
        return results
    
    async def _test_integration(self) -> List[TestResult]:
        """Test integration between all modules"""
        results = []
        
        # Test 1: Enhanced daemon manager initialization
        start_time = time.time()
        try:
            daemon_manager = EnhancedDaemonManager(self.test_config_dir)
            
            # Test that all modules are properly initialized
            modules_initialized = all([
                daemon_manager.service_detector is not None,
                daemon_manager.process_discovery is not None,
                daemon_manager.config_validator is not None,
                daemon_manager.performance_optimizer is not None
            ])
            
            results.append(TestResult(
                test_name="integration_daemon_manager_init",
                success=modules_initialized,
                duration=time.time() - start_time,
                performance_metrics={
                    "modules_count": 4,
                    "services_registered": len(daemon_manager.services)
                }
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="integration_daemon_manager_init",
                success=False,
                duration=time.time() - start_time,
                performance_metrics={},
                error_message=str(e)
            ))
        
        # Test 2: Service status coordination
        start_time = time.time()
        try:
            daemon_manager = EnhancedDaemonManager(self.test_config_dir)
            status_summary = daemon_manager.get_enhanced_status_summary()
            
            required_sections = ['daemon_manager', 'services', 'performance', 'detection_stats', 'discovery_stats']
            all_sections_present = all(section in status_summary for section in required_sections)
            
            results.append(TestResult(
                test_name="integration_status_coordination",
                success=all_sections_present,
                duration=time.time() - start_time,
                performance_metrics={
                    "status_sections": len(status_summary),
                    "required_sections": len(required_sections)
                }
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="integration_status_coordination",
                success=False,
                duration=time.time() - start_time,
                performance_metrics={},
                error_message=str(e)
            ))
        
        return results
    
    async def _test_performance_comparison(self) -> List[TestResult]:
        """Test performance improvements compared to baseline"""
        results = []
        
        # Test 1: Service detection performance
        start_time = time.time()
        try:
            # Run multiple detection cycles and measure performance
            detection_times = []
            for _ in range(10):
                cycle_start = time.time()
                self.service_detector.detect_service_comprehensive("test-service")
                detection_times.append(time.time() - cycle_start)
            
            avg_detection_time = sum(detection_times) / len(detection_times)
            performance_improvement = max(0, (self.baseline.detection_time_baseline - avg_detection_time) / self.baseline.detection_time_baseline * 100)
            
            results.append(TestResult(
                test_name="performance_comparison_detection",
                success=avg_detection_time < self.baseline.detection_time_baseline,
                duration=time.time() - start_time,
                performance_metrics={
                    "avg_detection_time": avg_detection_time,
                    "baseline_detection_time": self.baseline.detection_time_baseline,
                    "performance_improvement_percent": performance_improvement
                }
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="performance_comparison_detection",
                success=False,
                duration=time.time() - start_time,
                performance_metrics={},
                error_message=str(e)
            ))
        
        # Test 2: Memory usage comparison
        start_time = time.time()
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_efficiency = memory_mb < self.baseline.memory_usage_baseline
            
            results.append(TestResult(
                test_name="performance_comparison_memory",
                success=memory_efficiency,
                duration=time.time() - start_time,
                performance_metrics={
                    "current_memory_mb": memory_mb,
                    "baseline_memory_mb": self.baseline.memory_usage_baseline,
                    "memory_efficient": 1.0 if memory_efficiency else 0.0
                }
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="performance_comparison_memory",
                success=False,
                duration=time.time() - start_time,
                performance_metrics={},
                error_message=str(e)
            ))
        
        return results
    
    async def _test_stress_scenarios(self) -> List[TestResult]:
        """Test system under stress conditions"""
        results = []
        
        # Test 1: Concurrent service detection
        start_time = time.time()
        try:
            concurrent_tasks = []
            
            async def detect_service(service_id):
                return self.service_detector.detect_service_comprehensive(f"stress-test-{service_id}")
            
            # Run 20 concurrent detections
            for i in range(20):
                task = detect_service(i)
                concurrent_tasks.append(task)
            
            detection_results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            successful_detections = sum(1 for r in detection_results if not isinstance(r, Exception))
            
            results.append(TestResult(
                test_name="stress_concurrent_detection",
                success=successful_detections >= 18,  # Allow 2 failures
                duration=time.time() - start_time,
                performance_metrics={
                    "concurrent_tasks": len(concurrent_tasks),
                    "successful_detections": successful_detections,
                    "failure_rate": (len(concurrent_tasks) - successful_detections) / len(concurrent_tasks)
                }
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="stress_concurrent_detection",
                success=False,
                duration=time.time() - start_time,
                performance_metrics={},
                error_message=str(e)
            ))
        
        # Test 2: Rapid configuration validation
        start_time = time.time()
        try:
            validation_times = []
            
            for i in range(50):
                config = {
                    'site_id': f'test-{i}',
                    'bind_port': 8373 + i,
                    'vector_dim': 512
                }
                
                validation_start = time.time()
                self.config_validator.validate_configuration(config)
                validation_times.append(time.time() - validation_start)
            
            avg_validation_time = sum(validation_times) / len(validation_times)
            max_validation_time = max(validation_times)
            
            results.append(TestResult(
                test_name="stress_rapid_validation",
                success=avg_validation_time < 0.1 and max_validation_time < 0.5,  # Sub-100ms average, sub-500ms max
                duration=time.time() - start_time,
                performance_metrics={
                    "validations_performed": len(validation_times),
                    "avg_validation_time": avg_validation_time,
                    "max_validation_time": max_validation_time
                }
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="stress_rapid_validation",
                success=False,
                duration=time.time() - start_time,
                performance_metrics={},
                error_message=str(e)
            ))
        
        return results
    
    def _generate_test_report(self, all_passed: bool, total_duration: float) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        # Categorize results
        passed_tests = [r for r in self.test_results if r.success]
        failed_tests = [r for r in self.test_results if not r.success]
        
        # Calculate performance metrics
        total_performance_gain = 0
        performance_tests = [r for r in self.test_results if 'performance_improvement_percent' in r.performance_metrics]
        if performance_tests:
            total_performance_gain = sum(r.performance_metrics['performance_improvement_percent'] for r in performance_tests)
        
        # Memory efficiency
        memory_tests = [r for r in self.test_results if 'current_memory_mb' in r.performance_metrics]
        avg_memory_usage = sum(r.performance_metrics['current_memory_mb'] for r in memory_tests) / len(memory_tests) if memory_tests else 0
        
        # Detection efficiency
        detection_tests = [r for r in self.test_results if 'detection_time' in r.performance_metrics]
        avg_detection_time = sum(r.performance_metrics['detection_time'] for r in detection_tests) / len(detection_tests) if detection_tests else 0
        
        # Generate recommendations based on test results
        recommendations = []
        if failed_tests:
            recommendations.append(f"Address {len(failed_tests)} failing tests before production deployment")
        
        if avg_memory_usage > self.baseline.memory_usage_baseline:
            recommendations.append("Consider reducing memory usage through cache optimization")
        
        if avg_detection_time > 1.0:
            recommendations.append("Service detection times are high - consider optimization")
        
        report = {
            'test_summary': {
                'total_tests': len(self.test_results),
                'passed_tests': len(passed_tests),
                'failed_tests': len(failed_tests),
                'success_rate': len(passed_tests) / len(self.test_results) * 100 if self.test_results else 0,
                'total_duration': total_duration,
                'overall_result': 'PASSED' if all_passed else 'FAILED'
            },
            'performance_analysis': {
                'total_performance_gain_percent': total_performance_gain,
                'avg_memory_usage_mb': avg_memory_usage,
                'memory_vs_baseline': avg_memory_usage - self.baseline.memory_usage_baseline,
                'avg_detection_time': avg_detection_time,
                'detection_vs_baseline': avg_detection_time - self.baseline.detection_time_baseline
            },
            'test_results_by_category': {
                'service_detection': [r for r in self.test_results if r.test_name.startswith('service_detection')],
                'process_discovery': [r for r in self.test_results if r.test_name.startswith('process_discovery')],
                'configuration_validation': [r for r in self.test_results if r.test_name.startswith('config_validation')],
                'performance_optimization': [r for r in self.test_results if r.test_name.startswith('performance')],
                'integration': [r for r in self.test_results if r.test_name.startswith('integration')],
                'stress_tests': [r for r in self.test_results if r.test_name.startswith('stress')]
            },
            'failed_tests': [asdict(r) for r in failed_tests],
            'recommendations': recommendations,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': sys.version,
                'platform': sys.platform
            }
        }
        
        return report
    
    def cleanup(self):
        """Clean up test resources"""
        try:
            # Shutdown modules
            self.process_discovery.shutdown()
            self.config_validator.shutdown()
            self.performance_optimizer.shutdown()
            
            # Clean up test directory
            import shutil
            if self.test_dir.exists():
                shutil.rmtree(self.test_dir, ignore_errors=True)
                
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


async def main():
    """Run the comprehensive test suite"""
    test_suite = NeuralSyncTestSuite()
    
    try:
        print("🧪 Starting NeuralSync2 Enhanced Fixes Test Suite")
        print("=" * 60)
        
        # Run comprehensive tests
        report = await test_suite.run_comprehensive_tests()
        
        # Print summary
        summary = report['test_summary']
        performance = report['performance_analysis']
        
        print(f"\n📊 TEST SUMMARY")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Duration: {summary['total_duration']:.2f}s")
        print(f"Overall Result: {summary['overall_result']}")
        
        print(f"\n⚡ PERFORMANCE ANALYSIS")
        print(f"Total Performance Gain: {performance['total_performance_gain_percent']:.1f}%")
        print(f"Memory Usage: {performance['avg_memory_usage_mb']:.1f}MB (vs {test_suite.baseline.memory_usage_baseline:.1f}MB baseline)")
        print(f"Detection Time: {performance['avg_detection_time']:.3f}s (vs {test_suite.baseline.detection_time_baseline:.1f}s baseline)")
        
        if report['failed_tests']:
            print(f"\n❌ FAILED TESTS:")
            for failed_test in report['failed_tests']:
                print(f"  - {failed_test['test_name']}: {failed_test.get('error_message', 'Unknown error')}")
        
        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
        # Save detailed report
        report_file = test_suite.test_dir / "test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        print("\n" + "=" * 60)
        print(f"🏁 Test suite completed: {'✅ PASSED' if summary['overall_result'] == 'PASSED' else '❌ FAILED'}")
        
    finally:
        test_suite.cleanup()


if __name__ == "__main__":
    asyncio.run(main())