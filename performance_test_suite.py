#!/usr/bin/env python3
"""
Performance Test Suite for NeuralSync v2 Optimizations
Comprehensive testing to validate sub-second response times
"""

import time
import asyncio
import statistics
import logging
import json
import sys
import subprocess
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import concurrent.futures
from contextlib import contextmanager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PerformanceTestResult:
    """Individual test result"""
    test_name: str
    response_time_ms: float
    success: bool
    from_cache: bool = False
    context_size_bytes: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TestSuiteResults:
    """Complete test suite results"""
    total_tests: int
    successful_tests: int
    failed_tests: int
    average_response_time_ms: float
    median_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    sub_second_tests: int
    sub_500ms_tests: int
    sub_100ms_tests: int
    cache_hit_rate: float
    test_results: List[PerformanceTestResult] = field(default_factory=list)
    optimization_health_score: float = 0.0

class PerformanceTestSuite:
    """Comprehensive performance test suite"""
    
    def __init__(self, neuralsync_dir: Path = None):
        self.neuralsync_dir = neuralsync_dir or Path(__file__).parent
        self.nswrap_path = self.neuralsync_dir / 'nswrap_optimized'
        self.original_nswrap = self.neuralsync_dir / 'nswrap'
        
        # Test configuration
        self.test_timeout = 10.0  # 10 seconds max per test
        self.performance_targets = {
            'excellent_ms': 200,
            'good_ms': 500,
            'acceptable_ms': 1000,
            'target_cache_hit_rate': 0.7,
            'target_success_rate': 0.95
        }
        
        # Test categories
        self.test_categories = {
            'cli_response': 'CLI wrapper response times',
            'cache_performance': 'Caching system performance',
            'context_loading': 'Context loading optimization',
            'memory_recall': 'Memory recall speed',
            'concurrent_operations': 'Parallel operation handling',
            'system_integration': 'Full system integration'
        }
        
        logger.info(f"PerformanceTestSuite initialized: {self.neuralsync_dir}")
    
    @contextmanager
    def timer(self):
        """High-precision timer context manager"""
        start_time = time.perf_counter()
        yield lambda: (time.perf_counter() - start_time) * 1000
    
    async def test_cli_response_times(self) -> List[PerformanceTestResult]:
        """Test CLI wrapper response times with various commands"""
        
        logger.info("Testing CLI response times...")
        results = []
        
        test_commands = [
            # Fast commands that should bypass context loading
            (['echo', 'hello'], 'echo_simple'),
            (['pwd'], 'pwd_command'),
            (['date', '+%s'], 'date_command'),
            (['ls', '--help'], 'help_command'),
            
            # Commands that might load context
            (['python3', '-c', 'print("hello")'], 'python_simple'),
            (['git', 'status', '--help'], 'git_help'),
            
            # Commands with tool context
            (['echo', 'test', 'with', 'context'], 'echo_with_context'),
        ]
        
        for cmd, test_name in test_commands:
            try:
                with self.timer() as get_time:
                    result = subprocess.run([
                        str(self.nswrap_path), '--'
                    ] + cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=self.test_timeout,
                    env=dict(os.environ, **{
                        'TOOL_NAME': 'test-tool',
                        'NS_FAST_MODE': '1',
                        'NS_DEBUG_PERF': '1'
                    }))
                
                response_time = get_time()
                success = result.returncode == 0
                
                # Check for performance debug info
                from_cache = 'cache hits' in result.stderr if result.stderr else False
                
                results.append(PerformanceTestResult(
                    test_name=f"cli_{test_name}",
                    response_time_ms=response_time,
                    success=success,
                    from_cache=from_cache,
                    error_message=result.stderr if not success else None,
                    metadata={'command': ' '.join(cmd), 'stdout': result.stdout}
                ))
                
                logger.info(f"  {test_name}: {response_time:.1f}ms {'✓' if success else '✗'}")
                
            except subprocess.TimeoutExpired:
                results.append(PerformanceTestResult(
                    test_name=f"cli_{test_name}",
                    response_time_ms=self.test_timeout * 1000,
                    success=False,
                    error_message="Command timed out",
                    metadata={'command': ' '.join(cmd)}
                ))
                logger.error(f"  {test_name}: TIMEOUT")
                
            except Exception as e:
                results.append(PerformanceTestResult(
                    test_name=f"cli_{test_name}",
                    response_time_ms=0,
                    success=False,
                    error_message=str(e),
                    metadata={'command': ' '.join(cmd)}
                ))
                logger.error(f"  {test_name}: ERROR - {e}")
        
        return results
    
    async def test_cache_performance(self) -> List[PerformanceTestResult]:
        """Test caching system performance"""
        
        logger.info("Testing cache performance...")
        results = []
        
        try:
            # Import cache modules
            sys.path.insert(0, str(self.neuralsync_dir))
            from neuralsync.intelligent_cache import get_neuralsync_cache
            
            cache = get_neuralsync_cache()
            
            # Test cache operations
            test_data = {
                'small_text': 'Hello, world!',
                'medium_text': 'Lorem ipsum ' * 100,
                'large_text': 'Big data ' * 1000,
                'json_data': {'key': 'value', 'numbers': list(range(100))}
            }
            
            for data_name, data in test_data.items():
                # Test cache set operation
                with self.timer() as get_time:
                    success = await cache.persona_cache.set(f"test_{data_name}", data, 60000)
                
                set_time = get_time()
                results.append(PerformanceTestResult(
                    test_name=f"cache_set_{data_name}",
                    response_time_ms=set_time,
                    success=success,
                    context_size_bytes=len(str(data)),
                    metadata={'operation': 'set', 'data_type': data_name}
                ))
                
                # Test cache get operation
                with self.timer() as get_time:
                    retrieved = await cache.persona_cache.get(f"test_{data_name}")
                
                get_time_ms = get_time()
                get_success = retrieved is not None
                
                results.append(PerformanceTestResult(
                    test_name=f"cache_get_{data_name}",
                    response_time_ms=get_time_ms,
                    success=get_success,
                    from_cache=True,
                    context_size_bytes=len(str(retrieved)) if retrieved else 0,
                    metadata={'operation': 'get', 'data_type': data_name}
                ))
                
                logger.info(f"  {data_name}: set {set_time:.1f}ms, get {get_time_ms:.1f}ms")
            
            # Test cache statistics
            cache_stats = cache.get_comprehensive_stats()
            logger.info(f"  Cache hit rates: {cache_stats}")
            
        except Exception as e:
            logger.error(f"Cache performance test failed: {e}")
            results.append(PerformanceTestResult(
                test_name="cache_performance",
                response_time_ms=0,
                success=False,
                error_message=str(e)
            ))
        
        return results
    
    async def test_async_network_performance(self) -> List[PerformanceTestResult]:
        """Test async network operations performance"""
        
        logger.info("Testing async network performance...")
        results = []
        
        try:
            sys.path.insert(0, str(self.neuralsync_dir))
            from neuralsync.async_network import get_network_client
            
            # Test with both fast and normal modes
            for fast_mode in [True, False]:
                client = get_network_client(fast_mode=fast_mode)
                mode_name = 'fast' if fast_mode else 'normal'
                
                # Test health check
                with self.timer() as get_time:
                    health_ok = await client.health_check()
                
                health_time = get_time()
                results.append(PerformanceTestResult(
                    test_name=f"network_health_{mode_name}",
                    response_time_ms=health_time,
                    success=health_ok,
                    metadata={'mode': mode_name, 'operation': 'health_check'}
                ))
                
                # Test context fetching (if service is available)
                if health_ok:
                    with self.timer() as get_time:
                        context = await client.get_context_fast(tool='test-tool', query='test query')
                    
                    context_time = get_time()
                    results.append(PerformanceTestResult(
                        test_name=f"network_context_{mode_name}",
                        response_time_ms=context_time,
                        success=True,
                        context_size_bytes=len(context) if context else 0,
                        metadata={'mode': mode_name, 'operation': 'get_context'}
                    ))
                    
                    logger.info(f"  {mode_name} mode: health {health_time:.1f}ms, context {context_time:.1f}ms")
                else:
                    logger.warning(f"  {mode_name} mode: health check failed, skipping context test")
                
                await client.close()
                
        except Exception as e:
            logger.error(f"Async network test failed: {e}")
            results.append(PerformanceTestResult(
                test_name="async_network_performance",
                response_time_ms=0,
                success=False,
                error_message=str(e)
            ))
        
        return results
    
    async def test_concurrent_operations(self) -> List[PerformanceTestResult]:
        """Test concurrent CLI operations"""
        
        logger.info("Testing concurrent operations...")
        results = []
        
        try:
            # Define concurrent test commands
            commands = [
                ['echo', f'test_{i}'] for i in range(5)
            ] + [
                ['pwd'],
                ['date'],
                ['python3', '-c', 'print("concurrent")']
            ]
            
            # Run commands concurrently
            with self.timer() as get_time:
                tasks = []
                for i, cmd in enumerate(commands):
                    task = self._run_cli_command_async(cmd, f'concurrent_{i}')
                    tasks.append(task)
                
                # Wait for all tasks to complete
                task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = get_time()
            
            # Process results
            successful_tasks = 0
            total_response_time = 0
            
            for i, task_result in enumerate(task_results):
                if isinstance(task_result, Exception):
                    results.append(PerformanceTestResult(
                        test_name=f"concurrent_task_{i}",
                        response_time_ms=0,
                        success=False,
                        error_message=str(task_result)
                    ))
                elif isinstance(task_result, PerformanceTestResult):
                    results.append(task_result)
                    if task_result.success:
                        successful_tasks += 1
                        total_response_time += task_result.response_time_ms
            
            # Calculate concurrent efficiency
            avg_response_time = total_response_time / max(1, successful_tasks)
            concurrency_efficiency = (len(commands) * avg_response_time) / max(total_time, 1)
            
            results.append(PerformanceTestResult(
                test_name="concurrent_operations_summary",
                response_time_ms=total_time,
                success=successful_tasks > 0,
                metadata={
                    'total_tasks': len(commands),
                    'successful_tasks': successful_tasks,
                    'avg_task_time_ms': avg_response_time,
                    'concurrency_efficiency': concurrency_efficiency
                }
            ))
            
            logger.info(f"  Concurrent test: {successful_tasks}/{len(commands)} tasks in {total_time:.1f}ms")
            logger.info(f"  Average task time: {avg_response_time:.1f}ms")
            logger.info(f"  Concurrency efficiency: {concurrency_efficiency:.2f}x")
            
        except Exception as e:
            logger.error(f"Concurrent operations test failed: {e}")
            results.append(PerformanceTestResult(
                test_name="concurrent_operations",
                response_time_ms=0,
                success=False,
                error_message=str(e)
            ))
        
        return results
    
    async def _run_cli_command_async(self, cmd: List[str], test_name: str) -> PerformanceTestResult:
        """Run CLI command asynchronously"""
        
        try:
            with self.timer() as get_time:
                proc = await asyncio.create_subprocess_exec(
                    str(self.nswrap_path), '--', *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), 
                    timeout=self.test_timeout
                )
            
            response_time = get_time()
            success = proc.returncode == 0
            
            return PerformanceTestResult(
                test_name=test_name,
                response_time_ms=response_time,
                success=success,
                error_message=stderr.decode() if stderr and not success else None,
                metadata={
                    'command': ' '.join(cmd),
                    'stdout': stdout.decode() if stdout else ''
                }
            )
            
        except asyncio.TimeoutError:
            return PerformanceTestResult(
                test_name=test_name,
                response_time_ms=self.test_timeout * 1000,
                success=False,
                error_message="Command timed out"
            )
        except Exception as e:
            return PerformanceTestResult(
                test_name=test_name,
                response_time_ms=0,
                success=False,
                error_message=str(e)
            )
    
    async def test_optimization_comparison(self) -> List[PerformanceTestResult]:
        """Compare optimized vs original nswrap performance"""
        
        logger.info("Testing optimization comparison...")
        results = []
        
        # Test commands for comparison
        test_commands = [
            ['echo', 'benchmark'],
            ['python3', '-c', 'print("test")'],
            ['pwd']
        ]
        
        for cmd in test_commands:
            cmd_name = '_'.join(cmd[:2])
            
            # Test optimized version
            try:
                with self.timer() as get_time:
                    result_optimized = subprocess.run([
                        str(self.nswrap_path), '--'
                    ] + cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.test_timeout
                    )
                
                optimized_time = get_time()
                optimized_success = result_optimized.returncode == 0
                
                results.append(PerformanceTestResult(
                    test_name=f"optimized_{cmd_name}",
                    response_time_ms=optimized_time,
                    success=optimized_success,
                    metadata={'version': 'optimized', 'command': ' '.join(cmd)}
                ))
                
            except Exception as e:
                results.append(PerformanceTestResult(
                    test_name=f"optimized_{cmd_name}",
                    response_time_ms=0,
                    success=False,
                    error_message=str(e)
                ))
                continue
            
            # Test original version (if available)
            if self.original_nswrap.exists():
                try:
                    with self.timer() as get_time:
                        result_original = subprocess.run([
                            str(self.original_nswrap), '--'
                        ] + cmd,
                        capture_output=True,
                        text=True,
                        timeout=self.test_timeout
                        )
                    
                    original_time = get_time()
                    original_success = result_original.returncode == 0
                    
                    results.append(PerformanceTestResult(
                        test_name=f"original_{cmd_name}",
                        response_time_ms=original_time,
                        success=original_success,
                        metadata={'version': 'original', 'command': ' '.join(cmd)}
                    ))
                    
                    # Calculate improvement
                    if optimized_success and original_success:
                        improvement = ((original_time - optimized_time) / original_time) * 100
                        logger.info(f"  {cmd_name}: {optimized_time:.1f}ms vs {original_time:.1f}ms ({improvement:+.1f}% improvement)")
                    
                except Exception as e:
                    results.append(PerformanceTestResult(
                        test_name=f"original_{cmd_name}",
                        response_time_ms=0,
                        success=False,
                        error_message=str(e)
                    ))
            else:
                logger.info(f"  {cmd_name}: {optimized_time:.1f}ms (no original for comparison)")
        
        return results
    
    async def run_comprehensive_test_suite(self) -> TestSuiteResults:
        """Run the complete test suite"""
        
        logger.info("🚀 Starting NeuralSync v2 Performance Test Suite")
        start_time = time.perf_counter()
        
        all_results = []
        
        # Run all test categories
        test_methods = [
            self.test_cli_response_times,
            self.test_cache_performance,
            self.test_async_network_performance,
            self.test_concurrent_operations,
            self.test_optimization_comparison
        ]
        
        for test_method in test_methods:
            try:
                category_results = await test_method()
                all_results.extend(category_results)
            except Exception as e:
                logger.error(f"Test method {test_method.__name__} failed: {e}")
                all_results.append(PerformanceTestResult(
                    test_name=test_method.__name__,
                    response_time_ms=0,
                    success=False,
                    error_message=str(e)
                ))
        
        # Calculate summary statistics
        successful_results = [r for r in all_results if r.success]
        failed_results = [r for r in all_results if not r.success]
        
        response_times = [r.response_time_ms for r in successful_results if r.response_time_ms > 0]
        cached_results = [r for r in successful_results if r.from_cache]
        
        # Performance statistics
        if response_times:
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times)
            p99_response_time = statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else max(response_times)
        else:
            avg_response_time = median_response_time = p95_response_time = p99_response_time = 0
        
        # Performance buckets
        sub_100ms = len([t for t in response_times if t < 100])
        sub_500ms = len([t for t in response_times if t < 500])
        sub_1000ms = len([t for t in response_times if t < 1000])
        
        # Cache hit rate
        cache_hit_rate = len(cached_results) / len(successful_results) if successful_results else 0
        
        # Health score calculation
        health_score = self._calculate_optimization_health_score(
            avg_response_time, len(successful_results), len(all_results), cache_hit_rate
        )
        
        suite_results = TestSuiteResults(
            total_tests=len(all_results),
            successful_tests=len(successful_results),
            failed_tests=len(failed_results),
            average_response_time_ms=avg_response_time,
            median_response_time_ms=median_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            sub_second_tests=sub_1000ms,
            sub_500ms_tests=sub_500ms,
            sub_100ms_tests=sub_100ms,
            cache_hit_rate=cache_hit_rate,
            test_results=all_results,
            optimization_health_score=health_score
        )
        
        total_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"✅ Test suite completed in {total_time:.1f}ms")
        
        return suite_results
    
    def _calculate_optimization_health_score(self, 
                                           avg_response_time: float, 
                                           successful_tests: int, 
                                           total_tests: int, 
                                           cache_hit_rate: float) -> float:
        """Calculate optimization health score (0.0 to 1.0)"""
        
        scores = []
        
        # Response time score
        if avg_response_time <= self.performance_targets['excellent_ms']:
            scores.append(1.0)
        elif avg_response_time <= self.performance_targets['good_ms']:
            scores.append(0.8)
        elif avg_response_time <= self.performance_targets['acceptable_ms']:
            scores.append(0.6)
        else:
            scores.append(0.3)
        
        # Success rate score
        success_rate = successful_tests / max(1, total_tests)
        if success_rate >= self.performance_targets['target_success_rate']:
            scores.append(1.0)
        elif success_rate >= 0.8:
            scores.append(0.7)
        elif success_rate >= 0.6:
            scores.append(0.4)
        else:
            scores.append(0.2)
        
        # Cache hit rate score
        if cache_hit_rate >= self.performance_targets['target_cache_hit_rate']:
            scores.append(1.0)
        elif cache_hit_rate >= 0.5:
            scores.append(0.7)
        elif cache_hit_rate >= 0.3:
            scores.append(0.5)
        else:
            scores.append(0.3)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def generate_performance_report(self, results: TestSuiteResults, output_file: str = None) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        report = {
            'test_suite_info': {
                'timestamp': time.time(),
                'neuralsync_version': '2.0.0',
                'test_suite_version': '1.0.0',
                'total_tests': results.total_tests,
                'test_duration_ms': sum(r.response_time_ms for r in results.test_results if r.success)
            },
            'performance_summary': {
                'optimization_health_score': results.optimization_health_score,
                'performance_grade': self._get_performance_grade(results.optimization_health_score),
                'successful_tests': results.successful_tests,
                'failed_tests': results.failed_tests,
                'success_rate': results.successful_tests / max(1, results.total_tests)
            },
            'response_time_analysis': {
                'average_ms': results.average_response_time_ms,
                'median_ms': results.median_response_time_ms,
                'p95_ms': results.p95_response_time_ms,
                'p99_ms': results.p99_response_time_ms,
                'performance_buckets': {
                    'sub_100ms': results.sub_100ms_tests,
                    'sub_500ms': results.sub_500ms_tests,
                    'sub_1000ms': results.sub_second_tests,
                    'over_1000ms': results.total_tests - results.sub_second_tests
                }
            },
            'optimization_metrics': {
                'cache_hit_rate': results.cache_hit_rate,
                'sub_second_achievement': results.sub_second_tests / max(1, results.successful_tests),
                'performance_target_achievement': {
                    'excellent_responses': results.sub_100ms_tests / max(1, results.successful_tests),
                    'good_responses': results.sub_500ms_tests / max(1, results.successful_tests),
                    'acceptable_responses': results.sub_second_tests / max(1, results.successful_tests)
                }
            },
            'detailed_results': [
                {
                    'test_name': r.test_name,
                    'response_time_ms': r.response_time_ms,
                    'success': r.success,
                    'from_cache': r.from_cache,
                    'context_size_bytes': r.context_size_bytes,
                    'error_message': r.error_message,
                    'metadata': r.metadata
                }
                for r in results.test_results
            ],
            'recommendations': self._generate_performance_recommendations(results),
            'performance_targets': self.performance_targets
        }
        
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Performance report saved to {output_file}")
            except Exception as e:
                logger.error(f"Could not save performance report: {e}")
        
        return report
    
    def _get_performance_grade(self, health_score: float) -> str:
        """Get performance grade from health score"""
        if health_score >= 0.9:
            return 'A+ (Excellent)'
        elif health_score >= 0.8:
            return 'A (Very Good)'
        elif health_score >= 0.7:
            return 'B (Good)'
        elif health_score >= 0.6:
            return 'C (Acceptable)'
        elif health_score >= 0.5:
            return 'D (Needs Improvement)'
        else:
            return 'F (Poor)'
    
    def _generate_performance_recommendations(self, results: TestSuiteResults) -> List[str]:
        """Generate performance improvement recommendations"""
        
        recommendations = []
        
        # Response time recommendations
        if results.average_response_time_ms > self.performance_targets['acceptable_ms']:
            recommendations.append(
                "Average response time exceeds acceptable threshold - enable bypass mode for simple commands"
            )
        
        if results.sub_100ms_tests < results.successful_tests * 0.3:
            recommendations.append(
                "Less than 30% of responses are sub-100ms - consider increasing cache TTL and prewarming"
            )
        
        # Cache recommendations
        if results.cache_hit_rate < self.performance_targets['target_cache_hit_rate']:
            recommendations.append(
                f"Cache hit rate ({results.cache_hit_rate:.1%}) is below target - enable prewarming service"
            )
        
        # Success rate recommendations
        success_rate = results.successful_tests / max(1, results.total_tests)
        if success_rate < self.performance_targets['target_success_rate']:
            recommendations.append(
                f"Success rate ({success_rate:.1%}) is below target - check for system configuration issues"
            )
        
        # Specific performance recommendations
        if results.optimization_health_score < 0.7:
            recommendations.append(
                "Overall optimization health is low - run performance optimization installer"
            )
        
        # Positive reinforcement
        if results.optimization_health_score >= 0.9:
            recommendations.append(
                "Excellent performance! System is well optimized for sub-second response times"
            )
        
        return recommendations
    
    def print_performance_summary(self, results: TestSuiteResults):
        """Print formatted performance summary"""
        
        print("\n" + "="*80)
        print("🎯 NEURALSYNC v2 PERFORMANCE TEST RESULTS")
        print("="*80)
        
        # Overall health
        grade = self._get_performance_grade(results.optimization_health_score)
        print(f"\n📊 OVERALL PERFORMANCE GRADE: {grade}")
        print(f"   Health Score: {results.optimization_health_score:.2f}/1.00")
        
        # Test statistics
        print(f"\n📈 TEST STATISTICS:")
        print(f"   Total Tests: {results.total_tests}")
        print(f"   Successful: {results.successful_tests} ({results.successful_tests/results.total_tests*100:.1f}%)")
        print(f"   Failed: {results.failed_tests} ({results.failed_tests/results.total_tests*100:.1f}%)")
        
        # Response time analysis
        print(f"\n⚡ RESPONSE TIME ANALYSIS:")
        print(f"   Average: {results.average_response_time_ms:.1f}ms")
        print(f"   Median:  {results.median_response_time_ms:.1f}ms")
        print(f"   P95:     {results.p95_response_time_ms:.1f}ms")
        print(f"   P99:     {results.p99_response_time_ms:.1f}ms")
        
        # Performance buckets
        print(f"\n🎯 PERFORMANCE TARGETS:")
        total_successful = results.successful_tests
        if total_successful > 0:
            print(f"   Sub-100ms:  {results.sub_100ms_tests}/{total_successful} ({results.sub_100ms_tests/total_successful*100:.1f}%)")
            print(f"   Sub-500ms:  {results.sub_500ms_tests}/{total_successful} ({results.sub_500ms_tests/total_successful*100:.1f}%)")
            print(f"   Sub-1000ms: {results.sub_second_tests}/{total_successful} ({results.sub_second_tests/total_successful*100:.1f}%)")
        
        # Cache performance
        print(f"\n💾 CACHE PERFORMANCE:")
        print(f"   Hit Rate: {results.cache_hit_rate:.1%}")
        cache_results = [r for r in results.test_results if r.from_cache]
        if cache_results:
            avg_cache_time = statistics.mean([r.response_time_ms for r in cache_results])
            print(f"   Avg Cache Response: {avg_cache_time:.1f}ms")
        
        # Sub-second achievement
        sub_second_rate = results.sub_second_tests / max(1, results.successful_tests)
        print(f"\n🚀 SUB-SECOND ACHIEVEMENT: {sub_second_rate:.1%}")
        
        if sub_second_rate >= 0.95:
            print("   ✅ EXCELLENT: 95%+ responses are sub-second!")
        elif sub_second_rate >= 0.8:
            print("   ✅ GOOD: 80%+ responses are sub-second")
        elif sub_second_rate >= 0.6:
            print("   ⚠️  ACCEPTABLE: 60%+ responses are sub-second")
        else:
            print("   ❌ NEEDS IMPROVEMENT: Less than 60% responses are sub-second")
        
        # Recommendations
        recommendations = self._generate_performance_recommendations(results)
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "="*80)
        
        # Final assessment
        if results.optimization_health_score >= 0.8 and sub_second_rate >= 0.8:
            print("🎉 SUCCESS: NeuralSync v2 optimizations are working excellently!")
            print("   Your CLI should provide consistently fast, sub-second responses.")
        elif results.optimization_health_score >= 0.6:
            print("✅ GOOD: Performance optimizations are working well with room for improvement.")
        else:
            print("❌ NEEDS WORK: Performance optimizations need attention to achieve sub-second goals.")
        
        print("="*80 + "\n")


async def main():
    """Main test runner"""
    
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="NeuralSync v2 Performance Test Suite")
    parser.add_argument('--neuralsync-dir', default=None, help='NeuralSync directory path')
    parser.add_argument('--output-file', default=None, help='Output file for detailed results')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine NeuralSync directory
    neuralsync_dir = Path(args.neuralsync_dir) if args.neuralsync_dir else Path(__file__).parent
    
    # Initialize test suite
    test_suite = PerformanceTestSuite(neuralsync_dir)
    
    # Run tests
    try:
        logger.info("Starting comprehensive performance test suite...")
        results = await test_suite.run_comprehensive_test_suite()
        
        # Generate and print report
        report = test_suite.generate_performance_report(results, args.output_file)
        test_suite.print_performance_summary(results)
        
        # Exit code based on results
        if results.optimization_health_score >= 0.7 and results.sub_second_tests >= results.successful_tests * 0.8:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Performance targets not met
            
    except KeyboardInterrupt:
        logger.info("Test suite cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        sys.exit(2)

if __name__ == "__main__":
    import os
    os.environ.setdefault('PYTHONPATH', str(Path(__file__).parent))
    asyncio.run(main())