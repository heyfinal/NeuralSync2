#!/usr/bin/env python3
"""
Async Network Operations for NeuralSync v2
High-performance parallel network calls with intelligent caching and fallbacks
"""

import asyncio
import aiohttp
import time
import logging
import json
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from urllib.parse import urljoin
import ssl
import certifi

from .intelligent_cache import get_neuralsync_cache
from .utils import now_ms

logger = logging.getLogger(__name__)

@dataclass
class NetworkRequest:
    """Network request specification"""
    method: str
    url: str
    headers: Optional[Dict[str, str]] = None
    data: Optional[Dict[str, Any]] = None
    timeout: float = 2.0
    cache_ttl_ms: Optional[int] = None
    cache_key: Optional[str] = None
    retry_count: int = 2
    priority: int = 1  # Higher = more important

@dataclass
class NetworkResponse:
    """Network response with metadata"""
    status_code: int
    data: Any
    headers: Dict[str, str]
    response_time_ms: float
    from_cache: bool = False
    error: Optional[str] = None
    request_id: str = ""

class AsyncNetworkManager:
    """High-performance async network manager with intelligent caching"""
    
    def __init__(self, 
                 max_concurrent: int = 10,
                 connection_pool_size: int = 20,
                 total_timeout: float = 5.0):
        
        self.max_concurrent = max_concurrent
        self.connection_pool_size = connection_pool_size
        self.total_timeout = total_timeout
        
        # Connection management
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'avg_response_time_ms': 0.0,
            'parallel_requests_made': 0
        }
        
        # Cache integration
        self.cache = get_neuralsync_cache()
        
        logger.info(f"AsyncNetworkManager initialized: max_concurrent={max_concurrent}")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with optimized settings"""
        if self._session is None or self._session.closed:
            # SSL context for secure connections
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            
            # Connection configuration for performance
            connector = aiohttp.TCPConnector(
                limit=self.connection_pool_size,
                limit_per_host=10,
                ttl_dns_cache=300,  # 5 minute DNS cache
                use_dns_cache=True,
                ssl=ssl_context,
                enable_cleanup_closed=True,
                keepalive_timeout=30
            )
            
            # Request timeout configuration
            timeout = aiohttp.ClientTimeout(
                total=self.total_timeout,
                connect=1.0,
                sock_read=2.0
            )
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'User-Agent': 'NeuralSync/2.0'},
                raise_for_status=False  # Handle errors manually
            )
        
        return self._session
    
    async def single_request(self, request: NetworkRequest) -> NetworkResponse:
        """Execute single network request with caching and retry logic"""
        
        request_id = hashlib.md5(f"{request.method}:{request.url}:{request.data}".encode()).hexdigest()[:8]
        start_time = time.perf_counter()
        
        # Check cache first if cache_key provided
        if request.cache_key:
            try:
                cached_data = await self._get_from_cache(request.cache_key)
                if cached_data is not None:
                    self.stats['cache_hits'] += 1
                    return NetworkResponse(
                        status_code=200,
                        data=cached_data,
                        headers={},
                        response_time_ms=(time.perf_counter() - start_time) * 1000,
                        from_cache=True,
                        request_id=request_id
                    )
            except Exception as e:
                logger.debug(f"Cache lookup failed for {request_id}: {e}")
        
        # Execute request with retry logic
        last_error = None
        
        for attempt in range(request.retry_count + 1):
            try:
                async with self._semaphore:
                    session = await self._get_session()
                    
                    # Prepare request parameters
                    kwargs = {
                        'headers': request.headers or {},
                        'timeout': aiohttp.ClientTimeout(total=request.timeout)
                    }
                    
                    if request.data:
                        if request.method.upper() in ['POST', 'PUT', 'PATCH']:
                            kwargs['json'] = request.data
                        else:
                            kwargs['params'] = request.data
                    
                    # Execute request
                    async with session.request(request.method, request.url, **kwargs) as response:
                        response_time_ms = (time.perf_counter() - start_time) * 1000
                        
                        # Read response data
                        try:
                            if response.content_type == 'application/json':
                                data = await response.json()
                            else:
                                data = await response.text()
                        except Exception as e:
                            logger.warning(f"Response parsing failed for {request_id}: {e}")
                            data = await response.text()
                        
                        # Create response object
                        network_response = NetworkResponse(
                            status_code=response.status,
                            data=data,
                            headers=dict(response.headers),
                            response_time_ms=response_time_ms,
                            request_id=request_id
                        )
                        
                        # Cache successful responses
                        if (response.status == 200 and 
                            request.cache_key and 
                            request.cache_ttl_ms):
                            try:
                                await self._cache_response(request.cache_key, data, request.cache_ttl_ms)
                            except Exception as e:
                                logger.debug(f"Response caching failed for {request_id}: {e}")
                        
                        # Update statistics
                        self.stats['total_requests'] += 1
                        if response.status == 200:
                            self.stats['successful_requests'] += 1
                        else:
                            self.stats['failed_requests'] += 1
                        
                        # Update average response time
                        self.stats['avg_response_time_ms'] = (
                            0.9 * self.stats['avg_response_time_ms'] + 0.1 * response_time_ms
                        )
                        
                        return network_response
                        
            except asyncio.TimeoutError as e:
                last_error = f"Request timeout after {request.timeout}s"
                logger.warning(f"Request timeout for {request_id}, attempt {attempt + 1}")
                
            except aiohttp.ClientError as e:
                last_error = f"Client error: {str(e)}"
                logger.warning(f"Client error for {request_id}, attempt {attempt + 1}: {e}")
                
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error(f"Unexpected error for {request_id}, attempt {attempt + 1}: {e}")
            
            # Wait before retry (exponential backoff)
            if attempt < request.retry_count:
                wait_time = min(2.0, 0.1 * (2 ** attempt))
                await asyncio.sleep(wait_time)
        
        # All attempts failed
        self.stats['total_requests'] += 1
        self.stats['failed_requests'] += 1
        
        return NetworkResponse(
            status_code=0,
            data=None,
            headers={},
            response_time_ms=(time.perf_counter() - start_time) * 1000,
            error=last_error,
            request_id=request_id
        )
    
    async def parallel_requests(self, requests: List[NetworkRequest]) -> List[NetworkResponse]:
        """Execute multiple requests in parallel with intelligent prioritization"""
        
        if not requests:
            return []
        
        start_time = time.perf_counter()
        
        # Sort requests by priority (higher priority first)
        sorted_requests = sorted(requests, key=lambda r: r.priority, reverse=True)
        
        # Create tasks for parallel execution
        tasks = []
        for request in sorted_requests:
            task = asyncio.create_task(self.single_request(request))
            tasks.append(task)
        
        # Execute all requests in parallel
        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            network_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.error(f"Parallel request {i} failed: {response}")
                    network_responses.append(NetworkResponse(
                        status_code=0,
                        data=None,
                        headers={},
                        response_time_ms=0,
                        error=str(response),
                        request_id=f"parallel_{i}"
                    ))
                else:
                    network_responses.append(response)
            
            # Update parallel request statistics
            self.stats['parallel_requests_made'] += len(requests)
            total_time_ms = (time.perf_counter() - start_time) * 1000
            
            logger.debug(f"Parallel requests completed: {len(requests)} requests in {total_time_ms:.2f}ms")
            
            return network_responses
            
        except Exception as e:
            logger.error(f"Parallel requests failed: {e}")
            # Return error responses for all requests
            return [
                NetworkResponse(
                    status_code=0,
                    data=None,
                    headers={},
                    response_time_ms=0,
                    error=str(e),
                    request_id=f"parallel_error_{i}"
                )
                for i in range(len(requests))
            ]
    
    async def _get_from_cache(self, cache_key: str) -> Any:
        """Get data from cache"""
        # Try different cache types based on key prefix
        if cache_key.startswith('persona:'):
            return await self.cache.persona_cache.get(cache_key)
        elif cache_key.startswith('recall:'):
            return await self.cache.memory_cache.get(cache_key)
        elif cache_key.startswith('context:'):
            return await self.cache.context_cache.get(cache_key)
        else:
            # Use memory cache as default
            return await self.cache.memory_cache.get(cache_key)
    
    async def _cache_response(self, cache_key: str, data: Any, ttl_ms: int):
        """Cache response data"""
        if cache_key.startswith('persona:'):
            await self.cache.persona_cache.set(cache_key, data, ttl_ms)
        elif cache_key.startswith('recall:'):
            await self.cache.memory_cache.set(cache_key, data, ttl_ms)
        elif cache_key.startswith('context:'):
            await self.cache.context_cache.set(cache_key, data, ttl_ms)
        else:
            await self.cache.memory_cache.set(cache_key, data, ttl_ms)
    
    async def health_check(self, endpoint: str, timeout: float = 1.0) -> bool:
        """Fast health check for endpoint"""
        request = NetworkRequest(
            method='GET',
            url=urljoin(endpoint, '/health'),
            timeout=timeout,
            retry_count=0
        )
        
        response = await self.single_request(request)
        return response.status_code == 200
    
    def get_stats(self) -> Dict[str, Any]:
        """Get network performance statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'avg_response_time_ms': 0.0,
            'parallel_requests_made': 0
        }
    
    async def close(self):
        """Clean shutdown"""
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("AsyncNetworkManager closed")


class NeuralSyncNetworkClient:
    """Specialized network client for NeuralSync operations"""
    
    def __init__(self, 
                 base_url: str = "http://127.0.0.1:8373",
                 auth_token: Optional[str] = None,
                 enable_fast_mode: bool = False):
        
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.enable_fast_mode = enable_fast_mode
        
        # Network manager
        self.network_manager = AsyncNetworkManager(
            max_concurrent=5 if enable_fast_mode else 10,
            total_timeout=2.0 if enable_fast_mode else 5.0
        )
        
        # Cache instance
        self.cache = get_neuralsync_cache()
        
        # Headers
        self.default_headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        if auth_token:
            self.default_headers['Authorization'] = f'Bearer {auth_token}'
    
    async def get_persona_and_memories(self, 
                                     tool: Optional[str] = None,
                                     query: str = "",
                                     top_k: int = 3,
                                     scope: str = "any") -> Tuple[Optional[str], Optional[List[Dict]]]:
        """Fetch persona and memories in parallel with intelligent caching"""
        
        # Generate cache keys
        persona_key = "persona:current"
        query_hash = self.cache.hash_query(query, tool, scope, top_k)
        memory_key = f"recall:{query_hash}"
        
        # Create parallel requests
        requests = []
        
        # Persona request
        persona_request = NetworkRequest(
            method='GET',
            url=f"{self.base_url}/persona",
            headers=self.default_headers.copy(),
            timeout=1.0 if self.enable_fast_mode else 2.0,
            cache_key=persona_key,
            cache_ttl_ms=600000,  # 10 minutes
            priority=2  # Higher priority
        )
        requests.append(persona_request)
        
        # Memory recall request  
        memory_request = NetworkRequest(
            method='POST',
            url=f"{self.base_url}/recall",
            headers=self.default_headers.copy(),
            data={
                "query": query,
                "top_k": top_k,
                "scope": scope,
                "tool": tool
            },
            timeout=2.0 if self.enable_fast_mode else 3.0,
            cache_key=memory_key,
            cache_ttl_ms=300000,  # 5 minutes
            priority=1  # Lower priority
        )
        requests.append(memory_request)
        
        # Execute in parallel
        responses = await self.network_manager.parallel_requests(requests)
        
        # Process responses
        persona = None
        memories = None
        
        if len(responses) >= 1 and responses[0].status_code == 200:
            persona_data = responses[0].data
            if isinstance(persona_data, dict):
                persona = persona_data.get('text', '')
            
        if len(responses) >= 2 and responses[1].status_code == 200:
            memory_data = responses[1].data
            if isinstance(memory_data, dict):
                memories = memory_data.get('items', [])
        
        return persona, memories
    
    async def get_context_fast(self, tool: Optional[str] = None, query: str = "") -> str:
        """Get assembled context with maximum speed optimizations"""
        
        # Try cached context first
        context_hash = self.cache.hash_context(query, tool)
        context_key = f"context:{context_hash}"
        
        cached_context = await self.cache.get_context(context_hash)
        if cached_context:
            return cached_context
        
        # Fetch data in parallel
        persona, memories = await self.get_persona_and_memories(tool, query, top_k=3)
        
        # Assemble context
        context_parts = []
        
        if persona and not self.enable_fast_mode:
            context_parts.append(f"Persona: {persona}")
            context_parts.append("")
        
        if memories:
            for i, memory in enumerate(memories, 1):
                context_line = f"[M{i}] ({memory.get('kind', 'unknown')},{memory.get('scope', 'global')},conf={memory.get('confidence', '')})"
                context_line += f" {memory.get('text', '')}"
                context_parts.append(context_line)
        
        assembled_context = "\n".join(context_parts)
        if assembled_context:
            assembled_context += "\n\n"
        
        # Cache the assembled context
        await self.cache.set_context(context_hash, assembled_context, 180000)  # 3 minutes
        
        return assembled_context
    
    async def health_check(self) -> bool:
        """Fast health check"""
        return await self.network_manager.health_check(self.base_url)
    
    async def send_remember(self, text: str, kind: str = "note", scope: str = "global", 
                          tool: Optional[str] = None, confidence: float = 0.8) -> bool:
        """Send remember command asynchronously"""
        
        request = NetworkRequest(
            method='POST',
            url=f"{self.base_url}/remember",
            headers=self.default_headers.copy(),
            data={
                "text": text,
                "kind": kind,
                "scope": scope,
                "tool": tool,
                "confidence": confidence,
                "source": "nswrap"
            },
            timeout=3.0,
            retry_count=1
        )
        
        response = await self.network_manager.single_request(request)
        
        if response.status_code == 200:
            # Invalidate relevant caches
            await self.cache.memory_cache.invalidate_pattern("recall:")
            await self.cache.context_cache.invalidate_pattern("context:")
            return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            'network_stats': self.network_manager.get_stats(),
            'cache_stats': self.cache.get_comprehensive_stats(),
            'fast_mode_enabled': self.enable_fast_mode,
            'base_url': self.base_url
        }
    
    async def close(self):
        """Clean shutdown"""
        await self.network_manager.close()
        self.cache.close()


# Global client instance
_global_client: Optional[NeuralSyncNetworkClient] = None

def get_network_client(base_url: str = None, 
                      auth_token: str = None, 
                      fast_mode: bool = False) -> NeuralSyncNetworkClient:
    """Get global network client instance"""
    global _global_client
    if _global_client is None:
        import os
        
        if base_url is None:
            ns_host = os.environ.get('NS_HOST', '127.0.0.1')
            ns_port = os.environ.get('NS_PORT', '8373')
            base_url = f"http://{ns_host}:{ns_port}"
        
        if auth_token is None:
            auth_token = os.environ.get('NS_TOKEN')
        
        _global_client = NeuralSyncNetworkClient(
            base_url=base_url,
            auth_token=auth_token,
            enable_fast_mode=fast_mode
        )
    
    return _global_client

async def fetch_context_optimized(tool: Optional[str] = None, query: str = "") -> str:
    """Optimized context fetching with all performance enhancements"""
    client = get_network_client(fast_mode=True)
    return await client.get_context_fast(tool, query)

async def ensure_service_available(base_url: str = None) -> bool:
    """Fast service availability check"""
    client = get_network_client(base_url=base_url, fast_mode=True)
    return await client.health_check()