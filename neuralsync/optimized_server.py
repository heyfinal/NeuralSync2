#!/usr/bin/env python3
"""
Optimized NeuralSync Server v2
High-performance server with advanced caching, async operations, and monitoring
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .config import load_config
from .storage import connect
from .api import MemoryIn, RecallIn, PersonaIn
from .utils import now_ms, redact
from .auth import bearer_guard
from .intelligent_cache import get_neuralsync_cache
from .fast_recall import get_fast_recall_engine, fast_recall
from .performance_monitor import get_performance_monitor
from .cli_performance_integration import get_performance_integration

logger = logging.getLogger(__name__)

class OptimizedNeuralSyncServer:
    """High-performance NeuralSync server with comprehensive optimizations"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.cfg = load_config(config_path)
        
        # Initialize storage
        self.storage = connect(self.cfg.db_path)
        
        # Initialize performance components
        self.cache = get_neuralsync_cache()
        self.fast_recall_engine = get_fast_recall_engine(self.storage)
        self.perf_monitor = get_performance_monitor(self.storage)
        self.perf_integration = get_performance_integration(self.storage)
        
        # Create FastAPI app
        self.app = FastAPI(
            title='NeuralSync v2 - Optimized',
            version='2.0.0',
            description='High-performance neural synchronization with sub-second response times'
        )
        
        # Add middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Request tracking
        self.request_counter = 0
        
        # Setup routes
        self._setup_routes()
        
        # Background tasks
        self.background_tasks_enabled = True
        
        logger.info("OptimizedNeuralSyncServer initialized")
    
    def _setup_routes(self):
        """Setup all API routes with optimizations"""
        
        @self.app.middleware("http")
        async def performance_middleware(request, call_next):
            """Performance monitoring middleware"""
            start_time = time.perf_counter()
            
            # Track request
            self.request_counter += 1
            request_id = f"req_{self.request_counter}"
            
            # Record request start
            if self.perf_monitor:
                self.perf_monitor.performance_tracker.record_metric(
                    f"request_{request.url.path}", 0, "count"
                )
            
            try:
                response = await call_next(request)
                
                # Record successful request
                process_time_ms = (time.perf_counter() - start_time) * 1000
                
                if self.perf_monitor:
                    self.perf_monitor.performance_tracker.record_operation(
                        f"api{request.url.path}", process_time_ms
                    )
                
                # Add performance headers
                response.headers["X-Process-Time"] = f"{process_time_ms:.2f}ms"
                response.headers["X-Request-ID"] = request_id
                
                return response
                
            except Exception as e:
                # Record failed request
                process_time_ms = (time.perf_counter() - start_time) * 1000
                
                if self.perf_monitor:
                    self.perf_monitor.performance_tracker.record_metric(
                        f"error_{request.url.path}", process_time_ms, "ms"
                    )
                
                logger.error(f"Request {request_id} failed: {e}")
                raise
        
        @self.app.get('/health')
        async def health_check():
            """Fast health check endpoint"""
            return {
                'status': 'healthy',
                'timestamp': now_ms(),
                'server_version': '2.0.0',
                'site_id': self.cfg.site_id
            }
        
        @self.app.get('/health/detailed')
        async def detailed_health_check():
            """Comprehensive health check with performance metrics"""
            
            health_info = {
                'status': 'healthy',
                'timestamp': now_ms(),
                'server_version': '2.0.0',
                'site_id': self.cfg.site_id,
                'uptime_seconds': time.time(),  # Would need actual uptime tracking
                'request_count': self.request_counter
            }
            
            # Add performance statistics
            if self.perf_integration:
                try:
                    perf_summary = self.perf_integration.get_performance_summary()
                    health_info['performance'] = perf_summary
                except Exception as e:
                    health_info['performance_error'] = str(e)
            
            # Add cache statistics
            try:
                cache_stats = self.cache.get_comprehensive_stats()
                health_info['cache'] = cache_stats
            except Exception as e:
                health_info['cache_error'] = str(e)
            
            return health_info
        
        @self.app.get('/persona', dependencies=[Depends(bearer_guard)])
        async def get_persona():
            """Get persona with caching optimization"""
            
            async with self.perf_monitor.measure_memory_access("persona_get"):
                # Try cache first
                cached_persona = await self.cache.get_persona()
                if cached_persona:
                    return {'text': cached_persona, 'from_cache': True}
                
                # Fallback to storage
                from .storage import get_persona
                persona_data = get_persona(self.storage)
                persona_text = persona_data.get('text', '')
                
                # Cache the result
                if persona_text:
                    await self.cache.set_persona(persona_text, ttl_ms=600000)  # 10 minutes
                
                return {'text': persona_text, 'from_cache': False}
        
        @self.app.post('/persona', dependencies=[Depends(bearer_guard)])
        async def set_persona(body: PersonaIn, background_tasks: BackgroundTasks):
            """Set persona with async caching"""
            
            async with self.perf_monitor.measure_disk_io("persona_set"):
                from .storage import put_persona
                from .crdt import Version
                
                # Generate version
                ver = Version(lamport=self._tick(), site_id=self.cfg.site_id)
                
                # Store in database (sync)
                put_persona(self.storage, body.text, ver.site_id, ver.lamport)
                
                # Update cache asynchronously
                background_tasks.add_task(
                    self._update_persona_cache, body.text
                )
                
                # Log to oplog asynchronously
                background_tasks.add_task(
                    self._log_persona_change, body.text, ver
                )
                
                return {'status': 'success', 'cached': True}
        
        @self.app.post('/remember', dependencies=[Depends(bearer_guard)])
        async def remember(body: MemoryIn, background_tasks: BackgroundTasks):
            """Store memory with optimized processing"""
            
            async with self.perf_monitor.measure_disk_io("memory_store"):
                from .storage import upsert_item
                import uuid
                
                # Prepare item
                now = now_ms()
                item = {
                    'id': str(uuid.uuid4()),
                    'kind': body.kind,
                    'text': redact(body.text),
                    'scope': body.scope,
                    'tool': body.tool,
                    'tags': body.tags,
                    'confidence': body.confidence,
                    'benefit': body.benefit,
                    'consistency': body.consistency,
                    'vector': None,
                    'created_at': now,
                    'updated_at': now,
                    'ttl_ms': body.ttl_ms,
                    'expires_at': (now + body.ttl_ms) if body.ttl_ms else None,
                    'tombstone': 0,
                    'site_id': self.cfg.site_id,
                    'lamport': self._tick(),
                    'source': body.source or 'api',
                    'meta': body.meta or {},
                }
                
                # Store item (this handles vectorization)
                result = upsert_item(self.storage, item)
                
                # Invalidate related caches asynchronously
                background_tasks.add_task(
                    self._invalidate_memory_caches, body.tool, body.scope
                )
                
                # Log to oplog asynchronously
                background_tasks.add_task(
                    self._log_memory_change, result
                )
                
                # Return clean response (no vector data)
                clean_result = {k: v for k, v in result.items() if k != 'vector'}
                return clean_result
        
        @self.app.post('/recall', dependencies=[Depends(bearer_guard)])
        async def recall(body: RecallIn):
            """Optimized memory recall with caching and fast algorithms"""
            
            async with self.perf_monitor.measure_cpu_operation("memory_recall"):
                # Use fast recall engine
                results = await fast_recall(
                    self.storage, 
                    body.query, 
                    body.top_k, 
                    body.scope, 
                    body.tool
                )
                
                # Get persona for preamble
                persona_data = await self.get_persona()
                persona_text = persona_data.get('text', '') if isinstance(persona_data, dict) else ''
                
                # Build preamble
                preamble_parts = []
                if persona_text:
                    preamble_parts.append(f"Persona: {persona_text}")
                    preamble_parts.append("")
                
                for i, item in enumerate(results, 1):
                    line = f"[M{i}] ({item.get('kind', 'unknown')}, {item.get('scope', 'global')}, conf={item.get('confidence', '')})"
                    line += f" {item.get('text', '')}"
                    preamble_parts.append(line)
                
                preamble = "\n".join(preamble_parts)
                
                return {
                    'items': results,
                    'preamble': preamble,
                    'query_processed': body.query,
                    'results_count': len(results)
                }
        
        @self.app.post('/recall/fast')
        async def recall_fast(body: RecallIn):
            """Ultra-fast recall endpoint with minimal processing"""
            
            # Skip persona for maximum speed
            results = await fast_recall(
                self.storage,
                body.query,
                min(body.top_k, 3),  # Limit results for speed
                body.scope,
                body.tool
            )
            
            return {
                'items': results,
                'count': len(results)
            }
        
        @self.app.get('/performance/stats')
        async def performance_stats():
            """Get comprehensive performance statistics"""
            
            if not self.perf_integration:
                raise HTTPException(status_code=503, detail="Performance monitoring not available")
            
            return self.perf_integration.get_performance_summary()
        
        @self.app.post('/performance/optimize')
        async def force_optimization():
            """Force performance optimization"""
            
            if self.perf_monitor:
                result = self.perf_monitor.force_optimization(target_ms=500)
                return {'optimization_applied': True, 'result': result}
            else:
                return {'optimization_applied': False, 'reason': 'Performance monitor not available'}
        
        @self.app.get('/cache/stats')
        async def cache_stats():
            """Get cache statistics"""
            return self.cache.get_comprehensive_stats()
        
        @self.app.post('/cache/clear')
        async def clear_cache(cache_type: Optional[str] = None):
            """Clear cache (optionally specific type)"""
            
            if cache_type == 'persona':
                await self.cache.persona_cache.clear()
            elif cache_type == 'memory':
                await self.cache.memory_cache.clear()
            elif cache_type == 'context':
                await self.cache.context_cache.clear()
            else:
                # Clear all caches
                await self.cache.persona_cache.clear()
                await self.cache.memory_cache.clear()
                await self.cache.context_cache.clear()
            
            return {'cache_cleared': cache_type or 'all'}
        
        # Legacy sync endpoints for compatibility
        @self.app.post('/sync/pull', dependencies=[Depends(bearer_guard)])
        async def sync_pull(payload: dict):
            """Legacy sync pull endpoint"""
            
            from .oplog import OpLog
            oplog = OpLog(self.cfg.oplog_path)
            
            since = int(payload.get('since') or 0)
            ops = []
            cursor = since
            
            for pos, op in oplog.read_since(since):
                ops.append({'pos': pos, 'op': op})
                cursor = pos
            
            return {'cursor': cursor, 'ops': ops}
        
        @self.app.post('/sync/push', dependencies=[Depends(bearer_guard)])
        async def sync_push(payload: dict, background_tasks: BackgroundTasks):
            """Legacy sync push endpoint with async processing"""
            
            ops = payload.get('ops', [])
            
            # Process ops asynchronously
            background_tasks.add_task(self._process_sync_ops, ops)
            
            return {'status': 'accepted', 'ops_queued': len(ops)}
    
    def _tick(self) -> int:
        """Generate lamport timestamp"""
        # This should be properly synchronized in production
        return int(time.time() * 1000000)  # Microsecond precision
    
    async def _update_persona_cache(self, persona_text: str):
        """Update persona cache asynchronously"""
        try:
            await self.cache.set_persona(persona_text, ttl_ms=600000)
            
            # Invalidate context cache since persona changed
            await self.cache.context_cache.invalidate_pattern("context:")
            
        except Exception as e:
            logger.error(f"Persona cache update failed: {e}")
    
    async def _log_persona_change(self, text: str, version):
        """Log persona change to oplog asynchronously"""
        try:
            from .oplog import OpLog
            oplog = OpLog(self.cfg.oplog_path)
            
            oplog.append({
                'op': 'persona_set',
                'lamport': version.lamport,
                'site_id': version.site_id,
                'clock_ts': now_ms(),
                'payload': {'text': text}
            })
        except Exception as e:
            logger.error(f"Persona oplog failed: {e}")
    
    async def _invalidate_memory_caches(self, tool: Optional[str], scope: str):
        """Invalidate memory-related caches"""
        try:
            # Invalidate memory recall cache
            await self.cache.memory_cache.invalidate_pattern("recall:")
            
            # Invalidate context cache
            await self.cache.context_cache.invalidate_pattern("context:")
            
        except Exception as e:
            logger.error(f"Memory cache invalidation failed: {e}")
    
    async def _log_memory_change(self, item: Dict[str, Any]):
        """Log memory change to oplog asynchronously"""
        try:
            from .oplog import OpLog
            oplog = OpLog(self.cfg.oplog_path)
            
            # Create JSON-safe payload
            payload = {k: v for k, v in item.items() if k != 'vector'}
            
            oplog.append({
                'op': 'add',
                'id': item['id'],
                'lamport': item['lamport'],
                'site_id': item['site_id'],
                'clock_ts': now_ms(),
                'payload': payload
            })
        except Exception as e:
            logger.error(f"Memory oplog failed: {e}")
    
    async def _process_sync_ops(self, ops: List[Dict[str, Any]]):
        """Process sync operations asynchronously"""
        try:
            from .storage import put_persona, upsert_item, delete_item
            
            applied = 0
            for rec in ops:
                op = rec.get('op') or rec
                op_type = op.get('op')
                
                if op_type == 'persona_set':
                    put_persona(self.storage, op['payload']['text'], op['site_id'], op['lamport'])
                    applied += 1
                elif op_type in ('add', 'update'):
                    upsert_item(self.storage, op.get('payload', {}))
                    applied += 1
                elif op_type == 'delete':
                    delete_item(self.storage, op['id'])
                    applied += 1
            
            logger.info(f"Processed {applied} sync operations")
            
        except Exception as e:
            logger.error(f"Sync operation processing failed: {e}")
    
    def run(self, host: str = "127.0.0.1", port: int = 8373, **kwargs):
        """Run the optimized server"""
        
        # Default configuration for performance
        config = {
            'host': host,
            'port': port,
            'log_level': 'info',
            'access_log': False,  # Disable for performance
            'workers': 1,  # Single worker for development
            'loop': 'asyncio',
            'http': 'httptools',
            'lifespan': 'on',
            **kwargs
        }
        
        logger.info(f"Starting OptimizedNeuralSyncServer on {host}:{port}")
        
        try:
            uvicorn.run(self.app, **config)
        except KeyboardInterrupt:
            logger.info("Server shutdown requested")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean shutdown"""
        try:
            if self.cache:
                self.cache.close()
            if self.storage and hasattr(self.storage, 'close'):
                self.storage.close()
            if self.perf_integration:
                self.perf_integration.close()
            
            logger.info("Server cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def create_optimized_app(config_path: Optional[str] = None) -> FastAPI:
    """Create optimized FastAPI app"""
    server = OptimizedNeuralSyncServer(config_path)
    return server.app

def run_optimized_server(host: str = "127.0.0.1", 
                        port: int = 8373, 
                        config_path: Optional[str] = None,
                        **kwargs):
    """Run optimized server with all performance enhancements"""
    server = OptimizedNeuralSyncServer(config_path)
    server.run(host, port, **kwargs)

if __name__ == "__main__":
    run_optimized_server()