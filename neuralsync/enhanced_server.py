"""
Enhanced NeuralSync2 Server
Integrates all bleeding-edge components into a unified system
"""

import asyncio
import logging
import time
import os
import signal
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvloop
import orjson
from pathlib import Path

# Import enhanced components
from .memory_manager import get_memory_manager
from .ultra_comm import get_comm_manager, MessageTypes
from .crdt import ByzantineCRDT, AdvancedVersion
from .research_dedup import get_deduplicator
from .unleashed_mode import get_unleashed_manager
from .personality_manager import get_personality_manager
from .config import load_config
from .auth import bearer_guard
from .api import MemoryIn, RecallIn, PersonaIn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NeuralSync2System:
    """Main system orchestrator for NeuralSync2"""
    
    def __init__(self):
        self.config = load_config()
        
        # Core components
        self.memory_manager = get_memory_manager()
        self.comm_manager = get_comm_manager(self.config.site_id)
        self.crdt = ByzantineCRDT(self.config.site_id)
        self.deduplicator = get_deduplicator()
        self.unleashed_manager = get_unleashed_manager()
        self.personality_manager = get_personality_manager()
        
        # System state
        self.startup_time = time.time()
        self.is_running = False
        self.background_tasks = []
        
        # Performance metrics
        self.metrics = {
            'requests_handled': 0,
            'memory_operations': 0,
            'duplicate_requests_blocked': 0,
            'unleashed_sessions': 0,
            'avg_response_time': 0.0
        }
        
    async def initialize(self):
        """Initialize all system components"""
        logger.info("Initializing NeuralSync2 system...")
        
        try:
            # Start communication server
            await self.comm_manager.start_server()
            
            # Register message handlers
            self._register_message_handlers()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.is_running = True
            startup_time = time.time() - self.startup_time
            
            logger.info(f"NeuralSync2 system initialized in {startup_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise
            
    def _register_message_handlers(self):
        """Register handlers for inter-CLI communication"""
        
        async def handle_memory_sync(message):
            """Handle memory synchronization messages"""
            try:
                payload = orjson.loads(message.payload)
                operation_type = payload.get('type')
                
                if operation_type == 'memory_store':
                    # Store memory from another CLI tool
                    await self._handle_remote_memory_store(payload)
                elif operation_type == 'memory_recall':
                    # Handle memory recall request
                    await self._handle_remote_memory_recall(message.sender, payload)
                    
            except Exception as e:
                logger.error(f"Error handling memory sync: {e}")
                
        async def handle_personality_update(message):
            """Handle personality update messages"""
            try:
                payload = orjson.loads(message.payload)
                updates = payload.get('updates', {})
                
                # Apply personality updates
                await self.personality_manager.update_personality_state(
                    message.sender, 
                    payload.get('session_id', ''),
                    updates
                )
                
            except Exception as e:
                logger.error(f"Error handling personality update: {e}")
                
        async def handle_unleashed_request(message):
            """Handle unleashed mode requests"""
            try:
                payload = orjson.loads(message.payload)
                
                success, token = await self.unleashed_manager.enable_unleashed_mode(
                    message.sender,
                    payload.get('session_id', ''),
                    payload.get('context', {})
                )
                
                # Send response
                response_payload = orjson.dumps({
                    'success': success,
                    'token': token,
                    'timestamp': time.time()
                })
                
                await self.comm_manager.send_message(
                    message.sender,
                    'unleashed_response', 
                    response_payload
                )
                
            except Exception as e:
                logger.error(f"Error handling unleashed request: {e}")
                
        # Register handlers
        self.comm_manager.register_handler(MessageTypes.MEMORY_STORE, handle_memory_sync)
        self.comm_manager.register_handler(MessageTypes.MEMORY_RECALL, handle_memory_sync)
        self.comm_manager.register_handler(MessageTypes.PERSONALITY_UPDATE, handle_personality_update)
        self.comm_manager.register_handler(MessageTypes.UNLEASHED_MODE, handle_unleashed_request)
        
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        
        tasks = [
            self._memory_cleanup_task(),
            self._metrics_collection_task(),
            self._health_monitoring_task(),
            self._research_optimization_task()
        ]
        
        for task in tasks:
            background_task = asyncio.create_task(task)
            self.background_tasks.append(background_task)
            
    async def _memory_cleanup_task(self):
        """Background memory cleanup and optimization"""
        while self.is_running:
            try:
                # Cleanup memory manager
                stats = self.memory_manager.get_stats()
                if stats['active_allocations'] > 1000:
                    # Trigger garbage collection
                    pass
                    
                # Optimize deduplication index
                await self.deduplicator.optimize_index()
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Memory cleanup task error: {e}")
                await asyncio.sleep(60)
                
    async def _metrics_collection_task(self):
        """Collect and log system metrics"""
        while self.is_running:
            try:
                # Collect metrics from all components
                memory_stats = self.memory_manager.get_stats()
                comm_stats = self.comm_manager.get_stats()
                dedup_stats = self.deduplicator.get_stats()
                personality_stats = self.personality_manager.get_personality_stats()
                security_stats = self.unleashed_manager.get_security_status()
                
                # Log aggregated metrics
                logger.info(f"System metrics - "
                          f"Memory allocs: {memory_stats['active_allocations']}, "
                          f"Comm msgs: {comm_stats.get('messages_sent', 0)}, "
                          f"Dedup entries: {dedup_stats['stored_entries']}, "
                          f"Active personalities: {personality_stats['active_sessions']}, "
                          f"Security threat: {security_stats['threat_level']}")
                          
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)
                
    async def _health_monitoring_task(self):
        """Monitor system health and auto-recover"""
        while self.is_running:
            try:
                # Check component health
                components_healthy = await self._check_component_health()
                
                if not components_healthy:
                    logger.warning("System health issues detected - initiating recovery")
                    await self._initiate_recovery()
                    
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)
                
    async def _research_optimization_task(self):
        """Optimize research deduplication periodically"""
        while self.is_running:
            try:
                await self.deduplicator.optimize_index()
                await asyncio.sleep(1800)  # Every 30 minutes
                
            except Exception as e:
                logger.error(f"Research optimization error: {e}")
                await asyncio.sleep(300)
                
    async def _check_component_health(self) -> bool:
        """Check health of all components"""
        
        try:
            # Test memory manager
            test_mem = self.memory_manager.allocate_message(100, "health_check")
            if test_mem:
                self.memory_manager.release_message(test_mem)
            else:
                logger.warning("Memory manager health check failed")
                return False
                
            # Test communication
            comm_stats = self.comm_manager.get_stats()
            if comm_stats.get('avg_send_latency', 0) > 100:  # > 100ms latency
                logger.warning("Communication latency too high")
                return False
                
            # Test deduplication
            is_dup, _, _ = await self.deduplicator.is_duplicate_research("health check", "", False)
            # Should complete without error
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
            
    async def _initiate_recovery(self):
        """Initiate system recovery procedures"""
        
        logger.info("Initiating system recovery procedures...")
        
        try:
            # Clear caches
            self.deduplicator.clear_cache()
            self.personality_manager.personality_cache.clear()
            
            # Reset communication connections
            for sock in self.comm_manager.socket_pool.values():
                sock.close()
            self.comm_manager.socket_pool.clear()
            
            # Reset metrics
            self.metrics = {key: 0 if isinstance(val, (int, float)) else val 
                           for key, val in self.metrics.items()}
                           
            logger.info("System recovery completed")
            
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            
    async def _handle_remote_memory_store(self, payload: Dict[str, Any]):
        """Handle memory store from remote CLI tool"""
        
        # Extract memory data
        memory_data = payload.get('memory', {})
        
        # Store in local CRDT
        key = memory_data.get('id', f"remote_{time.time()}")
        self.crdt.set(key, memory_data)
        
        self.metrics['memory_operations'] += 1
        
    async def _handle_remote_memory_recall(self, sender: str, payload: Dict[str, Any]):
        """Handle memory recall request from remote CLI tool"""
        
        query = payload.get('query', '')
        
        # Check for duplicates first
        is_dup, cached_result, _ = await self.deduplicator.is_duplicate_research(
            query, payload.get('context', ''), False
        )
        
        if is_dup:
            self.metrics['duplicate_requests_blocked'] += 1
            
            # Send cached result
            response_payload = orjson.dumps({
                'type': 'recall_response',
                'cached': True,
                'result': cached_result,
                'timestamp': time.time()
            })
        else:
            # Perform actual recall
            # This would integrate with existing recall logic
            response_payload = orjson.dumps({
                'type': 'recall_response',
                'cached': False,
                'result': 'No cached result available',
                'timestamp': time.time()
            })
            
        await self.comm_manager.send_message(
            sender,
            MessageTypes.SYNC_RESPONSE,
            response_payload
        )
        
    async def shutdown(self):
        """Gracefully shutdown the system"""
        
        logger.info("Shutting down NeuralSync2 system...")
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
            
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
        # Cleanup components
        await self.comm_manager.cleanup()
        self.memory_manager.cleanup()
        self.deduplicator.cleanup()
        
        logger.info("System shutdown complete")
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        uptime = time.time() - self.startup_time
        
        return {
            'uptime_seconds': uptime,
            'is_running': self.is_running,
            'metrics': self.metrics,
            'component_status': {
                'memory_manager': self.memory_manager.get_stats(),
                'communication': self.comm_manager.get_stats(),
                'deduplication': self.deduplicator.get_stats(),
                'personality': self.personality_manager.get_personality_stats(),
                'security': self.unleashed_manager.get_security_status()
            }
        }


# Global system instance
_neuralsync_system: Optional[NeuralSync2System] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan management"""
    global _neuralsync_system
    
    # Startup
    _neuralsync_system = NeuralSync2System()
    await _neuralsync_system.initialize()
    
    yield
    
    # Shutdown
    if _neuralsync_system:
        await _neuralsync_system.shutdown()


# Create FastAPI app with enhanced capabilities
app = FastAPI(
    title='NeuralSync2',
    description='Bleeding-edge multi-agent memory synchronization system',
    version='2.0.0',
    lifespan=lifespan
)


# Enhanced API endpoints
@app.get('/health')
async def health():
    """Enhanced health check with detailed status"""
    if _neuralsync_system:
        status = _neuralsync_system.get_system_status()
        return JSONResponse(content=status)
    else:
        return JSONResponse(
            content={'error': 'System not initialized'},
            status_code=503
        )


@app.post('/remember/enhanced', dependencies=[Depends(bearer_guard)])
async def remember_enhanced(body: MemoryIn, background_tasks: BackgroundTasks):
    """Enhanced memory storage with deduplication and personality awareness"""
    
    if not _neuralsync_system:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    start_time = time.perf_counter()
    
    try:
        # Check for research duplication
        combined_query = f"{body.text} {body.scope}"
        is_dup, cached_result, similar = await _neuralsync_system.deduplicator.is_duplicate_research(
            combined_query, body.meta.get('context', '') if body.meta else ''
        )
        
        if is_dup:
            _neuralsync_system.metrics['duplicate_requests_blocked'] += 1
            return JSONResponse(content={
                'duplicate': True,
                'cached_result': cached_result,
                'similar_entries': similar[:3]  # Top 3 similar
            })
            
        # Store in enhanced memory system
        memory_view = _neuralsync_system.memory_manager.allocate_message(
            len(body.text.encode()),
            f"memory_{time.time()}"
        )
        
        if not memory_view:
            raise HTTPException(status_code=507, detail="Memory allocation failed")
            
        # Store in CRDT for conflict-free replication
        memory_id = f"mem_{int(time.time() * 1000000)}"
        memory_data = {
            'id': memory_id,
            'text': body.text,
            'scope': body.scope,
            'tool': body.tool,
            'timestamp': time.time(),
            'confidence': body.confidence,
            'benefit': body.benefit
        }
        
        _neuralsync_system.crdt.set(memory_id, memory_data)
        
        # Store research result to prevent future duplicates
        _neuralsync_system.deduplicator.store_research_result(
            body.text,
            body.meta.get('context', '') if body.meta else '',
            f"Stored memory: {memory_id}",
            {body.scope, body.tool} if body.tool else {body.scope}
        )
        
        # Update metrics
        _neuralsync_system.metrics['memory_operations'] += 1
        _neuralsync_system.metrics['requests_handled'] += 1
        
        # Record interaction for personality system
        background_tasks.add_task(
            _neuralsync_system.personality_manager.record_interaction,
            body.tool or 'unknown',
            'current',
            body.text,
            f"Memory stored: {memory_id}",
            'memory_storage',
            {body.scope}
        )
        
        # Calculate response time
        response_time = (time.perf_counter() - start_time) * 1000
        _neuralsync_system.metrics['avg_response_time'] = (
            _neuralsync_system.metrics['avg_response_time'] * 0.9 + response_time * 0.1
        )
        
        return JSONResponse(content={
            'id': memory_id,
            'stored': True,
            'duplicate': False,
            'response_time_ms': response_time
        })
        
    except Exception as e:
        logger.error(f"Enhanced memory storage error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/recall/enhanced', dependencies=[Depends(bearer_guard)])
async def recall_enhanced(body: RecallIn):
    """Enhanced memory recall with personality context"""
    
    if not _neuralsync_system:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    start_time = time.perf_counter()
    
    try:
        # Get unified personality for context
        personality_context = await _neuralsync_system.personality_manager.get_unified_personality(
            body.tool or 'unknown',
            'current'
        )
        
        # Check research deduplication first
        is_dup, cached_result, similar_entries = await _neuralsync_system.deduplicator.is_duplicate_research(
            body.query,
            personality_context.get('context_summary', ''),
            True
        )
        
        if is_dup:
            _neuralsync_system.metrics['duplicate_requests_blocked'] += 1
            
            return JSONResponse(content={
                'cached': True,
                'result': cached_result,
                'similar': similar_entries[:5],
                'personality_context': personality_context['prompt_context']
            })
            
        # Perform regular recall (would integrate with existing logic)
        # For now, return enhanced context
        
        response_time = (time.perf_counter() - start_time) * 1000
        _neuralsync_system.metrics['requests_handled'] += 1
        
        return JSONResponse(content={
            'cached': False,
            'query': body.query,
            'personality_context': personality_context['prompt_context'],
            'system_context': personality_context.get('context_summary', ''),
            'response_time_ms': response_time,
            'similar_queries': [entry[0] for entry in similar_entries[:3]]
        })
        
    except Exception as e:
        logger.error(f"Enhanced recall error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/unleashed/request', dependencies=[Depends(bearer_guard)])
async def unleashed_mode_request(request_data: Dict[str, Any]):
    """Request unleashed mode access"""
    
    if not _neuralsync_system:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    try:
        cli_tool = request_data.get('cli_tool', 'unknown')
        session_id = request_data.get('session_id', f"session_{time.time()}")
        context = request_data.get('context', {})
        
        # Add request timestamp and signature
        context.update({
            'request_timestamp': time.time(),
            'request_signature': f"sig_{hash(str(context))}"
        })
        
        success, token = await _neuralsync_system.unleashed_manager.enable_unleashed_mode(
            cli_tool,
            session_id, 
            context
        )
        
        if success:
            _neuralsync_system.metrics['unleashed_sessions'] += 1
            
        return JSONResponse(content={
            'success': success,
            'token': token,
            'message': 'Unleashed mode enabled' if success else 'Request denied'
        })
        
    except Exception as e:
        logger.error(f"Unleashed mode request error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/system/status')
async def system_status():
    """Get comprehensive system status"""
    
    if not _neuralsync_system:
        return JSONResponse(
            content={'error': 'System not initialized'},
            status_code=503
        )
        
    status = _neuralsync_system.get_system_status()
    return JSONResponse(content=status)


@app.post('/system/emergency_shutdown', dependencies=[Depends(bearer_guard)])
async def emergency_shutdown(reason: str = "Manual request"):
    """Emergency system shutdown"""
    
    if not _neuralsync_system:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    await _neuralsync_system.unleashed_manager.emergency_shutdown(reason)
    
    return JSONResponse(content={
        'shutdown': True,
        'reason': reason,
        'timestamp': time.time()
    })


def main():
    """Main entry point"""
    
    # Use uvloop for better performance on Unix systems
    if os.name != 'nt':  # Not Windows
        uvloop.install()
        
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        if _neuralsync_system:
            asyncio.create_task(_neuralsync_system.shutdown())
            
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get configuration
    config = load_config()
    
    # Start server
    import uvicorn
    uvicorn.run(
        app,
        host=config.host or '127.0.0.1',
        port=config.port or 8000,
        log_level='info',
        access_log=True,
        loop='uvloop' if os.name != 'nt' else 'asyncio'
    )


if __name__ == "__main__":
    main()