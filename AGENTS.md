# AGENTS.md

This file defines the agent profiles and operational guidelines for AI agents working on the NeuralSync v2 project.

---

## Agent Profiles

### 🤖 meta-agent-architect

**Primary Role**: System architect and project orchestrator  
**Activation**: Default for all project work  
**Specialization**: Enterprise-grade system design, parallel orchestration, production deployment

**Core Responsibilities**:
- Design complete system architectures
- Orchestrate parallel specialist teams
- Ensure production-ready code delivery
- Validate security and performance requirements
- Coordinate cross-system integrations

**Operating Loop**: Planner → Specialists (parallel execution) → Aggregator → Auditor → Final Output

**Output Standards**:
- ✅ Production-ready (no placeholders)
- ✅ Executable code with tests
- ✅ Complete documentation
- ✅ Installation/uninstallation scripts
- ✅ Security validation
- ✅ Performance benchmarks

### 🧠 memory-specialist

**Primary Role**: Memory system optimization and CRDT synchronization  
**Activation**: Memory-related tasks, performance optimization  
**Specialization**: Vector databases, semantic search, distributed systems

### 🚀 performance-specialist  

**Primary Role**: System performance optimization and monitoring  
**Activation**: Performance issues, daemon management, system optimization  
**Specialization**: Service detection, auto-recovery, resource management

### 🔐 security-specialist

**Primary Role**: Security framework implementation and validation  
**Activation**: Security requirements, authentication, encryption  
**Specialization**: JWT authentication, encryption, access control, audit logging

### 🌐 integration-specialist

**Primary Role**: Cross-system integration and API development  
**Activation**: CLI tool integration, external service connections  
**Specialization**: API design, service orchestration, network protocols

---

## Project Modification Guidelines

**CRITICAL PRINCIPLE**: When adding to or modifying a project, we do NOT create multiple installers or uninstallers. We ONLY modify the existing installers or uninstallers to accommodate our modifications.

### 📋 Modification Workflow

1. **Assessment Phase**
   - Identify existing installer/uninstaller scripts
   - Analyze current architecture and extension points
   - Plan integration approach for new features

2. **Integration Planning**
   - Design modular integration into existing systems
   - Define feature flags and configuration options
   - Plan backward compatibility approach

3. **Implementation Strategy**
   - Modify existing installers to include new functionality
   - Add configuration toggles for optional features
   - Implement graceful degradation for missing dependencies

4. **Validation Process**
   - Test installation with various feature combinations
   - Validate uninstallation removes all new components
   - Ensure no orphaned files or duplicate functionality

### 🛠️ Implementation Patterns

#### Single Installer Enhancement
```python
# CORRECT: Enhance existing installer
class ProjectInstaller:
    def __init__(self):
        self.feature_flags = {
            'enhanced_daemon': True,
            'warehouse_integration': False,  # Optional feature
            'monitoring': True
        }
    
    async def install(self):
        # Core installation (always runs)
        await self.install_core()
        
        # Enhanced features (configurable)
        if self.feature_flags['enhanced_daemon']:
            await self.setup_enhanced_daemon()
            
        if self.feature_flags['warehouse_integration']:
            await self.setup_warehouse()
```

#### Configuration-Driven Features
```python
# CORRECT: Use configuration for feature management
def load_installation_config():
    return {
        'features': {
            'enhanced_daemon': True,
            'warehouse': False,
            'monitoring': True
        },
        'performance_mode': 'adaptive',
        'security_level': 'standard'
    }
```

### ❌ Anti-Patterns (DO NOT DO)

#### Multiple Installers
```bash
# WRONG: Creating separate installers
install_base.py           # Base functionality
install_enhanced.py       # Enhanced features  
install_warehouse.py      # Warehouse features
install_monitoring.py     # Monitoring features
```

#### Duplicate Functionality
```python
# WRONG: Duplicating installation logic
def install_base_system():
    # Base installation logic
    
def install_enhanced_system():
    # Duplicates base logic + enhancements
```

#### Hardcoded Feature Selection
```python
# WRONG: No configuration options
def install():
    install_core()
    install_enhanced()      # Always installed, no options
    install_warehouse()     # Always installed, no choice
```

---

## Agent Coordination Rules

### 🤝 Parallel Execution Guidelines

1. **Task Decomposition**: Break complex tasks into independent, parallel workstreams
2. **Interface Definition**: Define clear interfaces between parallel components
3. **Dependency Management**: Identify and handle inter-component dependencies
4. **Result Aggregation**: Systematically combine parallel work products

### 🔄 Communication Protocols

#### Status Reporting
```markdown
## Agent Status Report
- **Agent**: [agent-name]
- **Task**: [current-task]  
- **Progress**: [percentage-complete]
- **Blockers**: [dependencies/issues]
- **Output**: [deliverable-status]
```

#### Handoff Protocol
```markdown
## Work Handoff
- **From**: [source-agent]
- **To**: [target-agent]
- **Deliverable**: [what-is-being-transferred]
- **Status**: [ready/needs-review/blocked]
- **Next Actions**: [what-target-agent-should-do]
```

### 🔍 Quality Gates

#### Pre-Integration Checklist
- [ ] Code is production-ready (no TODO/FIXME)
- [ ] Tests written and passing
- [ ] Documentation complete
- [ ] Security review completed
- [ ] Performance benchmarks satisfied
- [ ] Integration tests passing

#### Architecture Review Points
- [ ] Follows existing patterns and conventions
- [ ] Maintains backward compatibility
- [ ] Uses existing infrastructure (no reinvention)
- [ ] Proper error handling and logging
- [ ] Scalable and maintainable design

---

## Deployment Standards

### 🚀 Production Readiness Criteria

1. **Functionality**
   - All features work as documented
   - Error handling covers edge cases
   - Graceful degradation for failures

2. **Performance** 
   - Meets or exceeds performance benchmarks
   - Resource usage within acceptable limits
   - Startup/shutdown times optimized

3. **Security**
   - Authentication and authorization implemented
   - Data encryption at rest and in transit
   - Audit logging for sensitive operations
   - No hardcoded secrets or credentials

4. **Maintainability**
   - Clear code structure and documentation
   - Comprehensive test coverage
   - Monitoring and observability
   - Rollback/recovery procedures

### 📊 Success Metrics

#### Installation Success Rate
- Target: >99% success rate across supported platforms
- Measurement: Automated installation testing
- Recovery: Detailed error logging and troubleshooting guides

#### Feature Adoption
- Target: >80% of users enable enhanced features
- Measurement: Telemetry and usage analytics
- Recovery: User education and improved defaults

#### System Performance
- Target: <2s cold start, <100ms memory recall
- Measurement: Automated performance benchmarking  
- Recovery: Performance optimization and caching

#### User Satisfaction
- Target: >90% positive feedback
- Measurement: User surveys and issue tracking
- Recovery: Rapid issue resolution and feature improvements

---

## Best Practices

### 🎯 Development Guidelines

1. **Code Quality**
   - Follow existing code style and conventions
   - Write comprehensive unit and integration tests
   - Use type hints and documentation strings
   - Handle errors gracefully with proper logging

2. **Documentation**
   - Update README.md for user-facing changes
   - Document all new configuration options
   - Provide clear installation and usage examples
   - Include troubleshooting guides

3. **Testing Strategy**
   - Unit tests for individual components
   - Integration tests for cross-component functionality
   - End-to-end tests for user workflows
   - Performance tests for critical paths

4. **Security Practices**
   - Never commit secrets or credentials
   - Validate all inputs and sanitize outputs
   - Use secure communication protocols
   - Implement proper access controls

### 🔧 Integration Standards

1. **Configuration Management**
   - Use YAML/JSON for configuration files
   - Provide sensible defaults for all options
   - Support environment variable overrides
   - Validate configuration on startup

2. **Error Handling** 
   - Use structured logging with appropriate levels
   - Provide actionable error messages
   - Implement retry logic for transient failures
   - Support debugging and diagnostic modes

3. **Compatibility**
   - Maintain backward compatibility for existing users
   - Support multiple Python versions (3.9+)
   - Work across major operating systems
   - Handle version migrations gracefully

4. **Performance**
   - Optimize critical paths and frequently used functions
   - Use caching for expensive operations
   - Implement connection pooling for external services
   - Monitor and alert on performance degradation

---

## Troubleshooting and Support

### 🔍 Diagnostic Procedures

1. **Installation Issues**
   - Check system requirements and dependencies
   - Verify permissions and file system access
   - Review installation logs for error details
   - Test with minimal configuration

2. **Runtime Problems**
   - Check service status and process health
   - Review application logs for errors
   - Validate configuration files and settings
   - Test network connectivity and permissions

3. **Performance Issues**
   - Profile CPU and memory usage
   - Analyze database query performance
   - Check network latency and bandwidth
   - Review caching effectiveness

### 🛠️ Recovery Procedures

1. **Service Recovery**
   - Automatic restart for transient failures
   - Health checks and monitoring alerts
   - Graceful degradation when dependencies fail
   - Manual intervention procedures for complex issues

2. **Data Recovery**
   - Regular backups of critical data
   - Point-in-time recovery for corruption
   - Data validation and integrity checks
   - Import/export tools for data migration

3. **System Recovery**
   - Complete uninstallation and reinstallation procedures
   - Configuration reset to defaults
   - Selective feature disable/enable
   - Emergency contact and escalation procedures

This comprehensive agent framework ensures consistent, high-quality development while maintaining system reliability and user satisfaction.