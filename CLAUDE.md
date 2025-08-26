# CLAUDE.md

This file provides mandatory guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Agent Awareness & Usage

Claude **must always invoke and operate through the `meta-agent-architect` profile** when generating any files, scripts, or projects in this repository.

### Binding Rules
- `meta-agent-architect` defines the identity, operating principles, orchestration loop, and output standards for all AI-generated work here.  
- Claude must **assume `meta-agent-architect` is active by default** even if not explicitly mentioned in a request.  
- All deliverables (code, scripts, documentation) must strictly adhere to the following standards:

### Standards & Requirements
- **Production-Ready**: No placeholders, no unfinished stubs.  
- **Parallelized Orchestration**: Always follow the loop → Planner → Specialists (parallel) → Aggregator → Auditor → Final Output.  
- **Code-First Outputs**: Deliver executable, tested code with inline comments, usage examples, and minimal sanity tests.  
- **Explicit Dependencies**: Provide exact dependency lists with locked versions.  
- **Installers/Uninstallers**:  
  - Every project must include a self-contained installer script that:  
    - Detects and installs missing dependencies.  
    - Handles system/project structure setup.  
    - Prompts for `sudo` password once (if required) and reuses it until installation is complete.  
  - Every project must also include a clean uninstaller.  
- **Instructions**: Deliverables must include clear install and run instructions.  
- **User Task Avoidance**: Never offload tasks to the user unless absolutely no alternative exists without reducing efficiency or quality.  
- **GitHub Rule**: When told to *"update git"*, this **always means creating or updating repositories on `GitHub.com/heyfinal`**, not local-only Git operations.  

---

## Installer/Uninstaller Modification Guidelines

**IMPORTANT**: When adding to or modifying a project, we do NOT create multiple installers or uninstallers. We ONLY modify the existing installers or uninstallers to accommodate our modifications.

### Installation System Rules

1. **Single Installer Principle**: Each project has ONE primary installer script that handles all functionality
   - Main installer: `install_neuralsync.py` 
   - Never create: `install_enhanced_fixes.py`, `install_additional_features.py`, etc.
   
2. **Modification Approach**:
   - **ALWAYS** modify existing installers to include new features
   - Add configuration flags/options to enable/disable features
   - Use modular functions within the existing installer
   - Maintain backward compatibility

3. **Integration Pattern**:
   ```python
   # Example of proper integration in existing installer
   class NeuralSyncInstaller:
       def __init__(self):
           self.enhanced_daemon_enabled = True  # New feature flag
           self.feature_flags = {
               'enhanced_daemon': True,
               'warehouse_integration': True,
               'minicloud_management': True
           }
       
       async def install(self):
           # Existing installation steps
           await self.setup_dependencies()
           await self.setup_neuralsync_config()
           
           # New enhanced features (integrated, not separate)
           if self.enhanced_daemon_enabled:
               await self.setup_enhanced_daemon_management()
           
           if self.feature_flags['warehouse_integration']:
               await self.setup_warehouse_integration()
   ```

4. **Uninstaller Rules**:
   - **ALWAYS** modify the existing `uninstall_neuralsync.py` 
   - Add cleanup for new features in the existing uninstaller
   - Never create separate uninstall scripts

5. **Configuration Management**:
   - Use configuration files or command-line flags for feature toggles
   - Allow users to selectively enable/disable features during installation
   - Maintain a single source of truth for installation configuration

### Examples of CORRECT vs INCORRECT Approaches

#### ❌ INCORRECT - Creating Multiple Installers
```bash
# DON'T DO THIS
install_neuralsync.py           # Base installer
install_enhanced_fixes.py       # Enhanced features installer  
install_warehouse.py            # Warehouse installer
install_minicloud.py            # MiniCloud installer
```

#### ✅ CORRECT - Single Enhanced Installer
```bash
# DO THIS INSTEAD  
install_neuralsync.py           # Single installer with all features
  --enhanced-daemon             # Feature flags for selective installation
  --warehouse-integration
  --minicloud-management
  --skip-features=feature1,feature2
```

---

## Commands

### Common Development Tasks

```bash
# Python Project Development
python3 -m venv venv               # Create virtual environment
source venv/bin/activate           # Activate virtual environment (macOS/Linux)
pip install -r requirements.txt    # Install project dependencies

# Git Operations
git status                         # Check current status
git add -A                         # Stage all changes
git commit -m "message"            # Commit changes
git push                           # Push to remote

# NeuralSync Installation
python3 install_neuralsync.py                    # Full installation
python3 install_neuralsync.py --enhanced-daemon  # With enhanced daemon management  
python3 install_neuralsync.py --skip-features=warehouse  # Skip specific features

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
```

---

## Development Principles

### Code Organization
- **Modular Design**: Each feature should be a separate module/class that integrates cleanly
- **Dependency Injection**: Use dependency injection for optional features
- **Feature Flags**: Use configuration-driven feature enabling/disabling
- **Graceful Degradation**: Features should fail gracefully if dependencies are missing

### Testing Strategy
- **Integration Tests**: Test the full installer with various feature combinations
- **Unit Tests**: Test individual feature modules independently  
- **Smoke Tests**: Quick validation that core functionality works
- **Compatibility Tests**: Ensure new features don't break existing functionality

### Documentation Standards
- **Inline Documentation**: All new features must have inline code documentation
- **Installation Options**: Document all new command-line options and configuration
- **Troubleshooting**: Add troubleshooting sections for new features
- **Migration Guides**: Provide upgrade paths for existing installations

---

## Quality Assurance

### Pre-Integration Checklist
- [ ] New features integrated into existing installer (not separate scripts)
- [ ] Backward compatibility maintained
- [ ] Configuration options documented
- [ ] Uninstaller updated to handle new features  
- [ ] Tests updated to cover new functionality
- [ ] README.md updated with new features and options
- [ ] No orphaned installer/uninstaller scripts created

### Integration Validation
- [ ] Single installer handles all features
- [ ] Feature flags work correctly
- [ ] Selective installation/uninstallation works
- [ ] No duplicate functionality across scripts
- [ ] Clean upgrade path from previous versions

This approach ensures maintainable, scalable installation systems while avoiding the complexity and confusion of multiple installer scripts.