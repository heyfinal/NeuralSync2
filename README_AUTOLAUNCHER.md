# NeuralSync Auto-Launch Integration System

A production-ready auto-launch integration system for NeuralSync v2 that seamlessly wraps existing CLI tools with shared memory, persona synchronization, and inter-agent communication.

## 🚀 Quick Start

```bash
# Install the system
python3 install_neuralsync.py

# Validate installation  
python3 validate_installation.py

# Start using integrated CLIs
claude-ns --help
codex-ns --help
gemini-ns --help
```

## 📋 Features

### 🔄 Auto-Launch Integration
- **Smart Detection**: Automatically detects if NeuralSync server is running
- **Zero-Config Startup**: Starts server transparently on first use
- **Daemon Management**: Production-ready process management with health monitoring
- **Resource Limits**: Configurable limits to prevent resource exhaustion

### 🤝 Inter-Agent Communication
- **Ultra-Low Latency**: Sub-10ms communication via Unix domain sockets
- **Message Broker**: Centralized message routing between agents
- **Agent Discovery**: Dynamic discovery of available agents and their capabilities
- **Task Delegation**: Seamless task delegation between agents

### 🧠 Shared Memory & Persona
- **Unified Context**: Shared persona and memory across all agents
- **CRDT Synchronization**: Conflict-free replicated data for consistency
- **Memory Deduplication**: Intelligent deduplication to prevent information overflow
- **Session Persistence**: Maintains context across agent sessions

### 🎯 Agent Spawning & Lifecycle
- **Dynamic Spawning**: Spawn agents on demand for specific tasks
- **Resource Monitoring**: Real-time monitoring of agent resource usage
- **Auto-Recovery**: Automatic restart of failed agents
- **Lifecycle Management**: Complete agent lifecycle from spawn to termination

### 🔧 CLI Integration
- **Transparent Wrapping**: Existing CLI tools work unchanged
- **Context Injection**: Automatic injection of shared context
- **Enhanced Commands**: Additional commands for inter-agent coordination
- **Status Monitoring**: Real-time system status and health checks

## 🛠️ Installation

### Prerequisites

**Required:**
- Python 3.8+
- pip
- git

**Optional (for full functionality):**
- [Claude Code CLI](https://claude.ai/code) → `claude-ns`
- [CodexCLI](https://github.com/openai/codex-cli) → `codex-ns`
- [Gemini CLI](https://ai.google.dev/gemini-api/docs/cli) → `gemini-ns`

### Automated Installation

```bash
# Clone or download NeuralSync2
cd NeuralSync2

# Run installer (handles dependencies, setup, and configuration)
python3 install_neuralsync.py

# Optional: Skip dependency installation if already available
python3 install_neuralsync.py --skip-deps

# Validate installation
python3 validate_installation.py
```

### Manual Installation Steps

If you prefer manual installation:

1. **Setup Python Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Install Wrapper Scripts:**
   ```bash
   mkdir -p ~/.local/bin
   cp bin/* ~/.local/bin/
   chmod +x ~/.local/bin/claude-ns ~/.local/bin/codex-ns ~/.local/bin/gemini-ns
   ```

3. **Setup PATH:**
   ```bash
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```

4. **Initialize Configuration:**
   ```bash
   python3 -c "from neuralsync.config import load_config; load_config()"
   ```

## 📖 Usage

### Basic Usage

```bash
# Use any wrapper just like the original CLI
claude-ns "Write a Python function to sort a list"
codex-ns --complete "def fibonacci("
gemini-ns "Explain quantum computing"

# The NeuralSync server auto-starts on first use
# All agents share the same memory and persona
```

### Advanced Features

```bash
# Check system status
claude-ns --neuralsync-status

# Spawn another agent to handle a specific task
claude-ns --spawn-agent codex "Generate a REST API"

# Handle spawned tasks
codex-ns --task '{"task": "Generate a REST API", "context": {}}'

# View all available commands
claude-ns --help
```

### Agent Communication

Agents can seamlessly communicate and coordinate:

```python
# Agent A (claude-ns) analyzing code
claude-ns "Analyze this code for bugs" < myfile.py

# Agent A can spawn Agent B (codex-ns) to generate fixes  
# Agent B inherits context and persona from Agent A
# Results are shared back through NeuralSync memory
```

## ⚙️ Configuration

### Config File Location
- `~/.neuralsync/config.yaml`

### Key Settings
```yaml
site_id: "unique-installation-id"
db_path: "~/.neuralsync/memory.db"
oplog_path: "~/.neuralsync/oplog.jsonl"
bind_host: "127.0.0.1"
bind_port: 8373
token: ""  # Optional auth token
```

### Environment Variables
```bash
# Override default settings
export NS_HOST=127.0.0.1
export NS_PORT=8373
export NS_TOKEN=your-auth-token

# Debug mode
export NEURALSYNC_DEBUG=1
```

## 🧪 Testing & Validation

### Quick Validation
```bash
# Fast installation check
python3 validate_installation.py

# Detailed JSON report
python3 validate_installation.py --json
```

### Comprehensive Testing
```bash
# Full integration test suite
python3 tests/integration_test.py

# Verbose output
python3 tests/integration_test.py --verbose

# Custom timeout
python3 tests/integration_test.py --timeout 60
```

### Health Monitoring
```bash
# System status
claude-ns --neuralsync-status

# Detailed system info
python3 -c "
from neuralsync.daemon_manager import get_daemon_manager
import json
print(json.dumps(get_daemon_manager().get_system_info(), indent=2))
"
```

## 🏗️ Architecture

### System Components

1. **Daemon Manager** (`neuralsync.daemon_manager`)
   - Process lifecycle management
   - Health monitoring and recovery
   - Resource limit enforcement

2. **Communication System** (`neuralsync.ultra_comm`)
   - Ultra-low latency message broker
   - Agent discovery and routing
   - Message type definitions

3. **Agent Synchronization** (`neuralsync.agent_sync`)
   - Shared memory management
   - Persona state synchronization
   - CRDT-based conflict resolution

4. **Lifecycle Management** (`neuralsync.agent_lifecycle`)
   - Dynamic agent spawning
   - Task delegation and coordination
   - Resource monitoring

5. **Wrapper Scripts** (`bin/`)
   - CLI integration layer
   - Context injection
   - Command routing

### Data Flow

```
User Command → Wrapper Script → NeuralSync Check → Auto-Launch (if needed)
     ↓
Context Injection → Original CLI → Enhanced Output → Memory Storage
     ↓
Inter-Agent Communication → Task Delegation → Result Sharing
```

### Security Model

- **Sandboxed Execution**: Each agent runs in isolated process
- **Resource Limits**: Configurable CPU, memory, and process limits
- **Auth Tokens**: Optional authentication for API access
- **Emergency Shutdown**: System-wide emergency stop capability

## 🔧 Troubleshooting

### Common Issues

**1. Wrapper not found in PATH**
```bash
# Check if ~/.local/bin is in PATH
echo $PATH | grep -q "$HOME/.local/bin" && echo "OK" || echo "Missing"

# Add to PATH temporarily
export PATH="$HOME/.local/bin:$PATH"

# Permanent fix
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**2. NeuralSync services won't start**
```bash
# Check Python environment
python3 -c "import neuralsync; print('OK')"

# Check configuration
python3 -c "from neuralsync.config import load_config; print(load_config())"

# Manual start
cd NeuralSync2
source .venv/bin/activate
python3 -m neuralsync.daemon_manager start
```

**3. Port conflicts**
```bash
# Check if port 8373 is in use
lsof -i :8373

# Use different port
export NS_PORT=8374
claude-ns --neuralsync-status
```

**4. Permission errors**
```bash
# Fix wrapper script permissions
chmod +x ~/.local/bin/claude-ns ~/.local/bin/codex-ns ~/.local/bin/gemini-ns

# Fix configuration directory
chmod -R 755 ~/.neuralsync
```

### Debug Mode

```bash
# Enable debug logging
export NEURALSYNC_DEBUG=1

# Verbose validation
python3 validate_installation.py --verbose

# Check logs
tail -f ~/.neuralsync/logs/*.log
```

### Getting Help

1. **Run validation**: `python3 validate_installation.py`
2. **Check system status**: `claude-ns --neuralsync-status`
3. **Run integration tests**: `python3 tests/integration_test.py`
4. **Review logs**: `~/.neuralsync/logs/`

## 🗑️ Uninstallation

### Complete Removal
```bash
# Remove everything (configuration, data, scripts)
python3 uninstall_neuralsync.py

# Keep data and configuration
python3 uninstall_neuralsync.py --keep-data

# Remove Python environment too
python3 uninstall_neuralsync.py --remove-venv
```

### Manual Cleanup
```bash
# Remove wrapper scripts
rm ~/.local/bin/claude-ns ~/.local/bin/codex-ns ~/.local/bin/gemini-ns

# Remove configuration (optional)
rm -rf ~/.neuralsync

# Clean shell profile
# Edit ~/.bashrc and remove NeuralSync PATH entries
```

## 🚀 Advanced Usage

### Custom Agent Types

You can extend the system with custom agent types:

```python
# Register custom agent
from neuralsync.agent_lifecycle import get_lifecycle_manager

manager = get_lifecycle_manager()
await manager.spawn_agent(
    agent_type='custom',
    task_description='Custom task',
    context={'custom_config': 'value'}
)
```

### Message Handling

Custom message handlers for inter-agent communication:

```python
from neuralsync.ultra_comm import get_comm_manager

comm_manager = get_comm_manager()

async def handle_custom_message(data):
    # Process custom message
    return {'status': 'processed', 'result': data}

# Register handler
comm_manager.register_message_handler('custom_task', handle_custom_message)
```

### Memory Management

Direct memory operations:

```python
from neuralsync.agent_sync import get_agent_synchronizer

sync = get_agent_synchronizer()

# Add shared memory
memory_id = await sync.add_shared_memory(
    content="Important information",
    kind="fact",
    source_agent="my_agent",
    tags=["important", "shared"]
)

# Retrieve memories
memories = await sync.get_shared_memory(
    query="important information",
    limit=10
)
```

## 📈 Performance Tuning

### Resource Limits

Edit `~/.neuralsync/config.yaml`:

```yaml
# Agent limits
max_agents_per_type:
  claude: 3
  codex: 2  
  gemini: 2
max_total_agents: 5

# Memory settings
memory_cleanup_interval: 3600  # seconds
memory_retention_hours: 24

# Communication settings
heartbeat_interval: 15  # seconds
message_timeout: 30     # seconds
```

### System Optimization

```bash
# Monitor resource usage
claude-ns --neuralsync-status

# Check agent performance
python3 -c "
from neuralsync.agent_lifecycle import get_lifecycle_manager
import json
print(json.dumps(get_lifecycle_manager().get_lifecycle_stats(), indent=2))
"
```

## 🤝 Contributing

This system is designed to be extensible. Key extension points:

1. **New Agent Types**: Add support for additional CLI tools
2. **Message Types**: Define new inter-agent communication patterns  
3. **Memory Backends**: Add support for different storage systems
4. **Authentication**: Implement custom auth mechanisms
5. **Monitoring**: Add custom metrics and monitoring

## 📝 License

Same as NeuralSync2 project license.

---

**🎉 Enjoy seamless AI agent integration with NeuralSync Auto-Launch!**