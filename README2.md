# ⏺ 🧠 What NeuralSync Does When You Run `claude-ns`

When you run `claude-ns`, here's **exactly** what happens:

## 1. 🚀 Launches Claude Code Through NeuralSync Wrapper

- `claude-ns` is an alias/wrapper that launches the regular Claude Code CLI
- But **wraps it** through NeuralSync's `nswrap` script
- Sets environment variables: `TOOL_NAME="claude-code"`, `NS_HOST=127.0.0.1`, `NS_PORT=8373`

## 2. 🧬 Activates Shared Memory System

NeuralSync creates a **unified memory layer** that:
- **Shares context** between ALL your AI tools (Claude, Gemini, Codex)
- **Persists memories** across sessions (remembers previous conversations)
- **Syncs instantly** (<10ms) between different AI instances

## 3. 💾 Three-Tier Memory Architecture

When active, it manages:
- **Hot Memory** (RocksDB): Most recent/important memories, instant access
- **Warm Memory** (SQLite): Older but searchable memories
- **Cold Memory** (NAS): Archived memories for long-term storage

## 4. 🔄 Real-Time Synchronization

- Uses **CRDT** (Conflict-free Replicated Data Types) for perfect sync
- **Unix domain sockets** for ultra-low latency communication
- **Zero-copy operations** for maximum performance

## 5. 🧩 What This Means Practically:

### Without NeuralSync (`claude`):
- Each Claude session starts fresh
- No memory of previous conversations
- No sharing between tools

### With NeuralSync (`claude-ns`):
- Claude **remembers** your previous work
- Can **reference** things you did with Gemini or other tools
- **Deduplicates research** (won't repeat searches)
- **Shares personality/context** across all AI tools
- **Learns from patterns** in your usage

## 6. 🎯 Advanced Features Enabled:

- **Research Deduplication**: Won't repeat identical searches
- **Context Compression**: Intelligently compresses repeated patterns
- **Byzantine Fault Tolerance**: Survives corrupted data/crashes
- **Unleashed Mode**: Can bypass normal restrictions when needed
- **Self-Healing**: Automatically recovers from failures

## 7. 📊 Performance Impact:

- **Startup**: <500ms overhead
- **Memory lookup**: <1ms
- **Inter-tool communication**: <10ms
- **Can handle**: 100K operations/second
- **Scales to**: 10M+ memory entries

---

## 🎭 In Simple Terms:

Think of it like giving Claude (and all your AI tools) a **shared brain** that:
- Never forgets what you've worked on
- Learns from all your interactions
- Shares knowledge between different AI tools
- Gets smarter over time
- Works at lightning speed

> **When you type `claude-ns` instead of `claude`, you're essentially saying:**
> 
> *"Launch Claude with its full memory system activated and connected to all my other AI tools."*

---

### 🔮 The Magic Behind the Scenes

```bash
# Regular Claude (no memory)
$ claude "help me with my project"
# Starts fresh, no context

# NeuralSync Claude (with unified memory)
$ claude-ns "help me with my project"
# Remembers everything, shares with all AI tools
```

### 📡 Connection Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  claude-ns  │────▶│  NeuralSync  │◀────│   gemini    │
└─────────────┘     │    Daemon    │     └─────────────┘
                    └──────────────┘
                           │
                    ┌──────▼──────┐
                    │ Shared Brain │
                    │   (Memory)   │
                    └──────────────┘
```

### 🚄 Speed Comparison

| Operation | Without NeuralSync | With NeuralSync |
|-----------|-------------------|-----------------|
| Startup | Instant | +500ms |
| Context Recall | None | <1ms |
| Research Dedup | Never | Always |
| Cross-tool Sync | Impossible | <10ms |
| Memory Capacity | 0 | 10M+ entries |

---

## 🛠️ Technical Details

### Memory Hierarchy
1. **L1 Cache** - In-memory hot data (nanoseconds)
2. **L2 Storage** - RocksDB persistent (microseconds)
3. **L3 Archive** - SQLite searchable (milliseconds)
4. **L4 Cold** - NAS long-term (seconds)

### Synchronization Protocol
- **CRDT-based** conflict resolution
- **Vector clocks** for ordering
- **Byzantine fault tolerance** for reliability
- **Zero-copy** message passing

### Integration Points
- Unix domain sockets at `/tmp/neuralsync2_*.sock`
- Shared memory segments for large transfers
- QUIC protocol for network sync
- gRPC for service mesh integration

---

## 🎯 Why This Matters

**Without NeuralSync**, every AI session is like meeting someone with amnesia - they're helpful but have no memory of you.

**With NeuralSync**, your AI tools become a unified intelligence that grows smarter with every interaction, never forgets, and seamlessly collaborates across all your tools.

It's not just memory - it's **evolution**.