#!/usr/bin/env bash
set -euo pipefail
TOOL_NAME="claude-code" NS_HOST="${NS_HOST:-127.0.0.1}" NS_PORT="${NS_PORT:-8373}" exec "$(dirname "$0")/../nswrap" -- "$@"
