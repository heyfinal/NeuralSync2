#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export NS_HOST=${NS_HOST:-"127.0.0.1"}
export NS_PORT=${NS_PORT:-"8373"}
uvicorn neuralsync.server:app --host "$NS_HOST" --port "$NS_PORT"
