#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate || true
(pgrep -f "uvicorn neuralsync.server" && kill -9 $(pgrep -f "uvicorn neuralsync.server") ) || true
./run.sh &
PID=$!
sleep 1
./nsctl health
./nsctl persona set "Direct, concise senior engineer persona."
./nsctl remember --text "User prefers remote/hybrid roles until ankle fully heals." --kind "fact" --scope global --confidence 0.9
./nsctl recall "ankle surgery" -k 3
printf '@remember: kind=note scope=global "Loves minimal elegant code."\n' | ./nswrap -- cat >/dev/null
kill $PID || true
echo "✅ smoke OK"
