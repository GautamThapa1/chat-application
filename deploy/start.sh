#!/bin/sh
# Single-container start for Render: the gRPC ChatService (owns SQLite and the
# background workers) on loopback, plus the FastAPI app on $PORT. If either
# process exits, the container exits so Render restarts it.
set -eu

if ! [ -w "$(dirname "${DB_PATH:-/data/app.db}")" ]; then
  echo "Database directory for DB_PATH=${DB_PATH:-/data/app.db} is not writable." >&2
  exit 1
fi

GRPC_BIND=127.0.0.1 python -m backend.service.main &
SERVICE_PID=$!

# Wait (max 60s) until gRPC accepts connections before serving HTTP.
python - <<'PY'
import grpc
grpc.channel_ready_future(grpc.insecure_channel("127.0.0.1:50051")).result(timeout=60)
PY

# --proxy-headers: trust Render's X-Forwarded-For so rate limits see real
# client IPs. Keep ONE worker (in-memory generation state).
uvicorn backend.orchestrator.main:app --host 0.0.0.0 --port "${PORT:-8000}" \
  --proxy-headers --forwarded-allow-ips='*' &
APP_PID=$!

trap 'kill "$SERVICE_PID" "$APP_PID" 2>/dev/null' TERM INT

# Exit as soon as either process dies.
while kill -0 "$SERVICE_PID" 2>/dev/null && kill -0 "$APP_PID" 2>/dev/null; do
  sleep 2
done
kill "$SERVICE_PID" "$APP_PID" 2>/dev/null || true
wait || true
exit 1
