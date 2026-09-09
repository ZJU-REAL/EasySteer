#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

PYTHON=${EASYSTEER_PYTHON:-python3}
BACKEND_PORT=${EASYSTEER_BACKEND_PORT:-5000}
FRONTEND_PORT=${EASYSTEER_FRONTEND_PORT:-8111}
SERVER_HOST=${EASYSTEER_HOST:-127.0.0.1}

if [[ ! -f app/dist/index.html ]]; then
    echo 'Build the UI first: cd frontend/app && npm ci && npm run build' >&2
    exit 1
fi
if ! "$PYTHON" -c 'import gunicorn' 2>/dev/null; then
    echo 'Install frontend/requirements.txt in your EasySteer Python environment first.' >&2
    exit 1
fi

binds=(--bind "$SERVER_HOST:$FRONTEND_PORT")
if [[ "$BACKEND_PORT" != "$FRONTEND_PORT" ]]; then
    binds+=(--bind "$SERVER_HOST:$BACKEND_PORT")
fi

echo "Starting EasySteer UI and job API at http://$SERVER_HOST:$FRONTEND_PORT"
echo 'Generation requires a separate vllm-steer server; set its URL in the UI.'
# Job state and model instances live in this worker. Threads allow status
# polling while a job runs; multiple workers would have independent state.
# exec lets Gunicorn own startup errors, signals, and worker cleanup.
exec "$PYTHON" -m gunicorn --workers 1 --threads 8 --timeout 0 \
    "${binds[@]}" --access-logfile - --error-logfile - app:app
