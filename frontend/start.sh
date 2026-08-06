#!/bin/bash

echo "========================================"
echo "EasySteer - Steer Vector Control Panel"
echo "========================================"
echo ""

# Configuration (can be overridden by environment variables)
BACKEND_PORT=${EASYSTEER_BACKEND_PORT:-5000}
FRONTEND_PORT=${EASYSTEER_FRONTEND_PORT:-8111}

# Check if Python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not detected. Please install Python3 first."
    exit 1
fi

# Install dependencies
echo "[1/3] Checking and installing dependencies..."
pip3 install -r requirements.txt

echo ""
echo "[2/3] Starting job backend (extraction / training / SAE)..."
python3 app.py &
BACKEND_PID=$!

# Wait for server to start
echo "[*] Waiting for server to start..."
sleep 3

echo ""
echo "[3/3] Starting web UI..."
# The web UI is the Vite app in app/; serve the production build if present.
FRONTEND_PID=""
if [ -d "app/dist" ]; then
    (cd app/dist && python3 -m http.server "$FRONTEND_PORT") &
    FRONTEND_PID=$!
    FRONTEND_URL="http://localhost:$FRONTEND_PORT/"
else
    echo "[WARN] app/dist not found. Build the UI first:"
    echo "       cd app && npm install && npm run build"
    echo "       (or run it in dev mode: cd app && npm run dev)"
    FRONTEND_URL="(not started)"
fi

sleep 1

echo ""
echo "========================================"
echo "Startup Complete!"
echo ""
echo "Backend API:   http://localhost:$BACKEND_PORT"
echo "Frontend UI:   $FRONTEND_URL"
echo ""
echo "Note: text generation goes through the vllm-steer OpenAI-compatible"
echo "server (configure its URL in the UI), not this backend."
echo "========================================"

echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for user interrupt
trap "echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
