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
echo "[2/3] Starting backend server..."
python3 app.py &
BACKEND_PID=$!

# Wait for server to start
echo "[*] Waiting for server to start..."
sleep 3

echo ""
echo "[3/3] Starting frontend server..."
python3 -m http.server $FRONTEND_PORT &
FRONTEND_PID=$!

# Wait for frontend server to start
sleep 2

echo ""
echo "========================================"
echo "Startup Complete!"
echo ""
echo "Backend API:   http://localhost:$BACKEND_PORT"
echo "Frontend UI:   http://localhost:$FRONTEND_PORT/"
echo ""
echo "Opening browser..."
echo "========================================"

# Open browser based on OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open http://localhost:$FRONTEND_PORT/index.html
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open http://localhost:$FRONTEND_PORT/index.html 2>/dev/null || \
    sensible-browser http://localhost:$FRONTEND_PORT/index.html 2>/dev/null || \
    echo "Please manually open browser and visit: http://localhost:$FRONTEND_PORT/index.html"
fi

echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for user interrupt
trap "echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT TERM
wait 