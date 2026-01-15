#!/bin/bash

# Devin Agent Startup Script
# This script starts both the backend and frontend servers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

BACKEND_PORT=8000
FRONTEND_PORT=3000

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    echo ""
    echo "Shutting down servers..."
    
    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill "$BACKEND_PID" 2>/dev/null || true
        echo "Backend server stopped."
    fi
    
    if [ -n "$FRONTEND_PID" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        kill "$FRONTEND_PID" 2>/dev/null || true
        echo "Frontend server stopped."
    fi
    
    echo "All servers stopped. Goodbye!"
    exit 0
}

trap cleanup SIGINT SIGTERM

check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

echo "========================================"
echo "       Devin Agent Startup Script       "
echo "========================================"
echo ""

if check_port $BACKEND_PORT; then
    echo "Warning: Port $BACKEND_PORT is already in use."
    echo "Please stop the existing process or change the port."
    exit 1
fi

if check_port $FRONTEND_PORT; then
    echo "Warning: Port $FRONTEND_PORT is already in use."
    echo "Please stop the existing process or change the port."
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "Warning: .env file not found!"
    echo "Please create a .env file with your API keys."
    echo "You can copy from .env.example: cp .env.example .env"
    echo ""
fi

echo "Starting backend server on port $BACKEND_PORT..."
cd "$BACKEND_DIR"
python3 -m uvicorn main:app --host 0.0.0.0 --port $BACKEND_PORT &
BACKEND_PID=$!

sleep 2

if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "Error: Backend server failed to start."
    exit 1
fi

echo "Backend server started (PID: $BACKEND_PID)"
echo ""

echo "Starting frontend server on port $FRONTEND_PORT..."
cd "$FRONTEND_DIR"
python3 -m http.server $FRONTEND_PORT &
FRONTEND_PID=$!

sleep 1

if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    echo "Error: Frontend server failed to start."
    cleanup
    exit 1
fi

echo "Frontend server started (PID: $FRONTEND_PID)"
echo ""

echo "========================================"
echo "         Servers are running!           "
echo "========================================"
echo ""
echo "Frontend: http://localhost:$FRONTEND_PORT"
echo "Backend:  http://localhost:$BACKEND_PORT"
echo ""
echo "Press Ctrl+C to stop all servers."
echo ""

if command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:$FRONTEND_PORT" 2>/dev/null &
elif command -v open &> /dev/null; then
    open "http://localhost:$FRONTEND_PORT" 2>/dev/null &
fi

wait
