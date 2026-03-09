#!/bin/bash

# Movie Recommendation System Startup Script

echo "🎬 Starting Movie Recommendation System..."

# Pre-startup cleanup
echo "🧹 Cleaning up any existing processes on ports 7000 and 8000..."
lsof -ti:7000,8000 | xargs kill -9 > /dev/null 2>&1 || true

# 0. Sync dependencies
echo "🔄 Syncing dependencies..."
uv sync

# 1. Check for Together AI API Key
if [ -z "$TOGETHER_API_KEY" ]; then
    if [ -f .env ]; then
        export $(grep -v '^#' .env | xargs)
    fi
    if [ -z "$TOGETHER_API_KEY" ]; then
        echo "❌ TOGETHER_API_KEY is not set."
        echo "⚠️ Please add it to your .env file to enable movie title extraction."
        exit 1
    fi
fi

# 4. Start FastAPI Backend in background
echo "⚙️ Starting FastAPI Backend on port 8000..."
uv run uvicorn api.app:app --reload --port 8000 &
BACKEND_PID=$!

# Trap to kill background process on exit
trap "echo '🛑 Stopping Backend...'; kill $BACKEND_PID" EXIT

# 5. Start Chainlit UI in foreground
echo "🖥️ Starting Chainlit UI on port 7000..."

echo "🔗 UI: http://localhost:7000"
echo "🔗 API: http://localhost:8000/#docs"
uv run chainlit run main.py -w --port 7000
