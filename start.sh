#!/bin/bash

# Movie Recommendation System Startup Script

echo "🎬 Starting Movie Recommendation System..."

# 1. Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠️ Ollama is not installed."
    if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
        echo "🌐 Attempting to install Ollama for Mac/Linux..."
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "❌ Automatic installation not supported for your OS."
        echo "🔗 Please download Ollama manually from: https://ollama.com/download"
        exit 1
    fi
fi

# 2. Check if Ollama server is running
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "🚀 Starting Ollama server in background..."
    ollama serve >/dev/null 2>&1 &
    sleep 5
fi

# 3. Ensure llama3.1 model is pulled
echo "📥 Ensuring llama3.1:latest is available..."
ollama pull llama3.1:latest

# 4. Start FastAPI Backend in background
echo "⚙️ Starting FastAPI Backend on port 8000..."
uv run uvicorn api.app:app --reload --port 8000 &
BACKEND_PID=$!

# Trap to kill background process on exit
trap "echo '🛑 Stopping Backend...'; kill $BACKEND_PID" EXIT

# 5. Start Chainlit UI in foreground
echo "🖥️ Starting Chainlit UI on port 7000..."
echo "🔗 UI: http://localhost:7000"
echo "🔗 API: http://localhost:8000"
uv run chainlit run main.py -w --port 7000
