FROM python:3.13-slim

# Install system dependencies required for uv and compiling
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (sync without dev dependencies)
RUN uv sync --no-dev --frozen

# Copy project files
COPY . .

# Expose port for Chainlit (7000)
EXPOSE 7000

RUN echo '#!/bin/bash\n\
export FASTAPI_BASE_URL=http://127.0.0.1:8080\n\
uv run uvicorn api.app:app --host 127.0.0.1 --port 8080 &\n\
PORT=${PORT:-7000}\n\
uv run chainlit run main.py --host 0.0.0.0 --port $PORT\n\
' > /app/run.sh && chmod +x /app/run.sh

CMD ["/app/run.sh"]
