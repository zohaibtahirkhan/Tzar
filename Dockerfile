# =============================================================================
# Tzar Voice Assistant — CPU Docker image (API server mode)
#
# Build:  docker build -t tzar-assistant .
# Run:    docker run -p 127.0.0.1:8000:8000 -v $(pwd)/data:/app/data \
#             -e LLM_OLLAMA_HOST=http://host.docker.internal:11434 tzar-assistant
#
# GPU inference belongs in Ollama (run it natively or as the `ollama` compose
# service); this image talks to it over HTTP. There is no CUDA/ROCm variant.
# =============================================================================

FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libasound2-dev portaudio19-dev \
    build-essential cmake git curl espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt already pins llama-cpp-python; nothing to install separately.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY .env.example .env.example

# /bin/sh has no brace expansion — list the directories out.
RUN mkdir -p data/cache data/logs models AssistantWorkspace

RUN useradd -m -u 1000 -s /bin/bash tzar && chown -R tzar:tzar /app
USER tzar

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    LLM_BACKEND=ollama \
    LLM_OLLAMA_HOST=http://host.docker.internal:11434

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "app.main", "--mode", "server", "--host", "0.0.0.0"]
