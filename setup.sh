#!/usr/bin/env bash
# =============================================================================
# Local Voice Assistant — Setup Script
# Tested on Ubuntu 24.04 | i5-1235U | 16GB RAM
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  Local Voice Assistant Setup"
echo "============================================"

# ─── System dependencies ────────────────────────────────────────────────────
echo ""
echo "[1/7] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    ffmpeg \
    portaudio19-dev \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libsndfile1 \
    libasound2-dev \
    espeak-ng

# ─── Python virtual environment ──────────────────────────────────────────────
echo ""
echo "[2/7] Creating Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate
pip3.11 install --upgrade pip setuptools wheel

# ─── llama.cpp ───────────────────────────────────────────────────────────────
echo ""
echo "[3/7] Building llama.cpp..."
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
fi
cd llama.cpp
git pull
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc)"
cd ..

echo "  llama.cpp built at: llama.cpp/build/bin/llama-cli"

# ─── Python packages ─────────────────────────────────────────────────────────
echo ""
echo "[4/7] Installing Python packages..."

# llama-cpp-python (CPU-only build)
CMAKE_ARGS="-DLLAMA_BLAS=OFF -DLLAMA_CUBLAS=OFF" \
    pip install llama-cpp-python --no-cache-dir

pip install -r requirements.txt

# Download Whisper model weights
python3 -c "
from faster_whisper import WhisperModel
print('Downloading Whisper small model...')
WhisperModel('small', device='cpu', compute_type='int8')
print('Whisper model ready.')
"

# Download OpenWakeWord models
python3 -c "
import openwakeword
openwakeword.utils.download_models()
print('Wake word models downloaded.')
"

# ─── Directory structure ─────────────────────────────────────────────────────
echo ""
echo "[5/7] Creating directory structure..."
mkdir -p models data/{conversations,cache,logs} AssistantWorkspace

# ─── Model download ──────────────────────────────────────────────────────────
echo ""
echo "[6/7] LLM Model download..."
MODEL_PATH="models/qwen2.5-3b-instruct-q4_k_m.gguf"
if [ ! -f "$MODEL_PATH" ]; then
    echo "  Downloading Qwen2.5-3B-Instruct-Q4_K_M.gguf..."
    echo "  This is ~2GB, please wait..."
    wget -q --show-progress \
        "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf" \
        -O "$MODEL_PATH"
    echo "  Model downloaded: $MODEL_PATH"
else
    echo "  Model already exists: $MODEL_PATH"
fi

# ─── .env file ───────────────────────────────────────────────────────────────
echo ""
echo "[7/7] Creating .env file..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  Created .env from template. Edit it to customise settings."
else
    echo "  .env already exists — skipping."
fi

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Quick start:"
echo "    source venv/bin/activate"
echo "    python -m app.main --mode terminal    # text chat"
echo "    python -m app.main --mode voice       # voice mode"
echo "    python -m app.main --mode server      # API server"
echo "============================================"
