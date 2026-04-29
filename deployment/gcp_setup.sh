#!/bin/bash
# deployment/gcp_setup.sh
# GCP L4 GPU Setup Script for Antahkarana MedAssist
# Author: RAJAGANAPATHY M, SRM University
#
# Target: NVIDIA L4 24GB VRAM (Workbench instance)
# Run: bash gcp_setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║     ANTAHKARANA — GCP L4 Setup Script               ║"
echo "║     RAJAGANAPATHY M · SRM University                ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[1/8] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip python3-venv git curl wget \
    libgl1-mesa-glx libglib2.0-0 poppler-utils

# ── 2. Python venv ────────────────────────────────────────────────────────────
echo "[2/8] Creating Python virtual environment..."
python3 -m venv /opt/antahkarana_venv
source /opt/antahkarana_venv/bin/activate
pip install --upgrade pip -q

# ── 3. CUDA check ────────────────────────────────────────────────────────────
echo "[3/8] Checking CUDA / L4 GPU..."
if nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "    ✅ GPU detected: ${GPU_NAME} (${GPU_MEM} MiB)"
else
    echo "    ⚠️  No GPU detected — installing CPU-only packages"
    export USE_VLLM=false
fi

# ── 4. PyTorch ────────────────────────────────────────────────────────────────
echo "[4/8] Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

# ── 5. vLLM (GPU only) ───────────────────────────────────────────────────────
if [ "${USE_VLLM:-true}" = "true" ]; then
    echo "[5/8] Installing vLLM for Qwen2.5-7B..."
    pip install vllm -q
    echo "    ✅ vLLM installed"
else
    echo "[5/8] Skipping vLLM (CPU mode)"
fi

# ── 6. Backend dependencies ───────────────────────────────────────────────────
echo "[6/8] Installing backend dependencies..."
pip install -r backend/requirements.txt -q
echo "    ✅ Backend dependencies installed"

# ── 7. Pre-download models ────────────────────────────────────────────────────
echo "[7/8] Pre-downloading models (this takes ~5-10 min first time)..."
python3 - <<'EOF'
from sentence_transformers import SentenceTransformer
print("  Downloading all-MiniLM-L6-v2...")
SentenceTransformer("all-MiniLM-L6-v2")
print("  ✅ all-MiniLM-L6-v2 cached")
EOF

if [ "${USE_VLLM:-true}" = "true" ] && [ "${SKIP_MODEL_DOWNLOAD:-false}" != "true" ]; then
    echo "  Downloading Qwen2.5-7B-Instruct (requires ~15GB)..."
    python3 - <<'EOF'
from transformers import AutoTokenizer
print("  Downloading tokenizer...")
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
print("  ✅ Tokenizer cached (full model downloads lazily via vLLM)")
EOF
fi

# ── 8. Build ChromaDB index ───────────────────────────────────────────────────
echo "[8/8] Building ChromaDB vector index from medical PDFs..."
MEDASSIST_DATA_DIR=./data/drug_guides \
CHROMA_PATH=./data/chroma_db \
python3 - <<'EOF'
from rag.medassist_rag import build_index_if_needed
import sys
sys.path.insert(0, 'backend')
result = build_index_if_needed()
print(f"  {'✅ ChromaDB index built' if result else '⚠️ Index build failed — check data/drug_guides/'}")
EOF

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║     Setup Complete!                                  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Start the backend:"
echo "    source /opt/antahkarana_venv/bin/activate"
echo "    cd backend && uvicorn main:app --host 0.0.0.0 --port 8000"
echo ""
echo "  Frontend: deploy frontend/index.html to GitHub Pages"
echo "  Streamlit: streamlit run frontend/streamlit_app.py"
echo ""
