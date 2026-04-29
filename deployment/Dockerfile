# deployment/Dockerfile — Antahkarana MedAssist
# Target: GCP Cloud Run with L4 GPU (or CPU fallback)
# Author: RAJAGANAPATHY M, SRM University

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip python3.11-venv \
    libgl1-mesa-glx libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Copy backend code
COPY backend/ ./backend/
COPY data/drug_guides/ ./data/drug_guides/

# Install dependencies
RUN pip3 install --no-cache-dir -r backend/requirements.txt

# Pre-download embedding model
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Build ChromaDB index
ENV MEDASSIST_DATA_DIR=/app/data/drug_guides
ENV CHROMA_PATH=/app/data/chroma_db
RUN cd backend && python3 -c "import sys; sys.path.insert(0,'backend'); from rag.medassist_rag import build_index_if_needed; build_index_if_needed()" || true

# Environment
ENV USE_VLLM=true
ENV MODEL_ID=Qwen/Qwen2.5-7B-Instruct
ENV PORT=8000
ENV PYTHONPATH=/app/backend

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "backend"]
