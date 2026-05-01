# deployment/Dockerfile — Antahkarana MedAssist
# Clean CPU build — no CUDA, no GPU, no BLIP-2, no vLLM
# Vision: GitHub Models (GPT-4o)   LLM: Groq (GROQ_API_KEY)
# Author: RAJAGANAPATHY M, SRM University

FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgl1-mesa-dri \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY backend/ ./backend/
COPY data/drug_guides/ ./data/drug_guides/

# ── Pre-download embedding model (SentenceTransformer, CPU-only) ──────────────
RUN python -c \
    "from sentence_transformers import SentenceTransformer; \
     SentenceTransformer('all-MiniLM-L6-v2')"

# ── Pre-build ChromaDB index from PDFs ────────────────────────────────────────
ENV MEDASSIST_DATA_DIR=/app/data/drug_guides
ENV CHROMA_PATH=/app/data/chroma_db
RUN cd /app && python -c \
    "import sys; sys.path.insert(0, 'backend'); \
     from rag.medassist_rag import build_index_if_needed; \
     build_index_if_needed()" || true

# ── Runtime env (never GPU, never vLLM, never BLIP-2) ────────────────────────
ENV USE_VLLM=false
ENV USE_BLIP2=false
ENV PORT=8000
ENV PYTHONPATH=/app/backend

EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--app-dir", "backend"]
