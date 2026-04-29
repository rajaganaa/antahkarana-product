# 🧠 Antahkarana — Unified Medical AI Reasoning Engine

**Author:** RAJAGANAPATHY M, SRM University  
**Live Demo:** https://rajaganaa.github.io/medassist-frontend  
**Backend:** https://medassist-api-13ls6v.azurewebsites.net/api  

---

## Overview

Antahkarana integrates three research projects into one live product:

| Project | Model | Benchmark |
|---------|-------|-----------|
| NLP Reasoning | Qwen2.5-7B-Instruct via vLLM | HotpotQA 71.3% F1, MMLU 68.4% EM |
| VLM Reasoning | BLIP-2 Flan-T5-XL | 43.2% EM, 70.1% fewer model calls |
| MedAssist RAG | ChromaDB + all-MiniLM-L6-v2 | 5 medical PDFs, FDA API, tools |

---

## Architecture — The 5 Components

Inspired by the Indian philosophical model of the inner instrument (*antaḥkaraṇa*):

```
User Question + Medicine Image
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Step 1: 👁  VISION   — BLIP-2 Flan-T5-XL          │
│              Medicine image → drug name, strength,  │
│              expiry date, manufacturer               │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Step 2: 🧭  MANAS    — Question Router             │
│              Classifies: simple / math / dosage /   │
│              expiry / FDA / verification / comparison│
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Step 3: 📚  CHITTA   — Dense Retrieval             │
│              sentence-transformers + ChromaDB        │
│              Searches 5 medical PDFs                 │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Step 4: 🧠  BUDDHI   — Qwen2.5-7B Reasoner        │
│              Pass 1 (Tarka): Initial reasoning       │
│              Pass 2 (Pramana): Context verification  │
│              Pass 3 (Samsaya): Self-consistency      │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Step 5: ⚖️  AHAMKARA — Confidence Scorer           │
│              0.0-1.0 confidence, HIGH/MED/LOW label  │
│              Decides if Pass2/Pass3 needed           │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Step 6: 👁‍🗨  SAKSHI   — Witness/Verifier           │
│              Hallucination detection + correction    │
│              Token-overlap grounding check           │
└─────────────────────────────────────────────────────┘
         │
         ▼
     ✅ Final Answer + Sources + Disclaimer
```

---

## API Response Format

```json
{
  "request_id": "abc12345",
  "question": "What is the max dose of paracetamol?",
  "total_latency_s": 3.2,

  "vision": {
    "brand_name": "Crocin",
    "generic_name": "Paracetamol",
    "strength": "500mg",
    "form": "Tablet",
    "expiry_date": "Dec 2026",
    "extraction_method": "BLIP-2 Flan-T5-XL"
  },

  "manas": {
    "question_type": "medical",
    "confidence": 0.75,
    "entities": ["Paracetamol"],
    "routing_rationale": "General medical query — routing to RAG + Buddhi"
  },

  "chitta": {
    "retrieved_chunks": [
      {"content": "...", "source": "01_pain_relievers.pdf", "score": 0.87}
    ],
    "num_chunks": 5,
    "retrieval_method": "dense+chromadb"
  },

  "buddhi": {
    "reasoning_steps": ["Step 1: ...", "Step 2: ..."],
    "draft_answer": "The maximum adult dose is 1000mg per dose, up to 4g/day.",
    "pass2_fired": true,
    "pass2_verified": true,
    "pass3_fired": false,
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "latency_s": 2.1
  },

  "ahamkara": {
    "confidence_score": 0.88,
    "confidence_label": "HIGH",
    "pass_level": "Pass2 (Pramana Verification)",
    "needs_retry": false
  },

  "sakshi": {
    "verified": true,
    "corrected": false,
    "hallucination_flags": [],
    "final_answer": "The maximum adult dose of Paracetamol is 1000mg per dose...\n\n📄 01_pain_relievers.pdf",
    "sakshi_summary": "✅ Answer verified and grounded in medical context",
    "medical_disclaimer": "⚠️ This information is for educational purposes only..."
  },

  "final_answer": "...",
  "sources": ["01_pain_relievers.pdf"]
}
```

---

## Folder Structure

```
antahkarana_product/
├── backend/
│   ├── main.py                    ← FastAPI — all endpoints
│   ├── engine/
│   │   ├── manas.py               ← Question router
│   │   ├── chitta.py              ← Dense retrieval + ChromaDB
│   │   ├── buddhi.py              ← Qwen2.5-7B 3-pass reasoner
│   │   ├── ahamkara.py            ← Confidence scorer
│   │   └── sakshi.py              ← Hallucination verifier
│   ├── vision/
│   │   └── blip2_extractor.py     ← BLIP-2 medicine analysis
│   ├── rag/
│   │   └── medassist_rag.py       ← ChromaDB RAG pipeline
│   ├── tools/
│   │   ├── fda_api.py             ← OpenFDA adverse events
│   │   ├── dosage_calc.py         ← Weight-based dosage calculator
│   │   └── expiry_check.py        ← Medicine expiry checker
│   └── requirements.txt
├── frontend/
│   ├── index.html                 ← GitHub Pages — animated reasoning trace
│   └── streamlit_app.py           ← Local demo
├── deployment/
│   ├── gcp_setup.sh               ← GCP L4 GPU setup
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/
│   └── drug_guides/               ← 5 medical PDFs
│       ├── 01_pain_relievers.pdf
│       ├── 02_antibiotics.pdf
│       ├── 03_diabetes_medications.pdf
│       ├── 04_heart_bp_medications.pdf
│       └── 05_digestive_allergy.pdf
└── README.md
```

---

## Quick Start

### Option A — Local (CPU, mock vLLM)

```bash
cd antahkarana_product
cp .env.example .env

# Install dependencies
pip install -r backend/requirements.txt

# Build ChromaDB index
cd backend
MEDASSIST_DATA_DIR=../data/drug_guides CHROMA_PATH=../data/chroma_db \
python3 -c "from rag.medassist_rag import build_index_if_needed; build_index_if_needed()"

# Start backend
uvicorn main:app --host 0.0.0.0 --port 8000

# Open frontend
open ../frontend/index.html   # set API URL to http://localhost:8000

# OR Streamlit
streamlit run ../frontend/streamlit_app.py
```

### Option B — GCP L4 GPU (Full Qwen2.5-7B + BLIP-2)

```bash
# On GCP Workbench instance
git clone https://github.com/<your-repo>/antahkarana_product
cd antahkarana_product

# Run setup script
bash deployment/gcp_setup.sh

# Configure environment
cp .env.example .env
# Edit .env: USE_VLLM=true, USE_BLIP2=true

# Start backend
source /opt/antahkarana_venv/bin/activate
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000

# Update GitHub Pages frontend: change API URL to your GCP instance IP
```

### Option C — Docker

```bash
cd antahkarana_product
cp .env.example .env
docker-compose -f deployment/docker-compose.yml up --build
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/reason` | Full 7-step pipeline (image + question) |
| POST | `/api/search` | Direct ChromaDB search |
| POST | `/api/vision` | BLIP-2 image analysis only |
| GET | `/api/sources` | List available medical PDFs |
| GET | `/health` | Health check |

---

## Research Results (from source projects)

### Antahkarana NLP (Qwen2.5-7B)
| Dataset | EM | F1 | vs CoT |
|---------|----|----|--------|
| HotpotQA | 64.7% | 71.3% | +5.2% F1 |
| MMLU | 68.4% | — | +3.1% EM |
| TruthfulQA | 52.3% | — | +4.8% EM |
| FEVER | 71.2% | — | +2.9% EM |
| SVAMP | 58.1% | — | +3.4% EM |

### Antahkarana VLM (BLIP-2)
- 43.2% Exact Match on VQA benchmarks
- 70.1% fewer model calls vs Self-Consistency
- 5.6% hallucination reduction vs single-pass
- Tested on: VQAv2, GQA, ScienceQA, OKVQA

---

## Citation

```
RAJAGANAPATHY M (2026). Antahkarana: A 5-Component Reasoning Framework 
Inspired by Indian Philosophy for Medical Question Answering.
SRM University, Chennai, India.
Projects: Antahkarana-NLP (Qwen2.5-7B), Antahkarana-VLM (BLIP-2), MedAssist-RAG.
```

---

*"The inner instrument (antaḥkaraṇa) that perceives, stores, reasons, 
evaluates and witnesses — now serving healthcare."*
