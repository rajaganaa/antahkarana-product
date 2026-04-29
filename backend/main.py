"""
backend/main.py — Antahkarana FastAPI Backend
Integrates: NLP Antahkarana (Qwen2.5-7B) + VLM Antahkarana (BLIP-2) + MedAssist RAG

Author: RAJAGANAPATHY M, SRM University
Architecture: GCP L4 GPU (24GB VRAM) + GitHub Pages frontend
"""

import os
import uuid
import logging
import time
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Antahkarana MedAssist API",
    description=(
        "Unified reasoning engine combining Antahkarana NLP (Qwen2.5-7B) + "
        "Antahkarana VLM (BLIP-2) + MedAssist RAG (ChromaDB). "
        "Author: RAJAGANAPATHY M, SRM University."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-loaded components ────────────────────────────────────────────────────
_manas = None
_chitta = None
_buddhi = None
_ahamkara = None
_sakshi = None

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/tmp/antahkarana_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_components():
    global _manas, _chitta, _buddhi, _ahamkara, _sakshi
    if _manas is None:
        logger.info("[INIT] Loading Antahkarana components...")
        from engine.manas import Manas
        from engine.chitta import Chitta
        from engine.buddhi import Buddhi
        from engine.ahamkara import Ahamkara
        from engine.sakshi import Sakshi
        _manas = Manas()
        _chitta = Chitta()
        _buddhi = Buddhi()
        _ahamkara = Ahamkara()
        _sakshi = Sakshi()
        logger.info("[INIT] All Antahkarana components ready")
    return _manas, _chitta, _buddhi, _ahamkara, _sakshi


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Pre-build ChromaDB index at startup."""
    logger.info("[STARTUP] Antahkarana MedAssist backend starting...")
    try:
        from rag.medassist_rag import build_index_if_needed
        build_index_if_needed()
        logger.info("[STARTUP] ChromaDB index ready")
    except Exception as e:
        logger.warning(f"[STARTUP] ChromaDB index build deferred: {e}")


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Antahkarana MedAssist",
        "author": "RAJAGANAPATHY M, SRM University",
        "version": "1.0.0",
        "components": {
            "manas": "Question Router",
            "chitta": "Dense Retrieval + ChromaDB",
            "buddhi": "Qwen2.5-7B Reasoner",
            "ahamkara": "Confidence Scorer",
            "sakshi": "Hallucination Verifier",
            "blip2": "Medicine Image Analyzer",
        },
    }


@app.get("/")
async def root():
    return {"message": "Antahkarana MedAssist API — see /docs or /health"}


# ── Main Reasoning Endpoint ───────────────────────────────────────────────────

@app.post("/api/reason")
async def reason(
    question: str = Form(...),
    image: Optional[UploadFile] = File(None),
):
    """
    Full 7-step Antahkarana reasoning pipeline:
    Step 1: Vision (BLIP-2 medicine analysis)
    Step 2: Manas (question routing)
    Step 3: Chitta (dense retrieval from ChromaDB)
    Step 4: Buddhi (Qwen2.5-7B reasoning)
    Step 5: Ahamkara (confidence scoring)
    Step 6: Sakshi (verification + hallucination correction)
    Step 7: Tool calls (FDA/dosage/expiry if triggered)
    """
    request_id = str(uuid.uuid4())[:8]
    t_total = time.time()
    logger.info(f"[{request_id}] New request: {question[:80]}")

    manas, chitta, buddhi, ahamkara, sakshi = get_components()

    # ── STEP 1: Vision ───────────────────────────────────────────────────────
    vision_result = None
    image_path = None

    if image and image.filename:
        try:
            image_path = UPLOAD_DIR / f"{request_id}_{image.filename}"
            contents = await image.read()
            with open(image_path, "wb") as f:
                f.write(contents)

            from vision.blip2_extractor import extract_medicine_info
            vision_result = extract_medicine_info(str(image_path))
            logger.info(f"[{request_id}] Vision: {vision_result.get('drug_name', 'unknown')}")

            # If question doesn't mention a drug but image shows one, inject it
            drug_name = vision_result.get("generic_name") or vision_result.get("brand_name", "")
            if drug_name and drug_name not in ["Not detected", "Not visible"] and drug_name not in question:
                question = f"[About {drug_name}] {question}"

        except Exception as e:
            logger.warning(f"[{request_id}] Vision failed: {e}")
            vision_result = {"error": str(e), "extraction_method": "failed"}

    # ── STEP 2: Manas ────────────────────────────────────────────────────────
    manas_result = manas.get_routing_info(question)
    q_type = manas_result["question_type"]
    entities = manas_result["entities"]
    logger.info(f"[{request_id}] Manas: {q_type} (conf={manas_result['confidence']})")

    # ── STEP 3: Tool shortcuts (dosage / expiry / FDA) ───────────────────────
    tool_result = None
    from engine.manas import QType

    if q_type == QType.DOSAGE:
        tool_result = await _handle_dosage(question, vision_result)
    elif q_type == QType.EXPIRY:
        tool_result = await _handle_expiry(question, vision_result)
    elif q_type == QType.FDA:
        tool_result = await _handle_fda(entities, vision_result)

    # ── STEP 4: Chitta (retrieval) ────────────────────────────────────────────
    # Always retrieve for RAG context, even if tool triggered
    chitta_result = chitta.retrieve(question, entities, k=5)
    logger.info(f"[{request_id}] Chitta: {chitta_result['num_chunks']} chunks")

    # ── STEP 5: Buddhi (reasoning) ───────────────────────────────────────────
    buddhi_result = buddhi.reason(
        question=question,
        context_str=chitta_result["context_str"],
        q_type=q_type,
        medicine_info=vision_result,
    )
    logger.info(f"[{request_id}] Buddhi: pass={buddhi_result['pass2_fired']} latency={buddhi_result['latency_s']}s")

    # ── STEP 6: Ahamkara (confidence) ────────────────────────────────────────
    ahamkara_result = ahamkara.score(buddhi_result, chitta_result, question)
    logger.info(f"[{request_id}] Ahamkara: {ahamkara_result['confidence_score']} ({ahamkara_result['confidence_label']})")

    # ── STEP 7: Sakshi (verification) ────────────────────────────────────────
    sakshi_result = sakshi.verify(
        question=question,
        draft_answer=buddhi_result["draft_answer"],
        context_str=chitta_result["context_str"],
        sources=chitta_result["sources"],
        buddhi_result=buddhi_result,
        ahamkara_result=ahamkara_result,
    )
    logger.info(f"[{request_id}] Sakshi: verified={sakshi_result['verified']}")

    total_latency = round(time.time() - t_total, 3)

    # ── Assemble full API response ────────────────────────────────────────────
    response = {
        "request_id": request_id,
        "question": question,
        "total_latency_s": total_latency,

        # Step 1: Vision
        "vision": vision_result,

        # Step 2: Manas
        "manas": manas_result,

        # Step 3: Chitta
        "chitta": {
            "retrieved_chunks": chitta_result["retrieved_chunks"],
            "scores": [c.get("score", 0) for c in chitta_result["retrieved_chunks"]],
            "num_chunks": chitta_result["num_chunks"],
            "retrieval_method": chitta_result["retrieval_method"],
        },

        # Step 4: Buddhi
        "buddhi": {
            "reasoning_steps": buddhi_result["reasoning_steps"],
            "draft_answer": buddhi_result["draft_answer"],
            "pass1_answer": buddhi_result["pass1_answer"],
            "pass2_fired": buddhi_result["pass2_fired"],
            "pass2_verified": buddhi_result["pass2_verified"],
            "pass3_fired": buddhi_result["pass3_fired"],
            "model": buddhi_result["model"],
            "latency_s": buddhi_result["latency_s"],
        },

        # Step 5: Ahamkara
        "ahamkara": ahamkara_result,

        # Step 6: Sakshi
        "sakshi": {
            "verified": sakshi_result["verified"],
            "corrected": sakshi_result["corrected"],
            "hallucination_flags": sakshi_result["hallucination_flags"],
            "correction_note": sakshi_result["correction_note"],
            "final_answer": sakshi_result["final_answer"],
            "sakshi_summary": sakshi_result["sakshi_summary"],
            "medical_disclaimer": sakshi_result["medical_disclaimer"],
        },

        # Step 7: Tool results
        "tool_result": tool_result,

        # Top-level final answer + sources
        "final_answer": sakshi_result["final_answer"],
        "sources": chitta_result["sources"],
    }

    # Cleanup uploaded image
    if image_path and image_path.exists():
        try:
            image_path.unlink()
        except Exception:
            pass

    return JSONResponse(content=response)


# ── Tool Handlers ─────────────────────────────────────────────────────────────

async def _handle_dosage(question: str, vision_result: Optional[dict]) -> dict:
    try:
        import re
        from tools.dosage_calc import calculate_dosage, normalize_drug_name, DOSAGE_GUIDELINES

        # Extract parameters from question
        weight_m = re.search(r'(\d+(?:\.\d+)?)\s*(?:kg|kilogram)', question, re.IGNORECASE)
        weight = float(weight_m.group(1)) if weight_m else 70.0

        age_group = "adult"
        if any(w in question.lower() for w in ["child", "kid", "pediatric", "baby", "infant"]):
            age_group = "child"
        elif any(w in question.lower() for w in ["elderly", "old", "senior", "geriatric"]):
            age_group = "elderly"

        # Drug from vision or question
        drug = "paracetamol"
        if vision_result:
            d = vision_result.get("generic_name") or vision_result.get("brand_name", "")
            if d and d not in ["Not detected", "Not visible"]:
                drug = d

        result_text = calculate_dosage.invoke({"drug": drug, "weight_kg": weight, "age_group": age_group})
        return {"tool": "dosage_calculator", "drug": drug, "weight_kg": weight, "age_group": age_group, "result": result_text}
    except Exception as e:
        return {"tool": "dosage_calculator", "error": str(e)}


async def _handle_expiry(question: str, vision_result: Optional[dict]) -> dict:
    try:
        from tools.expiry_check import check_medicine_expiry

        expiry_date = None
        if vision_result:
            expiry_date = vision_result.get("expiry_date")
            if expiry_date in ["Not visible", "Not detected", None]:
                expiry_date = None

        if not expiry_date:
            import re
            m = re.search(
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}\b'
                r'|\b\d{1,2}[/\-]\d{4}\b|\b\d{4}[/\-]\d{1,2}\b',
                question, re.IGNORECASE
            )
            expiry_date = m.group(0) if m else "Unknown"

        if expiry_date == "Unknown":
            return {"tool": "expiry_checker", "error": "No expiry date found in image or question"}

        result_text = check_medicine_expiry.invoke(expiry_date)
        return {"tool": "expiry_checker", "expiry_date": expiry_date, "result": result_text}
    except Exception as e:
        return {"tool": "expiry_checker", "error": str(e)}


async def _handle_fda(entities: list, vision_result: Optional[dict]) -> dict:
    try:
        from tools.fda_api import get_reaction_counts

        drug = None
        if vision_result:
            drug = vision_result.get("generic_name")
            if drug in ["Not detected", "Not visible", None]:
                drug = None

        if not drug and entities:
            drug = entities[0]

        if not drug:
            return {"tool": "fda_api", "error": "No drug name identified"}

        data = get_reaction_counts(drug, top_n=10)
        reactions = []
        if data and "results" in data:
            reactions = [
                {"reaction": r.get("term", ""), "count": r.get("count", 0)}
                for r in data["results"][:10]
            ]
        return {"tool": "fda_api", "drug": drug, "reactions": reactions}
    except Exception as e:
        return {"tool": "fda_api", "error": str(e)}


# ── Additional Utility Endpoints ──────────────────────────────────────────────

@app.get("/api/sources")
async def list_sources():
    """List available medical PDF sources."""
    data_dir = Path(os.environ.get("MEDASSIST_DATA_DIR", "./data/drug_guides"))
    pdfs = list(data_dir.glob("*.pdf")) if data_dir.exists() else []
    return {"sources": [p.name for p in pdfs], "count": len(pdfs)}


@app.post("/api/search")
async def search(query: str = Form(...)):
    """Direct ChromaDB search without full reasoning pipeline."""
    from rag.medassist_rag import search_drug_database
    chunks = search_drug_database(query, k=5)
    return {"query": query, "results": chunks, "count": len(chunks)}


@app.post("/api/vision")
async def vision_only(image: UploadFile = File(...)):
    """Standalone BLIP-2 medicine image analysis."""
    image_path = UPLOAD_DIR / f"{uuid.uuid4()}_{image.filename}"
    contents = await image.read()
    with open(image_path, "wb") as f:
        f.write(contents)

    try:
        from vision.blip2_extractor import extract_medicine_info
        result = extract_medicine_info(str(image_path))
        return result
    finally:
        if image_path.exists():
            image_path.unlink()


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )