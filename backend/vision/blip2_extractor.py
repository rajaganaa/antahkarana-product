"""
vision/blip2_extractor.py — BLIP-2 Medicine Image Analyzer
Antahkarana VLM project adapted for MedAssist product.

Uses BLIP-2 Flan-T5-XL for medicine image understanding.
Exact model architecture from New_VLM_antahkarana_project_2500_samples.
Falls back to GPT-4o Vision (from session2 vision_extractor.py) if BLIP-2 unavailable.
"""

import os
import re
import base64
import logging
import time
from typing import Optional, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)

USE_BLIP2 = os.environ.get("USE_BLIP2", "true").lower() == "true"
OPENAI_API_KEY = os.environ.get("GITHUB_TOKEN", "")

# ── BLIP-2 Singleton ─────────────────────────────────────────────────────────
_blip2_processor = None
_blip2_model = None


def _load_blip2():
    global _blip2_processor, _blip2_model
    if _blip2_model is not None:
        return True

    if not USE_BLIP2:
        return False

    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        import torch

        logger.info("[BLIP2] Loading BLIP-2 Flan-T5-XL...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        _blip2_processor = Blip2Processor.from_pretrained(
            "Salesforce/blip2-flan-t5-xl"
        )
        _blip2_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        logger.info(f"[BLIP2] Model loaded on {device}")
        return True

    except Exception as e:
        logger.warning(f"[BLIP2] Load failed ({e}), will use fallback")
        return False


def _blip2_generate_batch(images, prompts: List[str]) -> List[str]:
    """
    Batch inference with BLIP-2.
    Exact interface from antahkarana_model.py blip2_generate().
    """
    import torch
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    # Process in batches of 4 (L4 GPU budget)
    BATCH_SIZE = 4
    for batch_start in range(0, len(images), BATCH_SIZE):
        batch_imgs = images[batch_start:batch_start + BATCH_SIZE]
        batch_prompts = prompts[batch_start:batch_start + BATCH_SIZE]

        # Ensure PIL images
        pil_imgs = []
        for img in batch_imgs:
            if isinstance(img, (str, Path)):
                pil_imgs.append(Image.open(img).convert("RGB"))
            else:
                pil_imgs.append(img.convert("RGB") if hasattr(img, 'convert') else img)

        inputs = _blip2_processor(
            images=pil_imgs,
            text=batch_prompts,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            output_ids = _blip2_model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=4,
            )

        decoded = _blip2_processor.batch_decode(output_ids, skip_special_tokens=True)
        results.extend(decoded)

    return results


# ── Medicine-specific extraction prompts (from Antahkarana VLM research) ─────

MEDICINE_PROMPTS = [
    "What is the brand name of this medicine?",
    "What is the generic drug name or active ingredient?",
    "What is the strength or dosage (in mg, ml, etc.)?",
    "What is the form of this medicine (tablet, capsule, syrup)?",
    "What is the expiry date or expiration date on this package?",
    "What is the manufacturer or company name?",
    "What warnings or precautions are visible?",
]


def extract_medicine_info_blip2(image_path: str) -> Dict:
    """
    Extract medicine information using BLIP-2 Flan-T5-XL.
    Uses multi-prompt approach from Antahkarana VLM pipeline.
    """
    from PIL import Image

    t0 = time.time()
    loaded = _load_blip2()
    if not loaded:
        logger.info("[BLIP2] Not available, using GPT-4o fallback")
        return extract_medicine_info_gpt4v(image_path)

    try:
        img = Image.open(image_path).convert("RGB")
        images = [img] * len(MEDICINE_PROMPTS)
        responses = _blip2_generate_batch(images, MEDICINE_PROMPTS)

        result = {
            "brand_name": _clean(responses[0]) if len(responses) > 0 else "Not detected",
            "generic_name": _clean(responses[1]) if len(responses) > 1 else "Not detected",
            "strength": _clean(responses[2]) if len(responses) > 2 else "Not detected",
            "form": _clean(responses[3]) if len(responses) > 3 else "Not detected",
            "expiry_date": _clean(responses[4]) if len(responses) > 4 else "Not visible",
            "manufacturer": _clean(responses[5]) if len(responses) > 5 else "Not detected",
            "warnings": _clean(responses[6]) if len(responses) > 6 else "Not visible",
            "extraction_method": "BLIP-2 Flan-T5-XL",
            "latency_s": round(time.time() - t0, 3),
        }

        # Post-process
        result["drug_name"] = result["generic_name"] or result["brand_name"]
        logger.info(f"[BLIP2] Extracted: {result['drug_name']} {result['strength']}")
        return result

    except Exception as e:
        logger.error(f"[BLIP2] Extraction error: {e}")
        return _empty_result(f"BLIP-2 error: {str(e)}")


def extract_medicine_info_gpt4v(image_path: str) -> Dict:
    """
    GPT-4o Vision fallback for medicine extraction.
    From session2_medassist_agent components/vision_extractor.py.
    """
    if not OPENAI_API_KEY:
        logger.warning("[VISION] No OPENAI_API_KEY — returning empty result")
        return _empty_result("No vision API available")

    t0 = time.time()
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=OPENAI_API_KEY
        )

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        ext = image_path.lower().split('.')[-1]
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")

        prompt = """Analyze this medicine/drug package image and extract ALL visible information.

Return ONLY valid JSON:
{
    "brand_name": "...",
    "generic_name": "...",
    "strength": "...",
    "form": "tablet/capsule/syrup/etc",
    "composition": "...",
    "manufacturer": "...",
    "expiry_date": "...",
    "manufacturing_date": "...",
    "batch_number": "...",
    "warnings": "..."
}

Use "Not visible" for fields you cannot read. For dates, write exactly as shown."""

        response = client.chat.completions.create(
            model="gpt-4o",   # GitHub Models supports this ✅
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_data}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=500,
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown fences
        raw = re.sub(r'^```(?:json)?\n?', '', raw).rstrip('`').strip()

        import json
        result = json.loads(raw)
        result["extraction_method"] = "GPT-4o Vision"
        result["latency_s"] = round(time.time() - t0, 3)
        result["drug_name"] = result.get("generic_name") or result.get("brand_name", "")
        return result

    except Exception as e:
        logger.error(f"[GPT4V] Extraction error: {e}")
        return _empty_result(f"GPT-4o Vision error: {str(e)}")


def extract_medicine_info(image_path: str) -> Dict:
    """
    Main entry point. Tries BLIP-2 first, falls back to GPT-4o.
    """
    logger.info(f"[VISION] Analyzing: {os.path.basename(image_path)}")
    return extract_medicine_info_blip2(image_path)


def _clean(text: str) -> str:
    """Clean BLIP-2 output."""
    if not text:
        return "Not detected"
    text = text.strip()
    # Remove common BLIP-2 artifacts
    for bad in ["answer:", "the answer is", "it is", "this is"]:
        if text.lower().startswith(bad):
            text = text[len(bad):].strip()
    return text if text else "Not detected"


def _empty_result(error: str = "") -> Dict:
    return {
        "brand_name": "Not detected",
        "generic_name": "Not detected",
        "strength": "Not detected",
        "form": "Not detected",
        "expiry_date": "Not visible",
        "manufacturer": "Not detected",
        "warnings": "Not visible",
        "drug_name": "Unknown",
        "extraction_method": "None",
        "error": error,
        "latency_s": 0,
    }
