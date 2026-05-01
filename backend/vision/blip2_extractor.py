"""
vision/blip2_extractor.py — Medicine Image Analyzer (GitHub Models / GPT-4o)
Antahkarana MedAssist Backend

BLIP-2 has been REMOVED. This file now uses GitHub Models (GPT-4o Vision)
following the exact pattern from faculty's vision_extractor.py, adapted for
the Antahkarana module interface (extract_medicine_info entry point preserved).

Config:
  GITHUB_TOKEN   — required, set in .env / docker-compose environment
  USE_BLIP2      — ignored (always treated as false; kept for env-compat)
  USE_VLLM       — ignored here

Author: RAJAGANAPATHY M, SRM University
"""

import os
import re
import base64
import json
import logging
import time
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# ── Client init (lazy, mirrors faculty pattern) ───────────────────────────────
# Faculty uses: client = OpenAI() at module level.
# We do lazy init so missing token doesn't crash on import.

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN env var is not set. "
            "Add it to your .env file or docker-compose environment section."
        )

    from openai import OpenAI
    _client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=token,
    )
    logger.info("[VISION] GitHub Models client initialized (gpt-4o)")
    return _client


# ── Helpers (same as faculty) ─────────────────────────────────────────────────

def _encode_image(image_path: str) -> str:
    """Encode image to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_image_mime(image_path: str) -> str:
    """Get MIME type from file extension."""
    ext = image_path.lower().split(".")[-1]
    return {
        "jpg":  "image/jpeg",
        "jpeg": "image/jpeg",
        "png":  "image/png",
        "gif":  "image/gif",
        "webp": "image/webp",
    }.get(ext, "image/jpeg")


# ── Core extraction (file path) ───────────────────────────────────────────────

def extract_medicine_info(image_path: str) -> Dict:
    """
    Main entry point used by main.py Step 1 and /api/vision endpoint.

    Extracts medicine information from a local image file using GPT-4o Vision
    via GitHub Models. Matches the faculty vision_extractor.py interface exactly.

    Args:
        image_path: Absolute or relative path to the medicine image.

    Returns:
        Dict with keys: brand_name, generic_name, strength, form, composition,
        manufacturer, expiry_date, manufacturing_date, batch_number, warnings,
        drug_name, extraction_method, latency_s
    """
    logger.info(f"[VISION] Analyzing: {os.path.basename(image_path)}")
    t0 = time.time()

    # Enhanced prompt — identical to faculty, plus JSON-only instruction
    prompt = """Analyze this medicine image carefully and extract ALL visible information.

Extract the following details:
1. Brand Name (the product name)
2. Generic/Drug Name (the active ingredient)
3. Strength (e.g., 500mg, 650mg)
4. Form (tablet, capsule, syrup, etc.)
5. Composition (list of active ingredients)
6. Manufacturer (company name)
7. Expiry Date (look for EXP, Expiry, Best Before, Use By - format as seen)
8. Manufacturing Date (look for MFG, Mfg Date, Manufacturing Date)
9. Batch/Lot Number (look for Batch No, Lot No, B.No)
10. Visible Warnings (any warning text visible)

Return ONLY valid JSON in this exact format:
{
    "brand_name": "...",
    "generic_name": "...",
    "strength": "...",
    "form": "...",
    "composition": "...",
    "manufacturer": "...",
    "expiry_date": "...",
    "manufacturing_date": "...",
    "batch_number": "...",
    "warnings": "..."
}

IMPORTANT:
- Use "Not visible" for any field you cannot clearly read
- For dates, write exactly as shown on package (e.g., "Dec 2025", "12/2025", "2025-12")
- Be precise - this information is used for medical safety checks"""

    try:
        client = _get_client()

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    # Faculty puts text first, then image
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{_get_image_mime(image_path)};"
                                f"base64,{_encode_image(image_path)}"
                            ),
                            "detail": "high",   # faculty uses "high" explicitly
                        },
                    },
                ],
            }],
            max_tokens=800,
        )

        text = response.choices[0].message.content

        # Strip markdown fences (faculty pattern)
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        try:
            result = json.loads(text.strip())
        except json.JSONDecodeError:
            logger.warning("[VISION] JSON parse failed, using fallback dict")
            result = {"generic_name": "Unknown", "brand_name": "Unknown", "raw_response": text}

        # Antahkarana-specific fields used by main.py / expiry handler
        result["drug_name"] = result.get("generic_name") or result.get("brand_name", "Unknown")
        result["extraction_method"] = "GPT-4o Vision (GitHub Models)"
        result["latency_s"] = round(time.time() - t0, 3)

        logger.info(
            f"[VISION] Extracted: {result['drug_name']} "
            f"{result.get('strength','?')} in {result['latency_s']}s"
        )
        return result

    except Exception as e:
        logger.error(f"[VISION] Extraction failed: {e}")
        return _empty_result(str(e))


# ── Base64 variant (used by web frontends / Streamlit) ───────────────────────

def extract_from_base64(base64_image: str, image_type: str = "image/jpeg") -> Dict:
    """
    Extract medicine info from a base64-encoded image string.

    Matches faculty's extract_from_base64() signature exactly.
    Used when image arrives as data URL (Streamlit, browser upload).

    Args:
        base64_image: Raw base64 string (no data: prefix).
        image_type:   MIME type, e.g. "image/jpeg" or "image/png".

    Returns:
        Same dict structure as extract_medicine_info().
    """
    logger.info("[VISION] Analyzing base64 image...")
    t0 = time.time()

    prompt = """Analyze this medicine image carefully and extract ALL visible information.

Extract the following details:
1. Brand Name (the product name)
2. Generic/Drug Name (the active ingredient)
3. Strength (e.g., 500mg, 650mg)
4. Form (tablet, capsule, syrup, etc.)
5. Composition (list of active ingredients)
6. Manufacturer (company name)
7. Expiry Date (look for EXP, Expiry, Best Before, Use By)
8. Manufacturing Date (look for MFG, Mfg Date)
9. Batch/Lot Number
10. Visible Warnings

Return ONLY valid JSON:
{
    "brand_name": "...",
    "generic_name": "...",
    "strength": "...",
    "form": "...",
    "composition": "...",
    "manufacturer": "...",
    "expiry_date": "...",
    "manufacturing_date": "...",
    "batch_number": "...",
    "warnings": "..."
}

Use "Not visible" for fields you cannot read clearly."""

    try:
        client = _get_client()

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_type};base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            }],
            max_tokens=800,
        )

        text = response.choices[0].message.content

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        try:
            result = json.loads(text.strip())
        except json.JSONDecodeError:
            result = {"generic_name": "Unknown", "brand_name": "Unknown"}

        result["drug_name"] = result.get("generic_name") or result.get("brand_name", "Unknown")
        result["extraction_method"] = "GPT-4o Vision (GitHub Models)"
        result["latency_s"] = round(time.time() - t0, 3)

        logger.info("[VISION] Base64 extraction complete")
        return result

    except Exception as e:
        logger.error(f"[VISION] Base64 extraction failed: {e}")
        return _empty_result(str(e))


# ── Empty result (used by Antahkarana error paths) ────────────────────────────

def _empty_result(error: str = "") -> Dict:
    return {
        "brand_name":        "Not detected",
        "generic_name":      "Not detected",
        "strength":          "Not detected",
        "form":              "Not detected",
        "composition":       "Not visible",
        "manufacturer":      "Not detected",
        "expiry_date":       "Not visible",
        "manufacturing_date":"Not visible",
        "batch_number":      "Not visible",
        "warnings":          "Not visible",
        "drug_name":         "Unknown",
        "extraction_method": "None",
        "error":             error,
        "latency_s":         0,
    }


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("VISION EXTRACTOR TEST — GitHub Models / GPT-4o")
    print("=" * 60)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            result = extract_medicine_info(path)
            print("\n📋 Full extraction result:")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ Image not found: {path}")
    else:
        print("\nUsage: python blip2_extractor.py <image_path>")
        print("\nNo image provided — showing mock output:\n")
        mock = {
            "brand_name":         "Dolo",
            "generic_name":       "Paracetamol",
            "strength":           "650mg",
            "form":               "Tablet",
            "composition":        "Paracetamol IP 650mg",
            "manufacturer":       "Micro Labs Ltd",
            "expiry_date":        "Dec 2025",
            "manufacturing_date": "Dec 2023",
            "batch_number":       "DL24A001",
            "warnings":           "Keep out of reach of children",
            "drug_name":          "Paracetamol",
            "extraction_method":  "GPT-4o Vision (GitHub Models)",
            "latency_s":          1.23,
        }
        print(json.dumps(mock, indent=2))
