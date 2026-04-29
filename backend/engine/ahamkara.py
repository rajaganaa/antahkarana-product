"""
engine/ahamkara.py — Ahamkara: Confidence Scorer
Antahkarana v16 adapted for MedAssist product.

Ahamkara is ego/self-identity in Indian philosophy.
It evaluates confidence in the answer and decides if retry is needed.
Logic from antahkarana_QWEN_500 confidence scoring + VLM project Ahamkara.
"""

import re
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


APOLOGY_PATTERNS = [
    "i don't know", "i cannot", "i am not sure", "no information",
    "sorry", "cannot determine", "unclear", "i'm not certain",
    "not available", "please consult", "not enough", "insufficient",
]

UNCERTAINTY_PATTERNS = [
    "may", "might", "could", "possibly", "perhaps", "unclear",
    "not sure", "approximately", "roughly", "generally", "usually",
]

HIGH_CONFIDENCE_MARKERS = [
    "is", "are", "contains", "the dose", "the maximum", "should not",
    "contraindicated", "mg per", "mg/kg", "daily",
]


class Ahamkara:
    """
    Ahamkara — Confidence scorer and retry controller.
    
    Scoring logic:
    - Pass 1 clean answer + context verified → 0.90
    - Pass 2 (Pramana) revised → 0.75  
    - Pass 3 (Samsaya) self-consistency used → vote fraction
    - Bad/uncertain answer → 0.30-0.50 (triggers retry)
    """

    CONFIDENCE_LOW    = 0.50
    CONFIDENCE_MEDIUM = 0.70
    CONFIDENCE_HIGH   = 0.85

    def score(
        self,
        buddhi_result: dict,
        chitta_result: dict,
        question: str,
    ) -> dict:
        """
        Compute Ahamkara confidence score.
        Returns scoring trace for API.
        """
        draft_answer = buddhi_result.get("draft_answer", "")
        pass2_verified = buddhi_result.get("pass2_verified", True)
        pass2_fired = buddhi_result.get("pass2_fired", False)
        pass3_fired = buddhi_result.get("pass3_fired", False)
        num_chunks = chitta_result.get("num_chunks", 0)

        # Start with base confidence
        base_score = self._compute_base_score(
            draft_answer,
            pass2_verified,
            pass2_fired,
            pass3_fired,
            num_chunks,
        )

        # Apply lexical modifiers
        modifier = self._compute_lexical_modifier(draft_answer)
        final_score = min(1.0, max(0.1, base_score + modifier))

        # Determine pass level
        if pass3_fired:
            pass_level = "Pass3 (Self-Consistency)"
        elif pass2_fired:
            pass_level = "Pass2 (Pramana Verification)"
        else:
            pass_level = "Pass1 (Tarka)"

        needs_retry = final_score < self.CONFIDENCE_LOW
        confidence_label = self._get_label(final_score)

        logger.info(
            f"[AHAMKARA] score={final_score:.3f} label={confidence_label} "
            f"pass={pass_level} needs_retry={needs_retry}"
        )

        return {
            "confidence_score": round(final_score, 3),
            "confidence_label": confidence_label,
            "pass_level": pass_level,
            "needs_retry": needs_retry,
            "score_breakdown": {
                "base_score": round(base_score, 3),
                "lexical_modifier": round(modifier, 3),
                "context_available": num_chunks > 0,
                "pramana_verified": pass2_verified,
                "samsaya_used": pass3_fired,
            },
        }

    def _compute_base_score(
        self,
        answer: str,
        pass2_verified: bool,
        pass2_fired: bool,
        pass3_fired: bool,
        num_chunks: int,
    ) -> float:
        # No context retrieved → lower confidence
        if num_chunks == 0:
            base = 0.45
        else:
            base = 0.90

        # Pramana corrections reduce confidence (answer was wrong in pass1)
        if pass2_fired and not pass2_verified:
            base = min(base, 0.75)
        elif pass2_fired and pass2_verified:
            base = min(base, 0.85)

        # Self-consistency used → uncertain
        if pass3_fired:
            base = min(base, 0.70)

        # Bad answer pattern → low
        if self._is_bad(answer):
            base = 0.30

        return base

    def _compute_lexical_modifier(self, answer: str) -> float:
        """Small +/- based on answer text quality."""
        if not answer:
            return -0.20

        ans_lower = answer.lower()
        modifier = 0.0

        # Uncertainty language → reduce
        uncertainty_count = sum(1 for p in UNCERTAINTY_PATTERNS if p in ans_lower)
        modifier -= 0.02 * uncertainty_count

        # High-confidence markers → boost slightly
        hc_count = sum(1 for p in HIGH_CONFIDENCE_MARKERS if p in ans_lower)
        modifier += 0.01 * min(hc_count, 3)

        # Short answer → slight boost (specific vs verbose)
        words = len(answer.split())
        if 3 <= words <= 30:
            modifier += 0.02
        elif words > 100:
            modifier -= 0.03

        return modifier

    def _is_bad(self, answer: str) -> bool:
        if not answer or len(answer.strip()) < 2:
            return True
        return any(p in answer.lower() for p in APOLOGY_PATTERNS)

    def _get_label(self, score: float) -> str:
        if score >= 0.85:
            return "HIGH"
        elif score >= 0.65:
            return "MEDIUM"
        elif score >= 0.45:
            return "LOW"
        else:
            return "VERY_LOW"
