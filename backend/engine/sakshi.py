"""
engine/sakshi.py — Sakshi: Witness/Verifier
Antahkarana v16 adapted for MedAssist product.

Sakshi means 'witness' in Sanskrit — it observes and corrects.
Verifies the final answer against retrieved context.
Detects and corrects hallucinations.
Logic from antahkarana_QWEN_500 Pramana system + VLM Sakshi class.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Medical safety: always append disclaimer for medical answers
MEDICAL_DISCLAIMER = (
    "⚠️ This information is for educational purposes only. "
    "Always consult a qualified healthcare professional before taking any medication."
)

# Hallucination indicators — answer claims facts not in context
HALLUCINATION_INDICATORS = [
    r'\b(?:always|never|definitely|absolutely|certainly|guaranteed)\b',
    r'\b\d+\s*(?:mg|ml|g)\b.*\b(?:exactly|precisely)\b',  # Overly precise doses
    r'\bcures?\b',          # Claiming cures
    r'\b100%\b',            # Absolute percentages
]


class Sakshi:
    """
    Sakshi — The witness verifier.
    
    Checks if the answer is grounded in context.
    Detects potential hallucinations.
    Applies medical safety corrections.
    Records all decisions in a structured log.
    """

    def __init__(self):
        self.log: List[Dict] = []

    def verify(
        self,
        question: str,
        draft_answer: str,
        context_str: str,
        sources: List[str],
        buddhi_result: dict,
        ahamkara_result: dict,
    ) -> dict:
        """
        Sakshi verification pipeline.
        Returns complete verification trace.
        """
        is_verified, verification_reason = self._check_grounding(
            draft_answer, context_str
        )

        hallucination_flags = self._detect_hallucinations(draft_answer, context_str)

        corrected_answer, was_corrected, correction_note = self._apply_corrections(
            draft_answer, context_str, hallucination_flags, is_verified
        )

        final_answer = self._apply_safety_formatting(
            corrected_answer, question, sources
        )

        result = {
            "verified": is_verified,
            "verification_reason": verification_reason,
            "hallucination_flags": hallucination_flags,
            "corrected": was_corrected,
            "correction_note": correction_note if was_corrected else None,
            "final_answer": final_answer,
            "sources_used": sources,
            "medical_disclaimer": MEDICAL_DISCLAIMER,
            "sakshi_summary": self._summarize(is_verified, hallucination_flags, was_corrected),
        }

        # Record in Sakshi log (matches VLM project Sakshi.record() pattern)
        self.log.append({
            "question": question[:100],
            "verified": is_verified,
            "hallucination_flags": len(hallucination_flags),
            "corrected": was_corrected,
            "confidence": ahamkara_result.get("confidence_score", 0),
        })

        logger.info(
            f"[SAKSHI] verified={is_verified} flags={len(hallucination_flags)} "
            f"corrected={was_corrected}"
        )

        return result

    def _check_grounding(
        self, answer: str, context: str
    ) -> Tuple[bool, str]:
        """
        Check if answer is grounded in retrieved context.
        Uses token overlap approach from Antahkarana Chitta.
        """
        if not context.strip():
            return False, "No context available — answer from model prior knowledge"

        if not answer.strip():
            return False, "Empty answer"

        # Token overlap check (exact from Antahkarana system.py)
        ans_tokens = set(answer.lower().split())
        ctx_tokens = set(context.lower().split())
        # Remove stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for"}
        ans_tokens -= stop_words
        ctx_tokens -= stop_words

        if not ans_tokens:
            return True, "Answer verified (no content tokens to check)"

        overlap = len(ans_tokens & ctx_tokens) / len(ans_tokens)

        if overlap > 0.35:
            return True, f"Answer grounded in context (token overlap: {overlap:.1%})"
        elif overlap > 0.15:
            return True, f"Partially grounded in context (token overlap: {overlap:.1%})"
        else:
            return False, f"Low context overlap ({overlap:.1%}) — may be hallucinated"

    def _detect_hallucinations(self, answer: str, context: str) -> List[str]:
        """
        Detect specific hallucination patterns in medical answers.
        """
        flags = []
        for pattern in HALLUCINATION_INDICATORS:
            if re.search(pattern, answer, re.IGNORECASE):
                flags.append(f"Detected absolute/risky language: {pattern}")

        # Check if answer introduces drugs not mentioned in context
        answer_drugs = re.findall(r'\b[A-Z][a-z]{3,}\b', answer)
        for drug in answer_drugs:
            if drug not in context and len(drug) > 4:
                # Only flag capitalized drug-like names
                flags.append(f"'{drug}' mentioned in answer but not in retrieved context")

        return flags[:3]  # Cap at 3 flags

    def _apply_corrections(
        self,
        answer: str,
        context: str,
        flags: List[str],
        is_verified: bool,
    ) -> Tuple[str, bool, str]:
        """Apply safety corrections if hallucination detected."""
        if not flags and is_verified:
            return answer, False, ""

        corrected = answer

        # Soften absolute language
        corrected = re.sub(r'\bAlways\b', 'Generally', corrected)
        corrected = re.sub(r'\bNever\b', 'Typically should not', corrected)
        corrected = re.sub(r'\bcures?\b', 'may help with', corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'\b100%\b', 'very effective', corrected)

        was_corrected = corrected != answer
        note = "Softened absolute language for medical safety" if was_corrected else "No corrections needed"

        return corrected, was_corrected, note

    def _apply_safety_formatting(
        self, answer: str, question: str, sources: List[str]
    ) -> str:
        """Format final answer with source citations."""
        formatted = answer.strip()

        if sources:
            source_str = ", ".join(f"📄 {s}" for s in sources[:3])
            formatted += f"\n\n**Sources:** {source_str}"

        return formatted

    def _summarize(self, verified: bool, flags: List[str], corrected: bool) -> str:
        if verified and not flags:
            return "✅ Answer verified and grounded in medical context"
        elif verified and flags:
            return f"⚠️ Answer partially verified — {len(flags)} flag(s) detected"
        elif corrected:
            return "🔧 Answer corrected for medical safety"
        else:
            return "⚠️ Answer could not be fully verified against retrieved context"
