"""
engine/manas.py — Manas: The Question Router
Antahkarana v16 (Qwen2.5-7B) adapted for MedAssist product.

Manas classifies the question type and extracts named entities.
In Indian philosophy, Manas is the sensory mind — it perceives and classifies.
"""

import re
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


class QType:
    SIMPLE       = "simple"
    MULTIHOP     = "multihop"
    COMPARISON   = "comparison"
    MATH         = "math"
    VERIFICATION = "verification"
    MCHOICE      = "mchoice"
    MEDICAL      = "medical"     # Med-specific extension
    DOSAGE       = "dosage"      # Dosage calculation trigger
    EXPIRY       = "expiry"      # Expiry check trigger
    FDA          = "fda"         # FDA adverse events trigger


# Keyword sets — exact from Antahkarana NLP research
MATH_KW = {
    "how many", "how much", "calculate", "total", "sum", "average",
    "percent", "ratio", "difference", "multiply", "divide", "dose",
    "dosage", "mg", "milligram", "weight", "kg", "kilogram",
}
COMPARE_KW = {
    "older", "younger", "longer", "shorter", "more", "less",
    "earlier", "later", "higher", "lower", "bigger", "smaller",
    "first", "last", "both", "same", "versus", "vs", "better", "worse",
}
VERIFY_KW = {
    "supports", "refutes", "true or false", "is it true", "verify",
    "claim", "expired", "expiry", "valid", "still good",
}
DOSAGE_KW = {
    "dose", "dosage", "how much", "how many mg", "calculate dose",
    "dose for", "safe dose", "maximum dose", "daily dose",
}
EXPIRY_KW = {
    "expire", "expiry", "expired", "expiration", "use by", "best before",
    "still good", "still valid", "can i use", "safe to use",
}
FDA_KW = {
    "side effect", "adverse", "reaction", "warning", "reported",
    "fda", "contraindication", "interact", "interaction",
}
MEDICAL_KW = {
    "drug", "medicine", "medication", "tablet", "capsule", "syrup",
    "treatment", "therapy", "cure", "prescription", "generic", "brand",
}


class Manas:
    """
    Manas — The question router.
    Classifies queries into types and extracts named entities.
    Exact logic from antahkarana_QWEN_500/antahkarana/system.py,
    extended with medical-domain categories for MedAssist.
    """

    def classify(self, question: str, dataset: str = "medassist") -> Tuple[str, float]:
        """
        Classify question type. Returns (q_type, routing_confidence).
        For medical product, we add dosage/expiry/fda routing on top of base logic.
        """
        q = question.lower()

        # Medical tool triggers (highest priority for this product)
        if any(k in q for k in DOSAGE_KW) and any(k in q for k in ["calculate", "for a", "for my", "kg", "weight", "age"]):
            logger.info(f"[MANAS] Routed to DOSAGE: {question[:60]}")
            return QType.DOSAGE, 0.92

        if any(k in q for k in EXPIRY_KW):
            logger.info(f"[MANAS] Routed to EXPIRY: {question[:60]}")
            return QType.EXPIRY, 0.90

        if any(k in q for k in FDA_KW):
            logger.info(f"[MANAS] Routed to FDA: {question[:60]}")
            return QType.FDA, 0.88

        # Base Antahkarana routing
        if any(k in q for k in VERIFY_KW):
            return QType.VERIFICATION, 0.85

        if any(k in q for k in MATH_KW):
            return QType.MATH, 0.82

        if any(k in q for k in COMPARE_KW):
            return QType.COMPARISON, 0.80

        if any(k in q for k in MEDICAL_KW):
            return QType.MEDICAL, 0.75

        logger.info(f"[MANAS] Routed to SIMPLE: {question[:60]}")
        return QType.SIMPLE, 0.70

    def extract_entities(self, question: str) -> List[str]:
        STOP_WORDS = {"What", "Why", "How", "When", "Where", "Which", 
                      "Is", "Are", "Can", "Does", "Do", "Will", "Should"}
        
        title_case = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*', question)
        upper_case = re.findall(r'\b[A-Z]{4,}\b', question)
        doses = re.findall(r'\d+\s*(?:mg|ml|g|mcg|iu)\b', question, re.IGNORECASE)
        
        entities = list(dict.fromkeys(title_case + upper_case + doses))
        entities = [e for e in entities if e not in STOP_WORDS]  # Filter AFTER building
        
        logger.debug(f"[MANAS] Entities: {entities}")
        return entities

    def get_routing_info(self, question: str) -> dict:
        """Full routing trace for API response."""
        q_type, confidence = self.classify(question)
        entities = self.extract_entities(question)
        return {
            "question_type": q_type,
            "confidence": round(confidence, 3),
            "entities": entities,
            "routing_rationale": self._get_rationale(q_type, question),
        }

    def _get_rationale(self, q_type: str, question: str) -> str:
        rationale_map = {
            QType.DOSAGE: "Contains dosage calculation keywords — routing to calculator tool",
            QType.EXPIRY: "Contains expiry-check keywords — routing to date checker",
            QType.FDA: "Contains adverse event keywords — routing to FDA API",
            QType.VERIFICATION: "Boolean/verification question detected",
            QType.MATH: "Numerical computation required",
            QType.COMPARISON: "Comparative reasoning required",
            QType.MEDICAL: "General medical query — routing to RAG + Buddhi",
            QType.SIMPLE: "Simple factual query — routing to RAG + Buddhi",
        }
        return rationale_map.get(q_type, "Default routing")
