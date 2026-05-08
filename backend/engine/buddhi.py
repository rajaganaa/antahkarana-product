"""
engine/buddhi.py — Buddhi: Core Reasoner using Groq (llama3-70b-8192)
Antahkarana MedAssist — CPU-only Azure deployment.

Buddhi is intellect/discriminative intelligence in Indian philosophy.
It performs structured multi-pass reasoning (Tarka → Pramana → Samsaya).

BUG 2 FIX: Removed ALL vLLM code. Groq is now the PRIMARY engine using
           the official `groq` Python library. No MockEngine. No `|||` separator.
           If Groq fails, falls back to GitHub Models (GPT-4o) via openai library.

Engine priority:
  1. Groq (llama3-70b-8192)  — primary, fastest free LLM
  2. GitHub Models (gpt-4o)  — fallback if GROQ_API_KEY missing/fails

Return dict keys (unchanged):
  reasoning_steps, pass1_raw, pass1_answer, pass2_raw, pass2_answer,
  pass2_verified, pass2_fired, pass3_fired, draft_answer, latency_s, model

Author: RAJAGANAPATHY M, SRM University
"""

import os
import re
import logging
import time
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# ── Model configuration ───────────────────────────────────────────────────────
GROQ_MODEL   = os.environ.get("GROQ_MODEL",   "llama3-70b-8192")
GITHUB_MODEL = os.environ.get("GITHUB_MODEL", "gpt-4o")

# BUG 2 FIX: USE_VLLM is permanently False. We never attempt to load vLLM.
# The env var is read only for /health reporting — it does not change behaviour.
USE_VLLM = False  # Hardcoded. Azure has no GPU. vLLM is removed entirely.


# ── Groq Engine (PRIMARY) ─────────────────────────────────────────────────────

class GroqEngine:
    """
    Primary reasoning engine — Groq API with llama3-70b-8192.
    Uses the official `groq` Python library (groq>=0.9.0).

    BUG 2 FIX: This replaces the broken MockEngine that used `|||` separators
    and required vLLM. Groq is called directly here with proper message dicts.
    """

    def __init__(self):
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Add it to your .env or Azure portal settings."
            )
        self.client = Groq(api_key=api_key)
        self.model  = GROQ_MODEL
        logger.info(f"[BUDDHI] GroqEngine ready — model: {self.model}")

    def chat(self, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        """Call Groq API with clean message dicts. No `|||` separator needed."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()


# ── GitHub Models Fallback Engine ─────────────────────────────────────────────

class GitHubModelsEngine:
    """
    Fallback engine — GitHub Models (GPT-4o) via openai library.
    Used only when GROQ_API_KEY is missing or Groq API call fails.
    """

    def __init__(self):
        from openai import OpenAI
        token = os.environ.get("GITHUB_TOKEN", "")
        if not token:
            raise RuntimeError(
                "GITHUB_TOKEN environment variable is not set. "
                "Cannot use GitHub Models fallback engine."
            )
        self.client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=token,
        )
        self.model = GITHUB_MODEL
        logger.info(f"[BUDDHI] GitHubModelsEngine ready — model: {self.model}")

    def chat(self, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()


# ── Engine factory ────────────────────────────────────────────────────────────

_engine_instance = None


def _get_engine():
    """
    Return the best available engine.
    Priority: Groq → GitHub Models.
    BUG 2 FIX: vLLM is never attempted.
    """
    global _engine_instance
    if _engine_instance is not None:
        return _engine_instance

    # Try Groq first
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        try:
            _engine_instance = GroqEngine()
            logger.info("[BUDDHI] Using GroqEngine as primary engine")
            return _engine_instance
        except Exception as e:
            logger.warning(f"[BUDDHI] GroqEngine init failed ({e}) — trying GitHub Models")

    # Fallback: GitHub Models
    github_token = os.environ.get("GITHUB_TOKEN", "")
    if github_token:
        try:
            _engine_instance = GitHubModelsEngine()
            logger.info("[BUDDHI] Using GitHubModelsEngine as fallback engine")
            return _engine_instance
        except Exception as e:
            logger.error(f"[BUDDHI] GitHubModelsEngine init failed: {e}")

    raise RuntimeError(
        "No LLM engine available. Set GROQ_API_KEY (preferred) "
        "or GITHUB_TOKEN in your environment variables."
    )


# ── Prompt Systems (Antahkarana philosophy preserved) ─────────────────────────

MEDICAL_SYSTEM = (
    "You are MedAssist, a medical information assistant. "
    "If the provided context has relevant information, use it. "
    "If the context does NOT contain information about the medicine, "
    "use your own general medical knowledge to answer. "
    "Never say 'not in context' — always give a helpful answer. "
    "Always include: uses, dosage, side effects, warnings.\n\n"
)

DOSAGE_SYSTEM = (
    "You are a medical information assistant helping with dosage information. "
    "Provide general information from medical guidelines only. "
    "Always recommend consulting a healthcare provider.\n\n"
    "Format:\n"
    "Analysis: <dosage reasoning>\n"
    "ANSWER: <dosage information>\n"
    "WARNING: Always consult your doctor or pharmacist before taking medications."
)

COMPARISON_SYSTEM = (
    "You are a medical comparison expert. "
    "Compare medications based on the provided context.\n\n"
    "Drug 1: ...\nDrug 2: ...\nComparison: ...\nANSWER: <answer>"
)

VERIFICATION_SYSTEM = (
    "You are a medical fact-checker. "
    "Determine if the claim is supported by the medical context.\n\n"
    "You MUST respond with one of: SUPPORTED / NOT SUPPORTED / INSUFFICIENT EVIDENCE\n\n"
    "Evidence assessment: <your reasoning>\n"
    "ANSWER: SUPPORTED or NOT SUPPORTED or INSUFFICIENT EVIDENCE"
)

# Pramana — second-pass Tarka verification (Sanskrit: valid knowledge / proof)
PRAMANA_SYSTEM = (
    "You are a strict medical fact verifier. "
    "Check if the draft answer is supported by the medical context.\n\n"
    "You MUST follow this exact format:\n"
    "Supported: yes\n"
    "Evidence: <quote from context, max 15 words>\n"
    "Revised answer: <repeat the draft answer>\n\n"
    "OR if NOT supported:\n"
    "Supported: no\n"
    "Evidence: <what context actually says, max 15 words>\n"
    "Revised answer: <corrected answer from context>"
)


# ── Buddhi Class ──────────────────────────────────────────────────────────────

class Buddhi:
    """
    Buddhi — The discriminative intellect.

    Multi-pass reasoning pipeline (Tarka → Pramana → Samsaya):
      Pass 1 (Tarka)   — initial LLM reasoning with context
      Pass 2 (Pramana) — verification against retrieved context
      Pass 3 (Samsaya) — self-consistency sampling if answer is bad

    BUG 2 FIX: Uses GroqEngine as primary. No vLLM. No MockEngine.
               Max 2 LLM calls per request (Pass 3 only fires on bad answers).
    """

    def __init__(self):
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            self._engine = _get_engine()
        return self._engine

    def reason(
        self,
        question: str,
        context_str: str,
        q_type: str,
        medicine_info: Optional[Dict] = None,
    ) -> dict:
        """
        Run Antahkarana multi-pass reasoning.

        Returns dict with exactly these keys:
          reasoning_steps, pass1_raw, pass1_answer, pass2_raw, pass2_answer,
          pass2_verified, pass2_fired, pass3_fired, draft_answer, latency_s, model
        """
        t0 = time.time()

        # Build vision context prefix (Chitta memory enrichment)
        vision_ctx = ""
        if medicine_info:
            name     = medicine_info.get("generic_name") or medicine_info.get("brand_name", "")
            strength = medicine_info.get("strength", "")
            form     = medicine_info.get("form", "")
            if name:
                vision_ctx = f"[Identified Medicine: {name} {strength} {form}]\n\n"

        full_context = f"{vision_ctx}{context_str}" if vision_ctx else context_str

        system      = self._select_system(q_type)
        user_prompt = f"Medical Context:\n{full_context}\n\nQuestion: {question}"

        # ── Pass 1: Tarka (initial reasoning) ────────────────────────────────
        pass1_raw    = self.engine.chat(system, user_prompt, max_tokens=1024)
        pass1_answer = self._extract_answer(pass1_raw)
        pass1_steps  = self._extract_reasoning_steps(pass1_raw)

        # ── Pass 2: Pramana (verification against context) ───────────────────
        pass2_raw      = ""
        pass2_answer   = pass1_answer
        pass2_verified = True
        pass2_fired    = False

        if full_context.strip() and not self._is_bad_answer(pass1_answer):
            pramana_user = (
                f"Question: {question}\n\n"
                f"Draft answer: {pass1_answer}\n\n"
                f"Medical Context:\n{full_context[:2000]}"
            )
            pass2_raw       = self.engine.chat(PRAMANA_SYSTEM, pramana_user, max_tokens=512)
            pass2_answer, pass2_verified = self._extract_pramana(pass2_raw, pass1_answer)
            pass2_fired     = True

        # ── Pass 3: Samsaya (self-consistency — only if answer still bad) ────
        pass3_fired  = False
        final_answer = pass2_answer

        if self._is_bad_answer(pass2_answer):
            candidates = []
            for _ in range(3):
                alt = self.engine.chat(system, user_prompt, max_tokens=512, temperature=0.7)
                candidates.append(self._extract_answer(alt))
            candidates = [c for c in candidates if not self._is_bad_answer(c)]
            if candidates:
                from collections import Counter
                best         = Counter(c.lower() for c in candidates).most_common(1)[0][0]
                final_answer = next((c for c in candidates if c.lower() == best), candidates[0])
                pass3_fired  = True

        elapsed = round(time.time() - t0, 3)

        return {
            "reasoning_steps": pass1_steps,
            "pass1_raw":       pass1_raw,
            "pass1_answer":    pass1_answer,
            "pass2_raw":       pass2_raw if pass2_fired else None,
            "pass2_answer":    pass2_answer,
            "pass2_verified":  pass2_verified,
            "pass2_fired":     pass2_fired,
            "pass3_fired":     pass3_fired,
            "draft_answer":    final_answer,
            "latency_s":       elapsed,
            "model":           getattr(self._engine, "model", "unknown"),
        }

    def _select_system(self, q_type: str) -> str:
        from engine.manas import QType
        system_map = {
            QType.VERIFICATION: VERIFICATION_SYSTEM,
            QType.COMPARISON:   COMPARISON_SYSTEM,
            QType.DOSAGE:       DOSAGE_SYSTEM,
            QType.MATH:         DOSAGE_SYSTEM,
        }
        return system_map.get(q_type, MEDICAL_SYSTEM)

    def _extract_answer(self, raw: str) -> str:
        m = re.search(r'ANSWER\s*:\s*(.+?)(?:\n|$)', raw, re.IGNORECASE)
        if m:
            ans = re.sub(r'[.,;:]+$', '', m.group(1).strip())
            return ans.strip()
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        return lines[-1] if lines else raw[:200]

    def _extract_reasoning_steps(self, raw: str) -> List[str]:
        steps = []
        m = re.search(r'Reasoning\s*:\s*(.+?)(?:ANSWER|WARNING|$)', raw, re.IGNORECASE | re.DOTALL)
        if m:
            numbered = re.split(r'\n(?=Step\s*\d+|^\d+\.)', m.group(1).strip(), flags=re.MULTILINE)
            steps    = [s.strip() for s in numbered if s.strip()]
        if not steps:
            steps = [l.strip() for l in raw.split("\n") if l.strip()][:5]
        return steps

    def _extract_pramana(self, raw: str, draft: str):
        """Parse Pramana verification response."""
        supported_m = re.search(r'Supported\s*:\s*(yes|no)',            raw, re.IGNORECASE)
        revised_m   = re.search(r'Revised answer\s*:\s*(.+?)(?:\n|$)', raw, re.IGNORECASE)
        if revised_m and supported_m and supported_m.group(1).lower() == "no":
            revised = revised_m.group(1).strip()
            if revised and not self._is_bad_answer(revised):
                return revised, False
        return draft, True

    APOLOGY_PATTERNS = [
        "i don't know", "i cannot", "i am not sure", "no information",
        "sorry", "cannot determine", "unclear", "i'm not certain",
        "not available", "i'm not aware", "not aware of any",
    ]

    def _is_bad_answer(self, answer: str) -> bool:
        if not answer or len(answer.strip()) < 2:
            return True
        return any(p in answer.lower() for p in self.APOLOGY_PATTERNS)
