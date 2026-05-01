"""
engine/buddhi.py — Buddhi: Core Reasoner using Qwen2.5-7B via vLLM
Antahkarana v16 adapted for MedAssist product.

Buddhi is intellect/discriminative intelligence in Indian philosophy.
It performs structured multi-pass reasoning using Qwen2.5-7B-Instruct.
Exact prompt systems from antahkarana_QWEN_500/antahkarana/system.py.
"""

import os
import re
import logging
import time
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# ── vLLM Engine Singleton ────────────────────────────────────────────────────
_engine = None

MODEL_ID  = os.environ.get("MODEL_ID",  "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN  = os.environ.get("HF_TOKEN",  "")
USE_VLLM  = os.environ.get("USE_VLLM",  "true").lower() == "true"


def _get_engine():
    global _engine
    if _engine is not None:
        return _engine

    if not USE_VLLM:
        logger.info("[BUDDHI] vLLM disabled — using GitHub Models / Groq fallback engine")
        _engine = MockEngine()
        return _engine

    try:
        from vllm import LLM
        from transformers import AutoTokenizer

        logger.info(f"[BUDDHI] Loading {MODEL_ID} via vLLM on L4 GPU...")
        llm = LLM(
            model=MODEL_ID,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.92,
            max_model_len=4096,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=False,
            max_num_batched_tokens=8192,
            max_num_seqs=64,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN or None,
            trust_remote_code=True,
        )
        _engine = VLLMEngine(llm, tokenizer)
        logger.info("[BUDDHI] vLLM engine ready")
    except Exception as e:
        logger.warning(f"[BUDDHI] vLLM load failed ({e}) — falling back to GitHub Models / Groq engine")
        _engine = MockEngine()

    return _engine


class VLLMEngine:
    def __init__(self, llm, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer

    def apply_chat_template(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        from vllm import SamplingParams
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=0.95)
        outputs = self.llm.generate([prompt], sp, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()


class MockEngine:
    """
    Fallback engine — tries GitHub Models (GPT-4o) first, then Groq (LLaMA).
    No keys are cached at import time; always read fresh from env.
    """

    def apply_chat_template(self, system: str, user: str) -> str:
        self._system = system
        self._user   = user
        return f"{system}|||{user}"

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        import requests

        parts  = prompt.split("|||", 1)
        system = parts[0] if len(parts) > 1 else "You are a medical assistant."
        user   = parts[1] if len(parts) > 1 else prompt

        # ── Try GitHub Models (GPT-4o) first ────────────────────────────────
        github_token = os.environ.get("GITHUB_TOKEN", "")
        if github_token:
            try:
                headers = {
                    "Authorization": f"Bearer {github_token}",
                    "Content-Type":  "application/json",
                }
                body = {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    "max_tokens":  max_tokens,
                    "temperature": temperature,
                }
                r = requests.post(
                    "https://models.inference.ai.azure.com/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=30,
                )
                r.raise_for_status()
                result = r.json()["choices"][0]["message"]["content"].strip()
                logger.info("[BUDDHI] GitHub Models (GPT-4o) responded successfully")
                return result
            except Exception as e:
                logger.warning(f"[BUDDHI/GitHub] GPT-4o failed ({e}) — trying Groq")

        # ── Try Groq (LLaMA) as second fallback ─────────────────────────────
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if groq_key:
            try:
                headers = {
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type":  "application/json",
                }
                body = {
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    "max_tokens":  max_tokens,
                    "temperature": temperature,
                }
                r = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=30,
                )
                r.raise_for_status()
                result = r.json()["choices"][0]["message"]["content"].strip()
                logger.info("[BUDDHI] Groq (LLaMA) responded successfully")
                return result
            except Exception as e:
                logger.error(f"[BUDDHI/Groq] API call failed: {e}")
                return f"ANSWER: Error calling Groq API: {e}"

        return (
            "ANSWER: No API key configured. "
            "Set GITHUB_TOKEN or GROQ_API_KEY environment variable."
        )


# ── Prompt Systems ─────────────────────────────────────────────────────────────

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


class Buddhi:
    def __init__(self):
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            self._engine = _get_engine()
        return self._engine

    def reason(self, question: str, context_str: str, q_type: str, medicine_info: Optional[Dict] = None) -> dict:
        t0 = time.time()

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
        prompt      = self.engine.apply_chat_template(system, user_prompt)

        pass1_raw    = self.engine.generate(prompt, max_tokens=1024)
        pass1_answer = self._extract_answer(pass1_raw)
        pass1_steps  = self._extract_reasoning_steps(pass1_raw)

        pass2_raw      = ""
        pass2_answer   = pass1_answer
        pass2_verified = True
        pass2_fired    = False

        if full_context.strip() and not self._is_bad_answer(pass1_answer):
            pramana_user   = (
                f"Question: {question}\n\n"
                f"Draft answer: {pass1_answer}\n\n"
                f"Medical Context:\n{full_context[:2000]}"
            )
            pramana_prompt  = self.engine.apply_chat_template(PRAMANA_SYSTEM, pramana_user)
            pass2_raw       = self.engine.generate(pramana_prompt, max_tokens=512)
            pass2_answer, pass2_verified = self._extract_pramana(pass2_raw, pass1_answer)
            pass2_fired     = True

        pass3_fired  = False
        final_answer = pass2_answer

        if self._is_bad_answer(pass2_answer):
            candidates = []
            for _ in range(3):
                alt = self.engine.generate(prompt, max_tokens=512, temperature=0.7)
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
            "model":           MODEL_ID,
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
        supported_m = re.search(r'Supported\s*:\s*(yes|no)',            raw, re.IGNORECASE)
        revised_m   = re.search(r'Revised answer\s*:\s*(.+?)(?:\n|$)', raw, re.IGNORECASE)
        if revised_m and supported_m and supported_m.group(1).lower() == "no":
            revised = revised_m.group(1).strip()
            if revised and not self._is_bad_answer(revised):
                return revised, False
        return draft, True

    APOLOGY_PATTERNS = [
        "i don't know", "i cannot", "i am not sure", "no information",
        "sorry", "cannot determine", "unclear", "i'm not certain", "not available","i'm not aware", "not aware of any",
    ]

    def _is_bad_answer(self, answer: str) -> bool:
        if not answer or len(answer.strip()) < 2:
            return True
        return any(p in answer.lower() for p in self.APOLOGY_PATTERNS)
