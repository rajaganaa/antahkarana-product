"""
frontend/streamlit_app.py — Antahkarana Local Demo
Run: streamlit run streamlit_app.py

Author: RAJAGANAPATHY M, SRM University
"""

import os
import sys
import json
import time
import requests
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

st.set_page_config(
    page_title="Antahkarana MedAssist",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #080c18; color: #e2e8f0; }
#MainMenu, footer, header { visibility: hidden; }

.ant-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    color: #00d4aa;
    margin-bottom: 0.25rem;
}
.ant-sub { color: #64748b; font-size: 0.85rem; margin-bottom: 2rem; }

.trace-card {
    background: #0f1824;
    border: 1px solid #1e2d3d;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
}
.card-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.final-box {
    background: linear-gradient(135deg, rgba(0,212,170,0.08), rgba(139,92,246,0.05));
    border: 1px solid rgba(0,212,170,0.3);
    border-radius: 12px;
    padding: 1.2rem;
    font-size: 0.95rem;
    line-height: 1.7;
    white-space: pre-wrap;
}
.disclaimer {
    background: rgba(245,158,11,0.06);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    color: #f59e0b;
    font-size: 0.78rem;
    margin-top: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="ant-title">🧠 ANTAHKARANA</div>', unsafe_allow_html=True)
st.markdown('<div class="ant-sub">5-Component Medical Reasoning Engine · RAJAGANAPATHY M, SRM University · Qwen2.5-7B + BLIP-2 + ChromaDB</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_url = st.text_input(
        "Backend API URL",
        value=os.environ.get("API_URL", "http://localhost:8000"),
        help="URL of the running FastAPI backend"
    )
    st.markdown("---")
    st.markdown("### 🏗️ Architecture")
    st.markdown("""
    **Step 1** 👁 Vision — BLIP-2 Flan-T5-XL  
    **Step 2** 🧭 Manas — Question Router  
    **Step 3** 📚 Chitta — ChromaDB Retrieval  
    **Step 4** 🧠 Buddhi — Qwen2.5-7B Reasoner  
    **Step 5** ⚖️ Ahamkara — Confidence Scorer  
    **Step 6** 👁‍🗨 Sakshi — Hallucination Verifier  
    """)
    st.markdown("---")
    st.markdown("### 📊 Research Results")
    st.metric("HotpotQA F1", "71.3%", "+5.2% vs CoT")
    st.metric("MMLU EM", "68.4%", "+3.1% vs CoT")
    st.metric("VLM Efficiency", "70.1%", "fewer model calls")

# ── Main Input ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1.5])

with col1:
    st.markdown("#### 💊 Medicine Image (Optional)")
    uploaded_image = st.file_uploader(
        "Upload medicine package image",
        type=["jpg", "jpeg", "png", "webp"],
        help="BLIP-2 will extract drug name, strength, expiry date"
    )
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded medicine image", use_column_width=True)

    st.markdown("#### ❓ Your Question")
    example_q = st.selectbox(
        "Example questions",
        [
            "Custom question...",
            "What is the maximum dose of paracetamol for adults?",
            "Calculate dosage for a 25kg child — amoxicillin",
            "What are the side effects of ibuprofen?",
            "Is metformin safe for kidney disease patients?",
            "What are FDA-reported adverse reactions for cetirizine?",
            "Can diabetic patients take Metformin with alcohol?",
        ]
    )
    question = st.text_area(
        "Question",
        value="" if example_q == "Custom question..." else example_q,
        height=120,
        placeholder="Ask anything about medications..."
    )

    run_btn = st.button("⚡ Run Antahkarana Pipeline", type="primary", use_container_width=True)

# ── Pipeline Execution ────────────────────────────────────────────────────────
with col2:
    if run_btn:
        if not question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Running Antahkarana 5-component pipeline..."):
                try:
                    form_data = {"question": question}
                    files = {}
                    if uploaded_image:
                        files["image"] = (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)

                    resp = requests.post(
                        f"{api_url}/api/reason",
                        data=form_data,
                        files=files if files else None,
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    # ── Step 1: Vision ────────────────────────────────────────
                    if data.get("vision"):
                        v = data["vision"]
                        with st.expander("👁 Step 1: Vision — BLIP-2 Analysis", expanded=bool(uploaded_image)):
                            if v.get("error"):
                                st.error(v["error"])
                            else:
                                cols = st.columns(3)
                                cols[0].metric("Drug Name", v.get("generic_name") or v.get("brand_name", "—"))
                                cols[1].metric("Strength", v.get("strength", "—"))
                                cols[2].metric("Expiry", v.get("expiry_date", "—"))
                                if v.get("manufacturer") and v["manufacturer"] != "Not detected":
                                    st.caption(f"Manufacturer: {v['manufacturer']} | Method: {v.get('extraction_method', 'BLIP-2')}")

                    # ── Step 2: Manas ─────────────────────────────────────────
                    if data.get("manas"):
                        m = data["manas"]
                        with st.expander(f"🧭 Step 2: Manas — Routed to [{m.get('question_type','').upper()}]", expanded=True):
                            c1, c2 = st.columns(2)
                            c1.metric("Question Type", m.get("question_type", "—").upper())
                            c2.metric("Confidence", f"{int(m.get('confidence', 0) * 100)}%")
                            st.caption(f"Rationale: {m.get('routing_rationale', '')}")
                            if m.get("entities"):
                                st.markdown("**Entities:** " + " · ".join(f"`{e}`" for e in m["entities"]))

                    # ── Step 3: Chitta ────────────────────────────────────────
                    if data.get("chitta"):
                        ch = data["chitta"]
                        with st.expander(f"📚 Step 3: Chitta — {ch.get('num_chunks', 0)} chunks retrieved"):
                            for i, chunk in enumerate(ch.get("retrieved_chunks", [])[:3]):
                                st.markdown(f"**Chunk {i+1}** — `{chunk.get('source', '?')}` — {int(chunk.get('score', 0)*100)}% match")
                                st.caption(chunk.get("content", "")[:300] + "…")
                                st.divider()

                    # ── Step 4: Buddhi ────────────────────────────────────────
                    if data.get("buddhi"):
                        b = data["buddhi"]
                        pass_label = "Pass3" if b.get("pass3_fired") else "Pass2" if b.get("pass2_fired") else "Pass1"
                        with st.expander(f"🧠 Step 4: Buddhi — {pass_label} · {b.get('latency_s', 0)}s", expanded=True):
                            steps = b.get("reasoning_steps", [])
                            if steps:
                                for i, step in enumerate(steps[:6]):
                                    st.markdown(f"**{i+1}.** {step}")
                            st.markdown(f"**Draft Answer:** {b.get('draft_answer', '—')}")
                            flags = []
                            if b.get("pass2_fired"): flags.append(f"Pramana: {'✅ verified' if b.get('pass2_verified') else '🔧 corrected'}")
                            if b.get("pass3_fired"): flags.append("Samsaya: ⚡ self-consistency")
                            if flags: st.caption(" · ".join(flags))

                    # ── Step 5: Ahamkara ──────────────────────────────────────
                    if data.get("ahamkara"):
                        a = data["ahamkara"]
                        score_pct = int(a.get("confidence_score", 0) * 100)
                        with st.expander(f"⚖️ Step 5: Ahamkara — Confidence {score_pct}% [{a.get('confidence_label','')}]", expanded=True):
                            st.progress(score_pct / 100)
                            cols = st.columns(3)
                            cols[0].metric("Score", f"{score_pct}%")
                            cols[1].metric("Label", a.get("confidence_label", "—"))
                            cols[2].metric("Pass Level", a.get("pass_level", "—").split(" ")[0])
                            if a.get("needs_retry"):
                                st.warning("Low confidence — retry recommended")

                    # ── Step 6: Sakshi ────────────────────────────────────────
                    if data.get("sakshi"):
                        s = data["sakshi"]
                        icon = "✅" if s.get("verified") else "⚠️"
                        with st.expander(f"👁‍🗨 Step 6: Sakshi — {icon} {s.get('sakshi_summary', '')}", expanded=True):
                            if s.get("hallucination_flags"):
                                for flag in s["hallucination_flags"]:
                                    st.warning(f"⚠️ {flag}")
                            if s.get("corrected"):
                                st.info(f"🔧 Correction: {s.get('correction_note', '')}")

                    # ── Final Answer ──────────────────────────────────────────
                    st.markdown("---")
                    st.markdown("### ✅ Final Answer")
                    final = data.get("final_answer") or data.get("sakshi", {}).get("final_answer", "No answer generated.")
                    # Strip source line from final answer display
                    clean_final = final.split("\n\n**Sources:**")[0].strip() if "**Sources:**" in final else final
                    st.markdown(f'<div class="final-box">{clean_final}</div>', unsafe_allow_html=True)

                    if data.get("sources"):
                        st.caption("📄 Sources: " + " · ".join(data["sources"]))

                    st.markdown(f'<div class="disclaimer">⚠️ {data.get("sakshi", {}).get("medical_disclaimer", "Always consult a qualified healthcare professional.")}</div>', unsafe_allow_html=True)

                    st.caption(f"⏱ Total pipeline latency: {data.get('total_latency_s', '?')}s · Request ID: {data.get('request_id', '?')}")

                except requests.ConnectionError:
                    st.error(f"Cannot connect to backend at {api_url}. Make sure it's running:\n\n```\ncd backend && python main.py\n```")
                except requests.HTTPError as e:
                    st.error(f"API error: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.markdown("### 🔮 Reasoning Trace")
        st.info("Enter a question and click **Run Antahkarana Pipeline** to see the full 7-step reasoning trace.")
        st.markdown("""
        **Pipeline steps:**
        1. 👁 **Vision** — BLIP-2 Flan-T5-XL medicine image analysis
        2. 🧭 **Manas** — Question routing (simple/math/dosage/expiry/FDA/comparison)
        3. 📚 **Chitta** — Dense retrieval from ChromaDB (5 medical PDFs)
        4. 🧠 **Buddhi** — Qwen2.5-7B structured reasoning (3-pass)
        5. ⚖️ **Ahamkara** — Confidence scoring + retry decision
        6. 👁‍🗨 **Sakshi** — Hallucination detection + correction
        7. ✅ **Answer** — Verified, grounded, sourced response
        """)
