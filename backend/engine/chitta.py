"""
engine/chitta.py — Chitta: Dense Retrieval + ChromaDB Search
Antahkarana v16 adapted for MedAssist product.

Chitta is memory/consciousness in Indian philosophy — it stores and retrieves.
Combines sentence-transformer dense retrieval with ChromaDB vector search.
Exact dense scoring logic from antahkarana_QWEN_500/antahkarana/system.py.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# ── Embedder singleton (same model as Antahkarana NLP research) ──────────────
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("[CHITTA] Dense embedder loaded: all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"[CHITTA] Embedder load failed ({e}), falling back to lexical")
            _embedder = "lexical"
    return _embedder


# ── ChromaDB singleton ────────────────────────────────────────────────────────
_chroma_collection = None

def _get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    chroma_path = os.environ.get("CHROMA_PATH", "./data/chroma_db")
    collection_name = os.environ.get("CHROMA_COLLECTION", "medassist_drugs")

    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(path=chroma_path)
        try:
            _chroma_collection = client.get_collection(collection_name)
            logger.info(f"[CHITTA] ChromaDB loaded: {collection_name} ({_chroma_collection.count()} docs)")
        except Exception:
            _chroma_collection = None
            logger.warning("[CHITTA] ChromaDB collection not found — will use text-only retrieval")
    except ImportError:
        logger.warning("[CHITTA] chromadb not installed")
        _chroma_collection = None

    return _chroma_collection


class Chitta:
    """
    Chitta — Dense retrieval with ChromaDB integration.
    Logic from antahkarana_QWEN_500 + MedAssist RAG (session1/session2).
    """

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def search_chroma(
        self, question: str, entities: List[str], k: int = 5
    ) -> List[Dict]:
        """
        Search ChromaDB vector store.
        Returns list of {content, source, score} dicts.
        """
        collection = _get_chroma_collection()
        if collection is None:
            return []

        try:
            # Build enriched query (question + key entities)
            query = question
            if entities:
                query = f"{question} {' '.join(entities[:3])}"

            results = collection.query(
                query_texts=[query],
                n_results=min(k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            chunks = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]

            for doc, meta, dist in zip(docs, metas, dists):
                # ChromaDB returns L2 distance; convert to similarity score
                score = max(0.0, 1.0 - (dist / 2.0))
                source = os.path.basename(meta.get("source", "unknown.pdf"))
                chunks.append({
                    "content": doc,
                    "source": source,
                    "score": round(score, 4),
                })

            logger.info(f"[CHITTA] ChromaDB returned {len(chunks)} chunks for: {question[:50]}")
            return chunks

        except Exception as e:
            logger.error(f"[CHITTA] ChromaDB search error: {e}")
            return []

    def score_passages(
        self, question: str, passages: List[Dict], entities: List[str]
    ) -> List[Tuple[float, Dict]]:
        """
        Dense semantic scoring with lexical fallback.
        Exact logic from antahkarana_QWEN_500/antahkarana/system.py Chitta class.
        """
        embedder = _get_embedder()
        scored = []

        if embedder != "lexical":
            try:
                para_texts = [p.get("content", p.get("text", "")) for p in passages]
                if para_texts:
                    q_emb = embedder.encode(question, convert_to_numpy=True, show_progress_bar=False)
                    p_embs = embedder.encode(para_texts, convert_to_numpy=True, show_progress_bar=False)
                    for i, (para, text) in enumerate(zip(passages, para_texts)):
                        sem_score = self._cosine(q_emb, p_embs[i])
                        ent_bonus = 0.1 * sum(1 for e in entities if e.lower() in text.lower())
                        scored.append((sem_score + ent_bonus, para))
                    return sorted(scored, key=lambda x: x[0], reverse=True)
            except Exception as e:
                logger.debug(f"[CHITTA] Dense scoring failed ({e}), using lexical")

        # Lexical fallback
        q_words = set(question.lower().split())
        for para in passages:
            text = para.get("content", para.get("text", ""))
            words = set(text.lower().split())
            overlap = len(q_words & words) / (len(q_words) + 1)
            ent_bonus = 0.1 * sum(1 for e in entities if e.lower() in text.lower())
            scored.append((overlap + ent_bonus, para))
        return sorted(scored, key=lambda x: x[0], reverse=True)

    def retrieve(
        self, question: str, entities: List[str], k: int = 5
    ) -> dict:
        """
        Main retrieval method. Returns full Chitta trace for API.
        """
        # ChromaDB search
        chroma_chunks = self.search_chroma(question, entities, k=k)

        # Re-rank with dense scoring
        if chroma_chunks:
            scored = self.score_passages(question, chroma_chunks, entities)
            top_chunks = [chunk for _, chunk in scored[:k]]
        else:
            top_chunks = []

        # Build context string for Buddhi
        context_parts = [c["content"] for c in top_chunks]
        context_str = "\n\n---\n\n".join(context_parts)

        sources = list(dict.fromkeys(c.get("source", "unknown") for c in top_chunks))

        return {
            "retrieved_chunks": top_chunks,
            "context_str": context_str,
            "sources": sources,
            "num_chunks": len(top_chunks),
            "retrieval_method": "dense+chromadb" if chroma_chunks else "fallback",
        }
