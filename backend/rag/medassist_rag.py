"""
rag/medassist_rag.py — MedAssist ChromaDB RAG Pipeline
From session1_medassist_rag_workshop + session2_medassist_agent.

Builds and queries ChromaDB vector store from 5 medical PDFs.
Uses HuggingFace all-MiniLM-L6-v2 embeddings (free, no API key needed).
"""

import os
import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("MEDASSIST_DATA_DIR", "./data/drug_guides")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./data/chroma_db")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "medassist_drugs")

_vector_store = None
_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        logger.info("[RAG] Loading all-MiniLM-L6-v2 embeddings...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        logger.info("[RAG] Embeddings loaded")
    except ImportError:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("[RAG] Using sentence-transformers directly")
            _embeddings = _STEmbeddings()
        except Exception as e:
            logger.error(f"[RAG] Embedding load failed: {e}")
            _embeddings = None

    return _embeddings


class _STEmbeddings:
    """Minimal LangChain-compatible wrapper for sentence-transformers."""
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


def get_vector_store():
    """Load or build ChromaDB vector store."""
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    embeddings = _get_embeddings()
    if embeddings is None:
        return None

    try:
        from langchain_community.vectorstores import Chroma

        if os.path.exists(CHROMA_PATH):
            logger.info(f"[RAG] Loading existing ChromaDB from {CHROMA_PATH}")
            _vector_store = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
            )
            count = _vector_store._collection.count()
            logger.info(f"[RAG] ChromaDB loaded: {count} documents")
        else:
            logger.info("[RAG] Building ChromaDB from PDFs...")
            _vector_store = _build_vector_store(embeddings)

        return _vector_store

    except Exception as e:
        logger.error(f"[RAG] Vector store error: {e}")
        return None


def _build_vector_store(embeddings):
    """Build ChromaDB from PDF documents."""
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma

    if not os.path.exists(DATA_DIR):
        logger.error(f"[RAG] Data directory not found: {DATA_DIR}")
        return None

    # Load PDFs
    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        logger.warning("[RAG] No PDF documents found")
        return None

    logger.info(f"[RAG] Loaded {len(documents)} pages from {DATA_DIR}")

    # Split into chunks (from session1 rag_configurable.py)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"[RAG] Created {len(chunks)} chunks")

    # Create vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )
    logger.info(f"[RAG] ChromaDB built with {len(chunks)} chunks at {CHROMA_PATH}")
    return vector_store


def search_drug_database(query: str, k: int = 5) -> List[Dict]:
    """
    Search ChromaDB for relevant medical information.
    Adapted from session2 tools/rag_search.py search_drug_database tool.
    """
    vector_store = get_vector_store()
    if vector_store is None:
        return []

    try:
        results = vector_store.similarity_search_with_score(query, k=k)
        chunks = []
        for doc, score in results:
            source = os.path.basename(doc.metadata.get("source", "unknown.pdf"))
            # Convert distance to similarity (ChromaDB L2 distance)
            similarity = max(0.0, 1.0 - (score / 2.0))
            chunks.append({
                "content": doc.page_content,
                "source": source,
                "score": round(similarity, 4),
                "page": doc.metadata.get("page", 0),
            })

        logger.info(f"[RAG] {len(chunks)} chunks retrieved for: {query[:50]}")
        return chunks

    except Exception as e:
        logger.error(f"[RAG] Search error: {e}")
        return []


def build_index_if_needed():
    """Ensure ChromaDB index is built. Call at startup."""
    vs = get_vector_store()
    if vs is not None:
        logger.info("[RAG] Index ready")
        return True
    logger.warning("[RAG] Index could not be built")
    return False
