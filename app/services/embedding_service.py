# app/services/embedding_service.py

import os
import faiss
import fitz
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

from app.services.text_preprocess import (
    looks_like_toc_or_cover,
    remove_headers_footers,
    normalize_spaces,
    chunk_title_aware,
    basic_deduplicate,
)

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "distiluse-base-multilingual-cased-v1")
INDEX_FILE = os.getenv("VECTOR_DB_INDEX", "data/vector_db/index.faiss")
DOC_FILE = os.getenv("VECTOR_DB_DOCS", "data/vector_db/docs.pkl")

MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms

def _load_or_create_ip_index(dim: int) -> faiss.IndexFlatIP:
    if os.path.exists(INDEX_FILE):
        idx = faiss.read_index(INDEX_FILE)
        if not isinstance(idx, faiss.IndexFlatIP):
            raise ValueError("Existing FAISS index is not IndexFlatIP.")
        if idx.d != dim:
            raise ValueError(f"FAISS dim mismatch. Expected {dim}, got {idx.d}.")
        return idx
    return faiss.IndexFlatIP(dim)

def _read_pdf_pages(pdf_path: str) -> List[str]:
    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pages.append(page.get_text())
    return pages

def build_vector_index(pdf_path: str, max_chars: int, overlap: int):
    pdf_name = os.path.basename(pdf_path)
    raw_pages = _read_pdf_pages(pdf_path)

    # 1) Remove headers/footers
    cleaned_pages = remove_headers_footers([normalize_spaces(p) for p in raw_pages])

    # 2) Filter TOC/cover-like pages
    useful_pages = []
    for i, p in enumerate(cleaned_pages):
        if not looks_like_toc_or_cover(p, i):
            useful_pages.append((i, p))

    # 3) Chunk per page, title-aware
    chunk_objs: List[dict] = []
    for page_idx, page_text in useful_pages:
        chunks = chunk_title_aware(page_text, pdf_name, page_start=page_idx, max_chars=max_chars, overlap=overlap)
        chunk_objs.extend(chunks)

    # 4) Deduplicate
    chunk_objs = basic_deduplicate(chunk_objs)
    if not chunk_objs:
        print(f"⚠️ No useful chunks found in {pdf_name}. Skipping.")
        return

    # 5) Encode + normalize
    texts = [c["text"] for c in chunk_objs]
    emb = MODEL.encode(texts, convert_to_numpy=True)
    emb = _normalize(emb)
    dim = emb.shape[1]

    # 6) Index append
    index = _load_or_create_ip_index(dim)
    index.add(emb)

    # 7) Persist index & docs (now docs is a list[dict] with text + meta)
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    faiss.write_index(index, INDEX_FILE)

    if os.path.exists(DOC_FILE):
        with open(DOC_FILE, "rb") as f:
            existing = pickle.load(f)
        if not isinstance(existing, list):
            existing = []
    else:
        existing = []

    existing.extend(chunk_objs)

    with open(DOC_FILE, "wb") as f:
        pickle.dump(existing, f)

    print(f"✅ Indexed {len(chunk_objs)} chunks from {pdf_name}. Total chunks: {len(existing)}")
