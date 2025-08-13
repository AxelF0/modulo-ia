# app/api/debug.py

from fastapi import APIRouter
import os
from typing import List
from app.schemas.debug import DebugSearchRequest, DebugSearchResponse, ChunkDebug
from app.services.ia_service import get_relevant_chunks
from app.services import ia_service

router = APIRouter(prefix="/debug", tags=["debug"])
MIN_SIM_THRESHOLD = float(os.getenv("MIN_SIM_THRESHOLD", "0.32"))

@router.post("/search", response_model=DebugSearchResponse)
def debug_search(payload: DebugSearchRequest) -> DebugSearchResponse:
    # Now get_relevant_chunks returns (text, score, meta)
    chunks = get_relevant_chunks(payload.query, top_k=payload.top_k)

    results: List[ChunkDebug] = []
    if chunks:
        for text, score, meta in chunks:
            excerpt = text if len(text) <= 400 else text[:400] + "…"
            results.append(ChunkDebug(
                score=round(score, 4),
                passed_threshold=score >= MIN_SIM_THRESHOLD,
                excerpt=excerpt,
                pdf=(meta or {}).get("pdf"),
                page_start=(meta or {}).get("page_start"),
                title=(meta or {}).get("title"),
            ))

    prompt_preview = None
    if payload.include_prompt_preview and chunks:
        context_str = "\n\n".join([
            f"- [{(m or {}).get('pdf','?')} p.{(m or {}).get('page_start','?')}] "
            f"{t[:260]}{'…' if len(t) > 260 else ''}"
            for t, _, m in chunks
        ])
        prompt_preview = (
            f"{ia_service.SYSTEM_INSTRUCTION}\n\n"
            f"Contexto (fragmentos relevantes):\n{context_str}\n\n"
            f"Pregunta: {payload.query}\nRespuesta:"
        )

    return DebugSearchResponse(
        query=payload.query,
        top_k=payload.top_k,
        min_sim_threshold=MIN_SIM_THRESHOLD,
        results=results,
        prompt_preview=prompt_preview
    )
