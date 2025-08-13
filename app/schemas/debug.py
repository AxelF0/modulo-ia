# app/schemas/debug.py
from pydantic import BaseModel
from typing import List, Optional

class ChunkDebug(BaseModel):
    score: float
    passed_threshold: bool
    excerpt: str
    pdf: Optional[str] = None
    page_start: Optional[int] = None
    title: Optional[str] = None

class DebugSearchRequest(BaseModel):
    query: str
    top_k: int = 4
    include_prompt_preview: bool = True

class DebugSearchResponse(BaseModel):
    query: str
    top_k: int
    min_sim_threshold: float
    results: List[ChunkDebug]
    prompt_preview: Optional[str] = None

