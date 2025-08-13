# app/services/text_preprocess.py
import re
from typing import List, Tuple

TOC_MAX_DIGIT_RATIO = 0.35   # pages with too many digits/punctuation → likely TOC
MIN_CHUNK_CHARS = 380        # avoid tiny, noisy fragments
TITLE_RE = re.compile(r"^(?:\d+(?:\.\d+)*\s+)?([A-ZÁÉÍÓÚÑ][^\n]{3,80})$", re.MULTILINE)

def _digit_ratio(s: str) -> float:
    if not s:
        return 1.0
    digits = sum(ch.isdigit() for ch in s)
    punct  = sum(ch in ".·•…—-–" for ch in s)
    return (digits + punct) / max(1, len(s))

def looks_like_toc_or_cover(text: str, page_idx: int) -> bool:
    """
    Cheap heuristics:
    - very high digit/punctuation ratio (lots of '1.1.2', '......')
    - very short or very long lines with page numbers
    - first 1-2 pages often are cover/index
    """
    if page_idx <= 1 and _digit_ratio(text) > 0.20:
        return True
    if _digit_ratio(text) > TOC_MAX_DIGIT_RATIO:
        return True
    # TOC keywords
    lowered = text.lower()
    if "índice" in lowered or "contenido" in lowered or "contents" in lowered:
        return True
    return False

def remove_headers_footers(pages: List[str]) -> List[str]:
    """
    Remove repeating header/footer lines (simple heuristic: lines that appear in >40% pages).
    """
    from collections import Counter
    # collect candidate lines (top and bottom 2 lines per page)
    candidates = []
    per_page_lines = []
    for p in pages:
        lines = [l.strip() for l in p.splitlines() if l.strip()]
        per_page_lines.append(lines)
        if not lines:
            continue
        head = lines[:2]
        tail = lines[-2:]
        candidates.extend(head + tail)

    counts = Counter(candidates)
    threshold = max(1, int(0.4 * len(pages)))
    repetitive = {line for line, c in counts.items() if c >= threshold}

    cleaned = []
    for lines in per_page_lines:
        filtered = [l for l in lines if l not in repetitive]
        cleaned.append("\n".join(filtered))
    return cleaned

def normalize_spaces(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse big gaps
    return text.strip()

def extract_title(text: str) -> str:
    """
    Try to capture a section heading to anchor the chunk semantically.
    """
    m = TITLE_RE.search(text)
    if m:
        return m.group(1).strip()
    # fallback: first non-empty line up to 80 chars
    for line in text.splitlines():
        line = line.strip()
        if len(line) >= 5:
            return line[:80]
    return ""

def chunk_title_aware(text: str, pdf_name: str, page_start: int, max_chars: int = 1000, overlap: int = 180):
    """
    Fixed-size chunking with overlap, but prepend a detected title/heading to each chunk.
    Returns list of dicts with text + minimal metadata.
    """
    chunks = []
    i = 0
    n = len(text)
    title = extract_title(text)
    while i < n:
        end = min(i + max_chars, n)
        body = text[i:end].strip()
        # prepend title (once per page-chunk) to help retrieval
        chunk_text = (f"{title}\n\n{body}" if title else body).strip()
        if len(chunk_text) >= MIN_CHUNK_CHARS:
            chunks.append({
                "text": chunk_text,
                "meta": {
                    "pdf": pdf_name,
                    "page_start": page_start,
                    "page_end": page_start,  # single-page scope in this simple version
                    "title": title
                }
            })
        i = end - overlap if end - overlap > i else end
    return chunks

def basic_deduplicate(chunks: List[dict]) -> List[dict]:
    """
    Lightweight dedup: hash normalized text without digits to drop near-identical repeats.
    Not perfect, but effective to avoid trivial duplicates.
    """
    seen = set()
    dedup = []
    for ch in chunks:
        text = ch["text"].lower()
        text = re.sub(r"\d+", "", text)
        key = re.sub(r"[\W_]+", " ", text).strip()
        h = hash(key)
        if h not in seen:
            seen.add(h)
            dedup.append(ch)
    return dedup
