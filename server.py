# server.py — minimal RAG-API (FastAPI) med "no fallback"
from fastapi import FastAPI, Body, HTTPException, Query
import os, json
from typing import List, Dict, Any
import numpy as np
from openai import OpenAI

app = FastAPI()

# ---- Indstillinger ----
EMB_FILE = os.getenv("EMB_FILE", "kb_embeddings.jsonl")      # hvor embeddings ligger
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-small") # OpenAI model
MIN_SCORE_DEFAULT = float(os.getenv("MIN_SCORE", "0.30"))    # laveste cosine-score

# ---- Lager til embeddings ----
_vectors: np.ndarray | None = None
_texts:   List[str] = []
_meta:    List[Dict[str, Any]] = []
_ids:     List[str] = []

def _oa():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY mangler")
    return OpenAI(api_key=key)

def _nice_source(meta: dict) -> str:
    import os as _os
    src = meta.get("source") or meta.get("file") or meta.get("document") or ""
    return _os.path.basename(src) or "Ukendt"

def _nice_page(meta: dict) -> int:
    try:
        return int(meta.get("page", 0)) + 1  # 0 -> 1
    except Exception:
        return 1

def _clip(s: str, n: int = 500) -> str:
    return s[:n] + ("…" if len(s) > n else "")

def _load() -> int:
    """Læs embeddings fra EMB_FILE ind i hukommelsen."""
    global _vectors, _texts, _meta, _ids
    if not os.path.isfile(EMB_FILE):
        _vectors = None
        _texts, _meta, _ids = [], [], []
        return 0
    vecs, ids, texts, meta = [], [], [], []
    with open(EMB_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ids.append(rec.get("id"))
            m = rec.get("metadata", {}) or {}
            meta.append(m)
            texts.append(m.get("text") or rec.get("text") or m.get("chunk") or "")
            v = rec.get("values") or rec.get("embedding")
            if v is None: v = []
            vecs.append(v)
    if not vecs:
        _vectors = None
        _texts, _meta, _ids = [], [], []
        return 0
    arr = np.asarray(vecs, dtype=np.float32)
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)  # normaliser
    _vectors, _texts, _meta, _ids = arr, texts, meta, ids
    return len(_ids)

def _ensure_loaded() -> int:
    return _load() if _vectors is None else len(_ids)

@app.get("/health")
def health(pretty: int = 0):
    exists = os.path.isfile(EMB_FILE)
    size = os.path.getsize(EMB_FILE) if exists else 0
    count = _ensure_loaded()
    return {
        "ok": True,
        "emb_exists": exists,
        "emb_path": os.path.abspath(EMB_FILE),
        "emb_size_bytes": size,
        "records": count,
        "sample_keys": ["id","values","metadata"],
        "sample_text_preview": (_texts[0][:160] if _texts else "")
    }

@app.post("/search")
def search(payload: dict = Body(...)):
    """Returnér KUN matches (kan være tom liste) — ingen fallback-tekst."""
    q = (payload.get("q") or "").strip()
    top_k = int(payload.get("top_k", 5))
    min_score = float(payload.get("min_score", MIN_SCORE_DEFAULT))

    n = _ensure_loaded()
    if n == 0:
        raise HTTPException(400, "No embeddings loaded")

    if not q:
        return {"matches": []}

    emb = _oa().embeddings.create(model=EMB_MODEL, input=q).data[0].embedding
    qv = np.asarray(emb, dtype=np.float32)
    qv = qv / (np.linalg.norm(qv) + 1e-12)

    scores = (_vectors @ qv).astype(np.float32)   # cosine
    idx = np.argsort(-scores)[:top_k]

    results = []
    for i in idx:
        sc = float(scores[i])
        if sc < min_score:
            continue
        m = _meta[i]
        results.append({
            "source": _nice_source(m),      # fx "APA.pdf"
            "page":   _nice_page(m),        # 1-indekseret
            "text":   _clip(_texts[i]),     # kort uddrag
            "score":  sc
        })

    return {"matches": results}  # kan være []