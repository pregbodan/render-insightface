import base64
import hashlib
import io
import os
import time
from threading import Lock
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from PIL import Image

from insightface.app import FaceAnalysis


API_KEY = os.getenv("BEAS_RENDER_API_KEY", "beas_7f9d9a8c2e4b44e8a1f3d0c6_render_secret").strip()
MODEL_NAME = os.getenv("INSIGHTFACE_MODEL", "buffalo_l").strip() or "buffalo_l"
MODEL_ROOT = os.getenv("INSIGHTFACE_MODEL_ROOT", "/tmp/insightface").strip() or "/tmp/insightface"
DET_SIZE = int(os.getenv("INSIGHTFACE_DET_SIZE", "640"))
CACHE_MAX_ITEMS = int(os.getenv("EMBEDDING_CACHE_MAX_ITEMS", "512"))

app = FastAPI(title="BEAS InsightFace Service", version="1.0.0")
model_lock = Lock()
face_app: Optional[FaceAnalysis] = None
embedding_cache: Dict[str, Dict] = {}
enrolled_cache: Dict[str, List[float]] = {}
resolved_model_root: Optional[str] = None


class EmbedRequest(BaseModel):
    probeImage: Optional[str] = None
    image: Optional[str] = None
    image_b64: Optional[str] = None
    cacheKey: Optional[str] = None


class SyncEmbedding(BaseModel):
    studentId: str
    embedding: List[float]


class SyncRequest(BaseModel):
    embeddings: List[SyncEmbedding]


class MatchRequest(BaseModel):
    probeImage: Optional[str] = None
    image: Optional[str] = None
    image_b64: Optional[str] = None
    embedding: Optional[List[float]] = None
    cacheKey: Optional[str] = None


def require_api_key(x_beas_api_key: Optional[str]) -> None:
    if API_KEY and x_beas_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def resolve_model_root() -> str:
    global resolved_model_root
    if resolved_model_root is not None:
        return resolved_model_root

    candidates = [MODEL_ROOT, "/tmp/insightface"]
    for candidate in candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
            probe_path = os.path.join(candidate, ".write_test")
            with open(probe_path, "wb") as probe_file:
                probe_file.write(b"ok")
            os.remove(probe_path)
            resolved_model_root = candidate
            return resolved_model_root
        except Exception:
            continue

    resolved_model_root = "/tmp/insightface"
    return resolved_model_root


def get_model() -> FaceAnalysis:
    global face_app
    if face_app is not None:
        return face_app

    with model_lock:
        if face_app is not None:
            return face_app
        model_root = resolve_model_root()
        instance = FaceAnalysis(
            name=MODEL_NAME,
            root=model_root,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        instance.prepare(ctx_id=-1, det_size=(DET_SIZE, DET_SIZE))
        face_app = instance
        return face_app


def decode_image(image_b64: str) -> Image.Image:
    if "," in image_b64 and image_b64.strip().lower().startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    try:
        binary = base64.b64decode(image_b64, validate=True)
        return Image.open(io.BytesIO(binary)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}") from exc


def cache_hash(image_b64: str, explicit_key: Optional[str]) -> str:
    if explicit_key:
        return explicit_key
    return hashlib.sha256(image_b64.encode("utf-8")).hexdigest()


def normalize_embedding(face) -> List[float]:
    if getattr(face, "normed_embedding", None) is not None:
        return [float(v) for v in face.normed_embedding.tolist()]
    if getattr(face, "embedding", None) is not None:
        vector = np.array(face.embedding, dtype=float)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return [float(v) for v in vector.tolist()]
    raise HTTPException(status_code=422, detail="No embedding found")


def cosine_similarity(left: List[float], right: List[float]) -> float:
    count = min(len(left), len(right))
    if count == 0:
        return 0.0
    a = np.array(left[:count], dtype=float)
    b = np.array(right[:count], dtype=float)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a / a_norm, b / b_norm))


def embed_image_payload(image_b64: str, cache_key: Optional[str]) -> Dict:
    key = cache_hash(image_b64, cache_key)
    cached = embedding_cache.get(key)
    if cached:
        return {**cached, "cache": "hit"}

    image = decode_image(image_b64)
    faces = get_model().get(np.asarray(image))
    if not faces:
        raise HTTPException(status_code=422, detail="No face detected")

    embedding = normalize_embedding(faces[0])
    result = {
        "embedding": embedding,
        "model": MODEL_NAME,
        "cacheKey": key,
        "createdAt": int(time.time()),
    }
    if len(embedding_cache) >= CACHE_MAX_ITEMS:
        oldest_key = next(iter(embedding_cache))
        embedding_cache.pop(oldest_key, None)
    embedding_cache[key] = result
    return {**result, "cache": "miss"}


@app.on_event("startup")
def warm_model() -> None:
    get_model()


@app.get("/health")
def health() -> Dict:
    return {
        "ok": True,
        "model": MODEL_NAME,
        "modelRoot": resolve_model_root(),
        "embeddingCacheSize": len(embedding_cache),
        "enrolledCacheSize": len(enrolled_cache),
    }


@app.post("/embed")
def embed(payload: EmbedRequest, x_beas_api_key: Optional[str] = Header(default=None)) -> Dict:
    require_api_key(x_beas_api_key)
    image_b64 = payload.probeImage or payload.image or payload.image_b64 or ""
    if not image_b64:
        raise HTTPException(status_code=400, detail="No image provided")

    return embed_image_payload(image_b64, payload.cacheKey)


@app.post("/sync")
def sync(payload: SyncRequest, x_beas_api_key: Optional[str] = Header(default=None)) -> Dict:
    require_api_key(x_beas_api_key)
    enrolled_cache.clear()
    for item in payload.embeddings:
        enrolled_cache[str(item.studentId)] = [float(v) for v in item.embedding]
    return {"ok": True, "count": len(enrolled_cache)}


@app.post("/match")
def match(payload: MatchRequest, x_beas_api_key: Optional[str] = Header(default=None)) -> Dict:
    require_api_key(x_beas_api_key)
    if not enrolled_cache:
        raise HTTPException(status_code=409, detail="No enrolled embeddings synced")

    probe = payload.embedding
    if probe is None:
        image_b64 = payload.probeImage or payload.image or payload.image_b64 or ""
        if not image_b64:
            raise HTTPException(status_code=400, detail="No probe image or embedding provided")
        probe = embed_image_payload(image_b64, payload.cacheKey)["embedding"]

    best_id = None
    best_score = 0.0
    for student_id, enrolled in enrolled_cache.items():
        score = cosine_similarity(probe, enrolled)
        if score > best_score:
            best_id = student_id
            best_score = score

    return {
        "matchedStudentId": best_id,
        "score": best_score,
        "enrolledCacheSize": len(enrolled_cache),
    }