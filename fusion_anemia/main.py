# =========================
# Part 1) Imports + App + Config
# =========================
import asyncio
import base64
import math
import os
import time
import uuid
from typing import Any, Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Anemia Fusion API (Option B)", version="1.0")

# Downstream services (Railway). Defaults are your URLs, but can be overridden by env vars later.
IMAGE_API_URL = os.getenv(
    "IMAGE_API_URL",
    "https://graduation-project-production-174c.up.railway.app",
)
QUESTIONNAIRE_API_URL = os.getenv(
    "QUESTIONNAIRE_API_URL",
    "https://graduation-project-production-bfe4.up.railway.app",
)
NLP_API_URL = os.getenv(
    "NLP_API_URL",
    "https://graduation-project-production-4827.up.railway.app",
)

# Timeouts (seconds) - can tune later
IMAGE_TIMEOUT = float(os.getenv("IMAGE_TIMEOUT", "8.0"))
Q_TIMEOUT = float(os.getenv("Q_TIMEOUT", "6.0"))
NLP_TIMEOUT = float(os.getenv("NLP_TIMEOUT", "10.0"))

# Default fusion weights (can tune later)
W_IMAGE = float(os.getenv("W_IMAGE", "0.6"))
W_Q = float(os.getenv("W_Q", "0.3"))
W_T = float(os.getenv("W_T", "0.1"))

# Image API returns hb_value, so we map Hb to probability around a threshold
HB_THRESHOLD = float(os.getenv("HB_THRESHOLD", "12.5"))

# =========================
# Part 2) Request Schema
# =========================
class ImagePayload(BaseModel):
    filename: str = Field(..., examples=["eye.jpg"])
    base64_data: str = Field(..., description="Base64 image bytes (jpg/png).")


class FusionRequest(BaseModel):
    image: ImagePayload
    questionnaire: Dict[str, Any] = Field(default_factory=dict)
    free_text: str = ""

# =========================
# Part 3) Normalize outputs to probabilities [0..1]
# =========================
def hb_to_probability(hb: float) -> float:
    """
    Convert Hb value to anemia risk probability.
    Lower Hb => higher anemia probability.

    Using sigmoid around HB_THRESHOLD.
    """
    x = hb - HB_THRESHOLD
    p = 1.0 / (1.0 + math.exp(x))
    return float(min(max(p, 0.0), 1.0))


def nlp_to_probability(nlp_json: Dict[str, Any]) -> float:
    """
    Extract anemia score from NLP output.
    Expected: results_map['anemia']['percentage'] (0..100) OR (0..1)
    """
    results_map = nlp_json.get("results_map") or {}
    anemia_obj = results_map.get("anemia") or {}
    pct = anemia_obj.get("percentage", 0.0)

    try:
        pct_f = float(pct)
    except Exception:
        pct_f = 0.0

    if pct_f > 1.0:
        pct_f = pct_f / 100.0

    return float(min(max(pct_f, 0.0), 1.0))

# =========================
# Part 4) Fusion logic (weighted average with fallback)
# =========================
def fuse_probs(
    p_img: Optional[float],
    p_q: Optional[float],
    p_t: Optional[float],
) -> Tuple[float, Dict[str, float]]:
    """
    Weighted average with renormalization:
    - If a modality is missing (None), drop its weight and renormalize.
    """
    weights = {"image": W_IMAGE, "questionnaire": W_Q, "text": W_T}
    available = {
        "image": p_img is not None,
        "questionnaire": p_q is not None,
        "text": p_t is not None,
    }

    denom = sum(weights[k] for k in weights if available[k])
    if denom <= 0:
        raise ValueError("No modality predictions available (all downstream calls failed).")

    used = {k: (weights[k] / denom if available[k] else 0.0) for k in weights}

    p_final = 0.0
    if p_img is not None:
        p_final += used["image"] * p_img
    if p_q is not None:
        p_final += used["questionnaire"] * p_q
    if p_t is not None:
        p_final += used["text"] * p_t

    return float(p_final), used

# =========================
# Part 5) Downstream API calls
# =========================
async def call_image_api(client: httpx.AsyncClient, image_bytes: bytes, filename: str) -> Tuple[float, Dict[str, Any]]:
    """
    Calls anemia image service: POST /predict as multipart file 'file'
    Expects hb_value in response.
    """
    r = await client.post(
        f"{IMAGE_API_URL}/predict",
        files={"file": (filename, image_bytes, "application/octet-stream")},
        timeout=IMAGE_TIMEOUT,
    )
    r.raise_for_status()
    j = r.json()
    hb = float(j.get("hb_value"))
    return hb_to_probability(hb), j


async def call_questionnaire_api(client: httpx.AsyncClient, questionnaire: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Calls anemia questionnaire service: POST /predict JSON
    Expects anemia_probability in response.
    """
    r = await client.post(
        f"{QUESTIONNAIRE_API_URL}/predict",
        json=questionnaire,
        timeout=Q_TIMEOUT,
    )
    r.raise_for_status()
    j = r.json()
    p = float(j.get("anemia_probability"))
    p = float(min(max(p, 0.0), 1.0))
    return p, j


async def call_nlp_api(client: httpx.AsyncClient, text: str) -> Tuple[float, Dict[str, Any]]:
    """
    Calls NLP service: POST /predict JSON {"text": "..."}
    Expects results_map with anemia percentage.
    """
    if not text.strip():
        raise RuntimeError("free_text is empty (NLP skipped).")

    r = await client.post(
        f"{NLP_API_URL}/predict",
        json={"text": text},
        timeout=NLP_TIMEOUT,
    )
    r.raise_for_status()
    j = r.json()

    # Your NLP service sometimes returns {"error": "..."}
    if "error" in j:
        raise RuntimeError(str(j["error"]))

    return nlp_to_probability(j), j

# =========================
# Part 6) Endpoints
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "image_api": IMAGE_API_URL,
        "questionnaire_api": QUESTIONNAIRE_API_URL,
        "nlp_api": NLP_API_URL,
    }


@app.post("/predict/anemia/fusion")
async def predict_anemia_fusion(req: FusionRequest):
    start = time.perf_counter()
    patient_id = uuid.uuid4().hex[:12]

    # Decode image from base64
    try:
        image_bytes = base64.b64decode(req.image.base64_data)
        if not image_bytes:
            raise ValueError("decoded image is empty")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

    # Run 3 calls concurrently (Option B core idea)
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            call_image_api(client, image_bytes, req.image.filename),
            call_questionnaire_api(client, req.questionnaire),
            call_nlp_api(client, req.free_text),
            return_exceptions=True,
        )

    # Unpack results with graceful failure
    p_img = p_q = p_t = None
    raw_img = raw_q = raw_t = None
    err_img = err_q = err_t = None

    for idx, res in enumerate(results):
        if isinstance(res, Exception):
            if idx == 0:
                err_img = str(res)
            elif idx == 1:
                err_q = str(res)
            else:
                err_t = str(res)
            continue

        p, raw = res
        if idx == 0:
            p_img, raw_img = p, raw
        elif idx == 1:
            p_q, raw_q = p, raw
        else:
            p_t, raw_t = p, raw

    # Fuse
    try:
        p_final, used_w = fuse_probs(p_img, p_q, p_t)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Fusion failed: {e}")

    diagnosis = "anemia_positive" if p_final >= 0.5 else "anemia_negative"
    confidence = max(p_final, 1 - p_final)
    ms = int((time.perf_counter() - start) * 1000)

    return {
        "status": "success",
        "patient_id": patient_id,
        "overall_prediction": {
            "diagnosis": diagnosis,
            "probability": round(p_final, 4),
            "confidence": round(confidence, 4),
        },
        "modality_predictions": {
            "image": {"ok": p_img is not None, "probability": p_img, "error": err_img, "raw": raw_img},
            "questionnaire": {"ok": p_q is not None, "probability": p_q, "error": err_q, "raw": raw_q},
            "text": {"ok": p_t is not None, "probability": p_t, "error": err_t, "raw": raw_t},
        },
        "fusion_weights_used": {k: round(v, 4) for k, v in used_w.items()},
        "processing_time_ms": ms,
    }
