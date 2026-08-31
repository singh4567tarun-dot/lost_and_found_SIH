import math
import os
import time
from datetime import datetime
import numpy as np
import requests

# Hugging Face Free Serverless Inference Endpoints
SBERT_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
CLIP_API_URL = (
    "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"
)

# Reads token securely from Render Environment Variables
HF_TOKEN = os.getenv("HF_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


def query_hf_api(url: str, payload=None, data=None, retries: int = 3):
    """Queries Hugging Face with automatic retry if model is warming up."""
    for attempt in range(retries):
        try:
            if data is not None:
                headers = {
                    **HEADERS,
                    "Content-Type": "application/octet-stream",
                }
                res = requests.post(
                    url, headers=headers, data=data, timeout=15
                )
            else:
                res = requests.post(
                    url, headers=HEADERS, json=payload, timeout=15
                )

            if res.status_code == 200:
                return res.json()
            elif res.status_code == 503:
                time.sleep(4)
                continue
            else:
                print(f"HF API {res.status_code}: {res.text}")
                break
        except Exception as e:
            print(f"HF API connection error: {e}")
            time.sleep(2)
    return None


def extract_text_embedding(text: str) -> np.ndarray:
    result = query_hf_api(
        SBERT_API_URL,
        payload={"inputs": text, "options": {"wait_for_model": True}},
    )
    if result is not None:
        emb = np.array(result)
        if emb.ndim > 1:
            emb = np.mean(emb, axis=0)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    np.random.seed(abs(hash(text)) % (2**32))
    vec = np.random.rand(384)
    return vec / np.linalg.norm(vec)


def extract_image_embedding(
    image_path_or_url: str = None, image_bytes: bytes = None
) -> np.ndarray:
    """Extracts 512-dim CLIP visual embedding from a URL, local file path, or raw bytes."""
    try:
        img_data = None
        if image_bytes:
            img_data = image_bytes
        elif image_path_or_url:
            if image_path_or_url.startswith(
                "http://"
            ) or image_path_or_url.startswith("https://"):
                img_res = requests.get(image_path_or_url, timeout=10)
                if img_res.status_code == 200:
                    img_data = img_res.content
            elif os.path.exists(image_path_or_url):
                with open(image_path_or_url, "rb") as f:
                    img_data = f.read()

        if img_data:
            result = query_hf_api(CLIP_API_URL, data=img_data)
            if result is not None:
                emb = np.array(result)
                if emb.ndim > 1:
                    emb = np.mean(emb, axis=0)
                norm = np.linalg.norm(emb)
                return emb / norm if norm > 0 else emb
    except Exception as e:
        print(f"Image embedding extraction error: {e}")

    vec = np.random.rand(512)
    return vec / np.linalg.norm(vec)


def extract_clip_text_embedding(text: str) -> np.ndarray:
    result = query_hf_api(
        CLIP_API_URL,
        payload={"inputs": text, "options": {"wait_for_model": True}},
    )
    if result is not None:
        emb = np.array(result)
        if emb.ndim > 1:
            emb = np.mean(emb, axis=0)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    np.random.seed(abs(hash(text)) % (2**32))
    vec = np.random.rand(512)
    return vec / np.linalg.norm(vec)


def calibrate_image_score(raw_score: float) -> float:
    if raw_score <= 0.62:
        return 0.0
    return float(np.clip((raw_score - 0.62) / 0.28, 0.0, 1.0))


def calibrate_cross_modal_score(raw_score: float) -> float:
    if raw_score <= 0.15:
        return 0.0
    return float(np.clip((raw_score - 0.15) / 0.20, 0.0, 1.0))


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if vec1 is None or vec2 is None:
        return 0.0
    return float(np.clip(np.dot(vec1, vec2), 0.0, 1.0))


def compute_haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def compute_geospatial_similarity(
    lat1: float, lon1: float, lat2: float, lon2: float, sigma: float = 200.0
) -> float:
    dist = compute_haversine_distance(lat1, lon1, lat2, lon2)
    return math.exp(-((dist**2) / (2 * (sigma**2))))


def compute_temporal_similarity(
    lost_time_str: str, found_time_str: str, tau_hours: float = 48.0
) -> float:
    try:
        t_lost = datetime.fromisoformat(lost_time_str.replace("Z", "+00:00"))
        t_found = datetime.fromisoformat(found_time_str.replace("Z", "+00:00"))
        delta_t_hours = (t_found - t_lost).total_seconds() / 3600.0
        if delta_t_hours < -2.0:
            return 0.0
        return math.exp(-max(0.0, delta_t_hours) / tau_hours)
    except Exception:
        return 0.5


def compute_fusion_score(
    s_text: float,
    s_image: float,
    s_cross: float,
    s_geo: float,
    s_time: float,
    image_mode: str = "none",
) -> float:
    if image_mode == "both":
        w_t, w_i, w_c, w_g, w_tau = 0.25, 0.40, 0.00, 0.175, 0.175
    elif image_mode == "one_side":
        w_t, w_i, w_c, w_g, w_tau = 0.30, 0.00, 0.35, 0.175, 0.175
    else:
        w_t, w_i, w_c, w_g, w_tau = 0.60, 0.00, 0.00, 0.20, 0.20
    return round(
        float(
            w_t * s_text
            + w_i * s_image
            + w_c * s_cross
            + w_g * s_geo
            + w_tau * s_time
        ),
        4,
    )