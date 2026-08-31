import math
from datetime import datetime
import numpy as np
import requests

# Hugging Face Free Serverless Inference Endpoints
SBERT_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
CLIP_IMAGE_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/openai/clip-vit-base-patch32"
CLIP_TEXT_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/openai/clip-vit-base-patch32"

print("[ML Engine] Configured for Serverless Cloud Inference (RAM < 50MB)")

# --- FEATURE EXTRACTION PIPELINES ---


def extract_text_embedding(text: str) -> np.ndarray:
    """Extracts 384-dimensional SBERT text embedding via Serverless API."""
    try:
        response = requests.post(
            SBERT_API_URL, json={"inputs": text}, timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            embedding = np.array(data)
            if embedding.ndim > 1:
                embedding = np.mean(embedding, axis=0)
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else embedding
    except Exception as e:
        print(f"Text embedding API error: {e}")

    # Fallback pseudo-vector if API is warming up
    vec = np.random.rand(384)
    return vec / np.linalg.norm(vec)


def extract_image_embedding(image_path: str) -> np.ndarray:
    """Extracts 512-dimensional CLIP image embedding via Serverless API."""
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        response = requests.post(
            CLIP_IMAGE_API_URL, data=img_bytes, timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            embedding = np.array(data)
            if embedding.ndim > 1:
                embedding = np.mean(embedding, axis=0)
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else embedding
    except Exception as e:
        print(f"Image embedding API error: {e}")

    vec = np.random.rand(512)
    return vec / np.linalg.norm(vec)


def extract_clip_text_embedding(text: str) -> np.ndarray:
    """Extracts 512-dimensional CLIP text embedding for Cross-Modal matching."""
    try:
        response = requests.post(
            CLIP_TEXT_API_URL, json={"inputs": text}, timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            embedding = np.array(data)
            if embedding.ndim > 1:
                embedding = np.mean(embedding, axis=0)
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else embedding
    except Exception as e:
        print(f"CLIP Text embedding API error: {e}")

    vec = np.random.rand(512)
    return vec / np.linalg.norm(vec)


# --- SCORE CALIBRATION MODULES ---


def calibrate_image_score(raw_score: float) -> float:
    if raw_score <= 0.62:
        return 0.0
    min_val, max_val = 0.62, 0.90
    calibrated = (raw_score - min_val) / (max_val - min_val)
    return float(np.clip(calibrated, 0.0, 1.0))


def calibrate_cross_modal_score(raw_score: float) -> float:
    if raw_score <= 0.15:
        return 0.0
    min_val, max_val = 0.15, 0.35
    calibrated = (raw_score - min_val) / (max_val - min_val)
    return float(np.clip(calibrated, 0.0, 1.0))


# --- SIMILARITY FORMULAS ---


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if vec1 is None or vec2 is None:
        return 0.0
    sim = np.dot(vec1, vec2)
    return float(np.clip(sim, 0.0, 1.0))


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
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def compute_geospatial_similarity(
    lat1: float, lon1: float, lat2: float, lon2: float, sigma: float = 200.0
) -> float:
    distance = compute_haversine_distance(lat1, lon1, lat2, lon2)
    return math.exp(-((distance**2) / (2 * (sigma**2))))


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

    s_total = (
        (w_t * s_text)
        + (w_i * s_image)
        + (w_c * s_cross)
        + (w_g * s_geo)
        + (w_tau * s_time)
    )
    return round(float(s_total), 4)