import math
from datetime import datetime
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

# 1. Automatic GPU/CPU Auto-Detection (Runs on RTX 4060 locally or Cloud CPU on HF)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[ML Engine] Initializing PyTorch models on device: {device}")

# 2. Load Lightweight Transformer Backbones
sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
    device
)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- FEATURE EXTRACTION PIPELINES ---


def extract_text_embedding(text: str) -> np.ndarray:
    """Generates 384-dimensional normalized text vector using SBERT."""
    embedding = sbert_model.encode(text, convert_to_numpy=True)
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


def extract_image_embedding(image_path: str) -> np.ndarray:
    """Generates 512-dimensional normalized image vector using CLIP."""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)

        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            features = features.pooler_output
        elif hasattr(features, "image_embeds") and features.image_embeds is not None:
            features = features.image_embeds
        elif not isinstance(features, torch.Tensor):
            features = features[0]

        embedding = features.squeeze(0).cpu().numpy()
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding
    except Exception as e:
        print(f"Error extracting image embedding: {e}")
        return None


def extract_clip_text_embedding(text: str) -> np.ndarray:
    """Generates 512-dimensional CLIP text embedding for Cross-Modal matching."""
    try:
        inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            features = clip_model.get_text_features(**inputs)

        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            features = features.pooler_output
        elif hasattr(features, "text_embeds") and features.text_embeds is not None:
            features = features.text_embeds
        elif not isinstance(features, torch.Tensor):
            features = features[0]

        embedding = features.squeeze(0).cpu().numpy()
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding
    except Exception as e:
        print(f"Error extracting CLIP text embedding: {e}")
        return None


# --- SCORE CALIBRATION MODULES ---


def calibrate_image_score(raw_score: float) -> float:
    """Maps raw CLIP image-to-image similarity (0.62 to 0.90) onto 0.0 to 1.0."""
    if raw_score <= 0.62:
        return 0.0
    min_val, max_val = 0.62, 0.90
    calibrated = (raw_score - min_val) / (max_val - min_val)
    return float(np.clip(calibrated, 0.0, 1.0))


def calibrate_cross_modal_score(raw_score: float) -> float:
    """Maps raw CLIP text-to-image similarity (0.15 to 0.35) onto 0.0 to 1.0."""
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