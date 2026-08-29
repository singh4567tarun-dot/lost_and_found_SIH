import math
from datetime import datetime
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

# 1. Initialize GPU Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[ML Engine] Initializing models on device: {device}")

# 2. Load Models onto GPU
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
    """Generates 512-dimensional normalized image vector using CLIP on GPU."""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)

        # Handle Transformers output wrappers
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
    """Generates 512-dimensional CLIP text embedding for Cross-Modal matching with images."""
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


# --- CALIBRATION FUNCTIONS ---


def calibrate_image_score(raw_score: float) -> float:
    """Normalizes raw CLIP image-to-image cosine similarity (naturally 0.62 to 0.90) onto 0.0 to 1.0."""
    if raw_score <= 0.62:
        return 0.0
    min_val = 0.62
    max_val = 0.90
    calibrated = (raw_score - min_val) / (max_val - min_val)
    return float(np.clip(calibrated, 0.0, 1.0))


def calibrate_cross_modal_score(raw_score: float) -> float:
    """Normalizes raw CLIP text-image cosine similarity (naturally 0.15 to 0.35) onto 0.0 to 1.0."""
    if raw_score <= 0.15:
        return 0.0
    min_val = 0.15
    max_val = 0.35
    calibrated = (raw_score - min_val) / (max_val - min_val)
    return float(np.clip(calibrated, 0.0, 1.0))


# --- MATHEMATICAL SIMILARITY FORMULAS ---


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates cosine similarity between two normalized vectors."""
    if vec1 is None or vec2 is None:
        return 0.0
    sim = np.dot(vec1, vec2)
    return float(np.clip(sim, 0.0, 1.0))


def compute_haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculates physical distance in meters between two geographical coordinates."""
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
    """Calculates S_geo using Gaussian radial decay kernel (default radius sigma = 200m)."""
    distance = compute_haversine_distance(lat1, lon1, lat2, lon2)
    s_geo = math.exp(-((distance**2) / (2 * (sigma**2))))
    return s_geo


def compute_temporal_similarity(
    lost_time_str: str, found_time_str: str, tau_hours: float = 48.0
) -> float:
    """Calculates S_time using asymmetric exponential decay kernel."""
    try:
        t_lost = datetime.fromisoformat(lost_time_str)
        t_found = datetime.fromisoformat(found_time_str)

        delta_t_seconds = (t_found - t_lost).total_seconds()
        delta_t_hours = delta_t_seconds / 3600.0

        if delta_t_hours < -2.0:
            return 0.0

        delta_t_hours = max(0.0, delta_t_hours)
        s_time = math.exp(-delta_t_hours / tau_hours)
        return s_time
    except Exception as e:
        print(f"Error computing temporal similarity: {e}")
        return 0.5


def compute_fusion_score(
    s_text: float,
    s_image: float,
    s_cross: float,
    s_geo: float,
    s_time: float,
    image_mode: str = "none",
) -> float:
    """Synthesizes calibrated component scores into composite score S_total."""
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


if __name__ == "__main__":
    print("\n--- RUNNING ML ENGINE UNIT TEST ---")
    txt_lost = "Lost black leather wallet near central library"
    txt_found = "Found dark leather wallet near central library"

    v_lost = extract_text_embedding(txt_lost)
    v_found = extract_text_embedding(txt_found)

    s_text = compute_cosine_similarity(v_lost, v_found)
    s_geo = compute_geospatial_similarity(
        25.4920, 81.8639, 25.4924, 81.8642, sigma=200.0
    )
    s_time = compute_temporal_similarity(
        "2026-08-28T10:00:00", "2026-08-28T13:00:00", tau_hours=48.0
    )

    s_total = compute_fusion_score(
        s_text=s_text,
        s_image=0.0,
        s_cross=0.0,
        s_geo=s_geo,
        s_time=s_time,
        image_mode="none",
    )
    print(f"Composite Fusion Score (S_total): {s_total} ({s_total * 100:.1f}%)")
    print("--- UNIT TEST COMPLETE ---")