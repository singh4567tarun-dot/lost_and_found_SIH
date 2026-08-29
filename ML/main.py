import os
import shutil
from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ml_engine import (
    calibrate_cross_modal_score,
    calibrate_image_score,
    compute_cosine_similarity,
    compute_fusion_score,
    compute_geospatial_similarity,
    compute_temporal_similarity,
    extract_clip_text_embedding,
    extract_image_embedding,
    extract_text_embedding,
)

app = FastAPI(title="SIH 2026 Lost & Found ML Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

FOUND_ITEMS_DB = []


@app.post("/api/report-found")
async def report_found_item(
    title: str = Form(...),
    description: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    timestamp: str = Form(...),
    image: Optional[UploadFile] = File(None),
):
    item_id = len(FOUND_ITEMS_DB) + 1
    image_path = None
    image_vec = None

    if image:
        image_path = os.path.join(UPLOAD_DIR, f"found_{item_id}_{image.filename}")
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        image_vec = extract_image_embedding(image_path)

    full_text = f"{title}. {description}"
    sbert_text_vec = extract_text_embedding(full_text)
    clip_text_vec = extract_clip_text_embedding(full_text)

    record = {
        "id": item_id,
        "title": title,
        "description": description,
        "latitude": latitude,
        "longitude": longitude,
        "timestamp": timestamp,
        "image_path": image_path,
        "sbert_text_vector": sbert_text_vec,
        "clip_text_vector": clip_text_vec,
        "image_vector": image_vec,
    }

    FOUND_ITEMS_DB.append(record)
    return {
        "status": "success",
        "message": f"Found item '{title}' registered successfully!",
        "item_id": item_id,
        "has_image": image_vec is not None,
    }


@app.post("/api/match-lost")
async def match_lost_item(
    description: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    timestamp: str = Form(...),
    image: Optional[UploadFile] = File(None),
):
    query_img_path = None
    query_img_vec = None

    if image:
        query_img_path = os.path.join(
            UPLOAD_DIR, f"temp_query_{image.filename}"
        )
        with open(query_img_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        query_img_vec = extract_image_embedding(query_img_path)

    query_sbert_text_vec = extract_text_embedding(description)
    query_clip_text_vec = extract_clip_text_embedding(description)

    has_lost_image = query_img_vec is not None
    candidate_results = []

    for item in FOUND_ITEMS_DB:
        has_found_image = item["image_vector"] is not None

        # 1. Text Similarity (SBERT vs SBERT)
        s_text = compute_cosine_similarity(
            query_sbert_text_vec, item["sbert_text_vector"]
        )

        # 2. Image-to-Image Similarity (CLIP Image vs CLIP Image) with CALIBRATION
        s_image = 0.0
        if has_lost_image and has_found_image:
            raw_img_score = compute_cosine_similarity(
                query_img_vec, item["image_vector"]
            )
            s_image = calibrate_image_score(raw_img_score)

        # 3. Cross-Modal Text-to-Image Similarity (CLIP Text vs CLIP Image) with CALIBRATION
        s_cross = 0.0
        image_mode = "none"

        if has_lost_image and has_found_image:
            image_mode = "both"
        elif has_lost_image and not has_found_image:
            raw_cross = compute_cosine_similarity(
                query_img_vec, item["clip_text_vector"]
            )
            s_cross = calibrate_cross_modal_score(raw_cross)
            image_mode = "one_side"
        elif not has_lost_image and has_found_image:
            raw_cross = compute_cosine_similarity(
                query_clip_text_vec, item["image_vector"]
            )
            s_cross = calibrate_cross_modal_score(raw_cross)
            image_mode = "one_side"

        # 4. Geospatial & Temporal Similarity
        s_geo = compute_geospatial_similarity(
            latitude, longitude, item["latitude"], item["longitude"]
        )
        s_time = compute_temporal_similarity(timestamp, item["timestamp"])

        # 5. Composite Fusion Score
        s_total = compute_fusion_score(
            s_text, s_image, s_cross, s_geo, s_time, image_mode=image_mode
        )

        candidate_results.append(
            {
                "found_item_id": item["id"],
                "title": item["title"],
                "description": item["description"],
                "confidence_score": s_total,
                "match_percentage": f"{round(s_total * 100, 1)}%",
                "score_breakdown": {
                    "text_score": round(s_text, 3),
                    "image_score": round(s_image, 3),
                    "cross_modal_score": round(s_cross, 3),
                    "geo_score": round(s_geo, 3),
                    "time_score": round(s_time, 3),
                    "matching_mode": image_mode,
                },
            }
        )

    candidate_results.sort(key=lambda x: x["confidence_score"], reverse=True)

    if query_img_path and os.path.exists(query_img_path):
        os.remove(query_img_path)

    return {"status": "success", "matches": candidate_results}