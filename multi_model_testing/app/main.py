from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from typing import List, Optional, Dict
from io import BytesIO
import base64
from pathlib import Path

from .utils import load_image_to_numpy
from .model_registry import (
    list_detection_models,
    list_classification_models,
    get_detection_model_config,
    get_classification_model_config,
    get_detection_labels,
    get_classification_labels,
)
from .inference import (
    DetectionModel,
    ClassificationModel,
    Detection,
    ClassificationResult,
    draw_detections,
    encode_crop_to_base64,
)
from .models import DetectionOut, PredictResponse, ClassificationOut, ImageResult


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

# Model storage directories (for uploaded models)
MODELS_DIR = BASE_DIR / "models"
DET_MODELS_DIR = MODELS_DIR / "detection"
CLS_MODELS_DIR = MODELS_DIR / "classification"
DET_MODELS_DIR.mkdir(parents=True, exist_ok=True)
CLS_MODELS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Multi Model Detection + Classification")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory caches for loaded ONNX sessions
det_cache: Dict[str, DetectionModel] = {}
cls_cache: Dict[str, ClassificationModel] = {}


def load_det(name: str) -> DetectionModel:
    if name not in det_cache:
        cfg = get_detection_model_config(name)
        # DetectionModel now auto-uses GPU if available
        det_cache[name] = DetectionModel(cfg["model_path"], cfg["labels_path"])
    return det_cache[name]


def load_cls(name: str) -> ClassificationModel:
    if name not in cls_cache:
        cfg = get_classification_model_config(name)
        # ClassificationModel now auto-uses GPU if available
        cls_cache[name] = ClassificationModel(cfg["model_path"], cfg["labels_path"])
    return cls_cache[name]


# ---------- UI pages ----------


@app.get("/")
async def root():
    """Main testing UI"""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=500,
            detail="static/index.html not found. Create it under project_root/static/",
        )
    return FileResponse(str(index_path))


@app.get("/upload")
async def upload_page():
    """Model upload UI"""
    page_path = STATIC_DIR / "upload.html"
    if not page_path.exists():
        raise HTTPException(
            status_code=500,
            detail="static/upload.html not found. Create it under project_root/static/",
        )
    return FileResponse(str(page_path))


# ---------- API: model listing ----------


@app.get("/models")
async def get_all_models():
    """
    Return available detection & classification models and their labels.
    This automatically picks up new models if model_registry scans folders.
    """
    return {
        "detection_models": list_detection_models(),
        "classification_models": list_classification_models(),
        "detection_labels": get_detection_labels(),
        "classification_labels": get_classification_labels(),
    }


# ---------- API: prediction ----------


@app.post("/predict", response_model=PredictResponse)
async def predict(
    files: List[UploadFile] = File(...),           # multiple images
    detection_models: str = Form(...),             # comma-separated names (can be empty)
    classification_models: str = Form(...),        # comma-separated names (can be empty)
    det_thresh: float = Form(0.3),
    cls_thresh: float = Form(0.5),
    det_labels: str = Form(""),                    # optional comma-separated
    cls_labels: str = Form(""),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    det_list = [m.strip() for m in detection_models.split(",") if m.strip()]
    cls_list = [m.strip() for m in classification_models.split(",") if m.strip()]

    # NEW: allow detection-only OR classification-only,
    # but at least one of them must be non-empty
    if not det_list and not cls_list:
        raise HTTPException(
            status_code=400,
            detail="Select at least one detection or classification model.",
        )

    all_det = list_detection_models()
    all_cls = list_classification_models()

    for m in det_list:
        if m not in all_det:
            raise HTTPException(status_code=400, detail=f"Unknown detection model: {m}")
    for m in cls_list:
        if m not in all_cls:
            raise HTTPException(status_code=400, detail=f"Unknown classification model: {m}")

    selected_det_labels: Optional[List[str]] = None
    selected_cls_labels: Optional[List[str]] = None
    if det_labels:
        selected_det_labels = [x for x in det_labels.split(",") if x]
    if cls_labels:
        selected_cls_labels = [x for x in cls_labels.split(",") if x]

    results: List[ImageResult] = []

    for idx, upload in enumerate(files):
        image_bytes = await upload.read()
        img_bgr, pil_img = load_image_to_numpy(image_bytes)
        h, w = img_bgr.shape[:2]

        detections: List[Detection] = []

        # ---- CASE 1: detection models selected (with or without classification) ----
        if det_list:
            # 1) run all detection models and gather detections
            for det_name in det_list:
                det_model = load_det(det_name)
                dets = det_model.predict(img_bgr, score_thr=det_thresh)
                for d in dets:
                    if selected_det_labels is not None and d.class_name not in selected_det_labels:
                        continue
                    d.det_model = det_name
                    detections.append(d)

            # 2) per detection, crop once and run all classification models (if any)
            for d in detections:
                x0, y0, x1, y1 = d.bbox
                x0 = max(0, min(x0, w - 1))
                y0 = max(0, min(y0, h - 1))
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))

                if x1 <= x0 or y1 <= y0:
                    continue

                crop = pil_img.crop((x0, y0, x1, y1))
                d.crop_image_base64 = encode_crop_to_base64(crop)
                d.cls_results = []

                # classification optional: if no cls_list, this loop is skipped
                for cls_name in cls_list:
                    cls_model = load_cls(cls_name)
                    label, score = cls_model.predict(crop)

                    keep_label = score >= cls_thresh
                    if selected_cls_labels is not None:
                        keep_label = keep_label and (label in selected_cls_labels)

                    d.cls_results.append(
                        ClassificationResult(
                            model_name=cls_name,
                            label=label if keep_label else None,
                            score=score,
                        )
                    )

        # ---- CASE 2: NO detection models, ONLY classification models ----
        elif cls_list:
            # treat the whole image as a single "detection"
            full_bbox = (0, 0, w - 1, h - 1)
            full_crop = pil_img  # full image
            det = Detection(
                bbox=full_bbox,
                score=1.0,
                class_id=0,
                class_name="image",       # detection label for classification-only mode
                det_model=None,
                crop_image_base64=encode_crop_to_base64(full_crop),
                cls_results=[],
            )

            for cls_name in cls_list:
                cls_model = load_cls(cls_name)
                label, score = cls_model.predict(full_crop)

                keep_label = score >= cls_thresh
                if selected_cls_labels is not None:
                    keep_label = keep_label and (label in selected_cls_labels)

                det.cls_results.append(
                    ClassificationResult(
                        model_name=cls_name,
                        label=label if keep_label else None,
                        score=score,
                    )
                )

            detections.append(det)

        # 3) annotate image for visualization (works for both cases)
        annotated = draw_detections(pil_img, detections)
        buf = BytesIO()
        annotated.save(buf, format="PNG")
        annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        detections_out: List[DetectionOut] = []
        for d in detections:
            detections_out.append(
                DetectionOut(
                    bbox=list(d.bbox),
                    score=d.score,
                    class_id=d.class_id,
                    class_name=f"{d.det_model}:{d.class_name}" if d.det_model else d.class_name,
                    crop_image_base64=d.crop_image_base64,
                    classifications=[
                        ClassificationOut(
                            model_name=r.model_name,
                            label=r.label,
                            score=r.score,
                        )
                        for r in d.cls_results
                    ],
                )
            )

        results.append(
            ImageResult(
                filename=upload.filename or f"image_{idx+1}",
                detections=detections_out,
                annotated_image_base64=annotated_b64,
            )
        )

    return PredictResponse(results=results)


# ---------- API: model upload ----------


@app.post("/upload_model")
async def upload_model(
    model_type: str = Form(...),          # "detection" or "classification"
    model_file: UploadFile = File(...),
    label_file: UploadFile = File(...),
):
    """
    Upload a new detection/classification model + label.txt.

    Rules:
    - model_type: "detection" or "classification"
    - label file must end with .txt
    - base name of model and label must match
      (example: person_face.onnx + person_face.txt)
    - if a model/label with that base name already exists in that folder, reject
    """

    model_type = model_type.lower().strip()
    if model_type not in ("detection", "classification"):
        raise HTTPException(status_code=400, detail="model_type must be 'detection' or 'classification'")

    if not label_file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Label file must be a .txt file")

    model_name = Path(model_file.filename).stem
    label_name = Path(label_file.filename).stem

    if model_name != label_name:
        raise HTTPException(
            status_code=400,
            detail="Model file name and label file name must match (without extension)",
        )

    dest_dir = DET_MODELS_DIR if model_type == "detection" else CLS_MODELS_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    model_ext = Path(model_file.filename).suffix.lower()  # allow .onnx, .pt, etc.
    dest_model_path = dest_dir / f"{model_name}{model_ext}"
    dest_label_path = dest_dir / f"{model_name}.txt"

    if dest_model_path.exists() or dest_label_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"A {model_type} model named '{model_name}' already exists",
        )

    # Save model
    model_bytes = await model_file.read()
    dest_model_path.write_bytes(model_bytes)

    # Save labels
    label_bytes = await label_file.read()
    dest_label_path.write_bytes(label_bytes)

    # Clear cache for safety
    if model_type == "detection":
        det_cache.clear()
    else:
        cls_cache.clear()

    return {
        "status": "ok",
        "message": f"Uploaded {model_type} model '{model_name}'",
        "model_type": model_type,
        "model_name": model_name,
        "model_path": str(dest_model_path),
        "labels_path": str(dest_label_path),
    }
