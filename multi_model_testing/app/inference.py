import base64
import subprocess
from io import BytesIO
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import onnxruntime as ort
from torchvision import transforms


def load_labels(label_path: str) -> List[str]:
    with open(label_path, "r") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    return labels


def get_color(idx: int) -> Tuple[int, int, int]:
    np.random.seed(idx)
    color = np.random.randint(0, 255, 3)
    return int(color[0]), int(color[1]), int(color[2])


# -------- GPU / CPU session helper --------

def _pick_best_cuda_device() -> Optional[int]:
    """
    Try to pick the GPU with the most free memory using nvidia-smi.
    If anything fails, return None and let ORT use default device.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        frees = [int(x.strip()) for x in out.splitlines() if x.strip()]
        if not frees:
            return None
        best_idx = max(range(len(frees)), key=lambda i: frees[i])
        return best_idx
    except Exception:
        return None


def create_ort_session(model_path: str, use_cuda: bool = True) -> ort.InferenceSession:
    """
    Create an ONNX Runtime session that:
    - uses CUDA if available (and use_cuda=True),
    - falls back to CPU otherwise,
    - if multiple GPUs exist, picks the one with most free memory.
    """
    available = ort.get_available_providers()

    if use_cuda and "CUDAExecutionProvider" in available:
        device_id = _pick_best_cuda_device()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        if device_id is not None:
            provider_options = [{"device_id": device_id}, {}]
            return ort.InferenceSession(
                model_path,
                providers=providers,
                provider_options=provider_options,
            )
        else:
            # CUDA available but we couldn't pick a device – let ORT decide
            return ort.InferenceSession(model_path, providers=providers)

    # CPU-only path
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


# -------- data structures --------

@dataclass
class ClassificationResult:
    model_name: str
    label: Optional[str]
    score: Optional[float]


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    score: float
    class_id: int
    class_name: str
    det_model: Optional[str] = None
    crop_image_base64: Optional[str] = None
    cls_results: List[ClassificationResult] = field(default_factory=list)


# -------- detection model --------

class DetectionModel:
    def __init__(
        self,
        model_path: str,
        labels_path: str,
        score_thr: float = 0.3,
        cuda: bool = True,
    ):
        """
        cuda=True => try to use GPU if available; otherwise CPU.
        """
        self.labels = load_labels(labels_path)
        self.score_thr_default = score_thr

        # automatically choose GPU/CPU
        self.session = create_ort_session(model_path, use_cuda=cuda)

        self.outname = [o.name for o in self.session.get_outputs()]
        self.inname = [i.name for i in self.session.get_inputs()]

    def _letterbox(
        self,
        img: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114),
        scaleup: bool = True,
    ):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return img_padded, r, (dw, dh)

    def _preprocess(self, img_bgr: np.ndarray):
        image = img_bgr.copy()
        image, ratio, dwdh = self._letterbox(image)
        image = image.transpose((2, 0, 1))[None]
        image = np.ascontiguousarray(image).astype(np.float32) / 255.0
        return {self.inname[0]: image}, ratio, dwdh

    def predict(self, img_bgr: np.ndarray, score_thr: Optional[float] = None) -> List[Detection]:
        if score_thr is None:
            score_thr = self.score_thr_default

        inp, ratio, dwdh = self._preprocess(img_bgr)
        outputs = self.session.run(self.outname, inp)[0]

        h, w = img_bgr.shape[:2]
        detections: List[Detection] = []

        for det in outputs:
            batch_id, x0, y0, x1, y1, cls_id, score = det
            if score < score_thr:
                continue

            cls_id = int(cls_id)
            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh * 2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            x0_i, y0_i, x1_i, y1_i = box

            x0_i = max(0, min(x0_i, w - 1))
            y0_i = max(0, min(y0_i, h - 1))
            x1_i = max(0, min(x1_i, w - 1))
            y1_i = max(0, min(y1_i, h - 1))

            if x1_i <= x0_i or y1_i <= y0_i:
                continue

            if 0 <= cls_id < len(self.labels):
                cls_name = self.labels[cls_id]
            else:
                cls_name = "cls_%d" % cls_id

            detections.append(
                Detection(
                    bbox=(x0_i, y0_i, x1_i, y1_i),
                    score=float(score),
                    class_id=cls_id,
                    class_name=cls_name,
                )
            )

        return detections


# -------- classification model --------

class ClassificationModel:
    def __init__(self, model_path: str, labels_path: str, cuda: bool = True):
        self.labels = load_labels(labels_path)

        if not model_path:
            raise ValueError("model_path is empty")

        # same GPU/CPU logic
        self.session = create_ort_session(model_path, use_cuda=cuda)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    def preprocess_crop(self, crop_rgb: Image.Image) -> np.ndarray:
        img_t = self.transform(crop_rgb).numpy().astype(np.float32)
        img_t = np.expand_dims(img_t, axis=0)
        return img_t

    def predict(self, crop_rgb: Image.Image) -> Tuple[str, float]:
        x = self.preprocess_crop(crop_rgb)
        outputs = self.session.run([self.output_name], {self.input_name: x})[0]

        pred_class_idx = int(np.argmax(outputs, axis=1)[0])
        if 0 <= pred_class_idx < len(self.labels):
            label = self.labels[pred_class_idx]
        else:
            label = "cls_%d" % pred_class_idx

        logits = outputs[0]
        exp = np.exp(logits - np.max(logits))
        prob = exp / np.sum(exp)
        score = float(prob[pred_class_idx])

        return label, score


# -------- utils for crops & drawing --------

def encode_crop_to_base64(crop: Image.Image) -> str:
    buf = BytesIO()
    crop.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def draw_detections(pil_rgb: Image.Image, detections: List[Detection]) -> Image.Image:
    annotated = pil_rgb.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x0, y0, x1, y1 = det.bbox
        color = get_color(det.class_id)

        draw.rectangle((x0, y0, x1, y1), outline=color, width=2)

        best_cls = None
        for r in det.cls_results:
            if r.label is None or r.score is None:
                continue
            if best_cls is None or r.score > best_cls.score:
                best_cls = r

        text = "%s %.1f%%" % (det.class_name, det.score * 100.0)
        if best_cls is not None:
            text += " | %s:%s %.1f%%" % (
                best_cls.model_name,
                best_cls.label,
                best_cls.score * 100.0,
            )

        text_w = font.getlength(text)
        text_h = font.size
        text_bg = (x0, max(0, y0 - text_h - 4), x0 + text_w + 4, y0)
        draw.rectangle(text_bg, fill=color)
        draw.text((x0 + 2, max(0, y0 - text_h - 2)), text, fill="white", font=font)

    return annotated
