from io import BytesIO
from typing import Tuple
from PIL import Image
import numpy as np


def load_image_to_numpy(data: bytes) -> Tuple[np.ndarray, Image.Image]:
    """Return (np_array_in_BGR, PIL_image_RGB)."""
    pil_img = Image.open(BytesIO(data)).convert("RGB")
    np_img = np.array(pil_img)[:, :, ::-1]  # BGR for OpenCV-style
    return np_img, pil_img
