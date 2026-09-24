import numpy as np
import cv2
from PIL import Image, ImageOps


def load_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(path)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return _load_with_pil(path)
    return np.ascontiguousarray(bgr)


def _load_with_pil(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    rgb = np.asarray(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_image(path, bgr):
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("encode failed")
    buf.tofile(path)


def to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_width(bgr, width):
    h, w = bgr.shape[:2]
    scale = width / float(w)
    return cv2.resize(
        bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    ) if scale != 1.0 else bgr