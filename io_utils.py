from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def read_image(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Nie udało się wczytać obrazu: {path}")
    return img


def write_image(path: str, image: np.ndarray) -> None:
    ext = Path(path).suffix.lower()
    if not ext:
        ext = ".png"
        path += ext

    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Nie udało się zakodować obrazu do formatu: {ext}")
    encoded.tofile(path)
