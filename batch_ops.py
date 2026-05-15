import os
from pathlib import Path

import cv2
import numpy as np

from io_utils import SUPPORTED_EXTENSIONS, read_image, write_image


def batch_resize_images(input_dir: str, output_dir: str, width: int, height: int) -> tuple[int, int, list[str]]:
    os.makedirs(output_dir, exist_ok=True)
    processed = 0
    skipped = 0
    errors: list[str] = []

    for file_name in os.listdir(input_dir):
        src = os.path.join(input_dir, file_name)
        if not os.path.isfile(src) or Path(src).suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped += 1
            continue

        try:
            img = read_image(src)
            resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            dst = os.path.join(output_dir, file_name)
            write_image(dst, resized)
            processed += 1
        except (OSError, ValueError, cv2.error) as exc:
            skipped += 1
            errors.append(f"{file_name}: {exc}")

    return processed, skipped, errors


def add_text_watermark(
    image: np.ndarray, 
    text: str, 
    opacity: float, 
    scale: float, 
    position: str,
    color: tuple[int, int, int] = (255, 255, 255),
    font_type: int = cv2.FONT_HERSHEY_SIMPLEX
) -> np.ndarray:
    overlay = image.copy()
    h, w = image.shape[:2]
    thickness = max(1, int(scale * 2))
    text_size, baseline = cv2.getTextSize(text, font_type, scale, thickness)
    tw, th = text_size
    margin = 20

    if position == "Lewy górny":
        x, y = margin, margin + th
    elif position == "Prawy górny":
        x, y = w - tw - margin, margin + th
    elif position == "Lewy dolny":
        x, y = margin, h - margin
    elif position == "Środek":
        x, y = (w - tw) // 2, (h + th) // 2
    else:
        x, y = w - tw - margin, h - margin

    cv2.putText(overlay, text, (x, y), font_type, scale, color, thickness, cv2.LINE_AA)
    if baseline > 0:
        cv2.putText(overlay, text, (x, y), font_type, scale, (0, 0, 0), 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0)


def batch_watermark_images(
    input_dir: str, 
    output_dir: str, 
    text: str, 
    opacity: float, 
    scale: float, 
    position: str,
    color: tuple[int, int, int] = (255, 255, 255),
    font_type: int = cv2.FONT_HERSHEY_SIMPLEX
) -> tuple[int, int, list[str]]:
    os.makedirs(output_dir, exist_ok=True)
    processed = 0
    skipped = 0
    errors: list[str] = []

    for file_name in os.listdir(input_dir):
        src = os.path.join(input_dir, file_name)
        if not os.path.isfile(src) or Path(src).suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped += 1
            continue

        try:
            img = read_image(src)
            watermarked = add_text_watermark(img, text, opacity, scale, position, color, font_type)
            dst = os.path.join(output_dir, file_name)
            write_image(dst, watermarked)
            processed += 1
        except (OSError, ValueError, cv2.error) as exc:
            skipped += 1
            errors.append(f"{file_name}: {exc}")

    return processed, skipped, errors
