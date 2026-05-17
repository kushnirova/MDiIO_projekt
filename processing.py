import cv2
import numpy as np


_FACE_PROTOTYPE_CACHE: list[np.ndarray] | None = None


def detect_edges(image: np.ndarray, algorithm: str) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if algorithm == "Canny":
        edges = cv2.Canny(gray, 100, 200)
    elif algorithm == "Sobel":
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sx, sy)
        edges = np.clip(magnitude, 0, 255).astype(np.uint8)
    else:
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        edges = np.clip(np.abs(lap), 0, 255).astype(np.uint8)

    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def threshold_image(image: np.ndarray, algorithm: str, binary_threshold: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if algorithm == "Binary":
        _, out = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY)
    elif algorithm == "Otsu":
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        out = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )

    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def apply_basic_edit(
    image: np.ndarray,
    saturation: float = 1.0,
    contrast: float = 1.0,
    exposure: float = 0.0,
    tint: float = 0.0,
    temperature: float = 0.0,
    brightness: float = 0.0,
    sharpness: float = 0.0,
) -> np.ndarray:
    out = image.astype(np.float32)

    exposure_scale = float(np.power(2.0, exposure))
    out *= exposure_scale
    out = (out - 127.5) * float(contrast) + 127.5
    out += float(brightness)
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(saturation), 0.0, 255.0)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    temp_shift = float(temperature) * 0.8
    bgr = out.astype(np.float32)
    bgr[:, :, 0] = np.clip(bgr[:, :, 0] - temp_shift, 0.0, 255.0)
    bgr[:, :, 2] = np.clip(bgr[:, :, 2] + temp_shift, 0.0, 255.0)
    out = bgr.astype(np.uint8)

    tint_shift = float(tint) * 0.6
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 1] = np.clip(lab[:, :, 1] + tint_shift, 0.0, 255.0)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    if sharpness > 0:
        blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.2, sigmaY=1.2)
        amount = float(sharpness)
        out = cv2.addWeighted(out, 1.0 + amount, blur, -amount, 0)

    return out


def build_ann_model() -> dict:
    classifier = _build_face_pattern_classifier()
    return {"classifier": classifier}


def apply_ann_processing(image: np.ndarray, ann_model: dict, effect_strength: float = 2.4, volume_strength: float = 2.2) -> np.ndarray:
    regions = _find_face_like_regions(image, ann_model["classifier"])
    if not regions:
        regions = _fallback_regions(image, count=6)

    warped = _warp_face_volume(image, regions, volume_strength=volume_strength, effect_strength=effect_strength)
    face_enhanced = _apply_face_like_render(warped, regions, effect_strength=effect_strength)
    return _post_psychedelic_phase(face_enhanced, effect_strength=effect_strength, volume_strength=volume_strength)


def _build_face_pattern_classifier():
    rng = np.random.default_rng(42)
    positive, negative = _build_synthetic_face_dataset(rng, count=1400, side=24)
    x = np.concatenate([positive, negative], axis=0).astype(np.float32)
    y = np.concatenate(
        [
            np.ones((positive.shape[0], 1), dtype=np.float32),
            -np.ones((negative.shape[0], 1), dtype=np.float32),
        ],
        axis=0,
    )

    classifier = cv2.ml.ANN_MLP_create()
    classifier.setLayerSizes(np.array([24 * 24, 96, 40, 1], dtype=np.int32))
    classifier.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
    classifier.setTrainMethod(cv2.ml.ANN_MLP_RPROP)
    classifier.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 700, 1e-6))
    classifier.train(x, cv2.ml.ROW_SAMPLE, y)
    return classifier


def _build_synthetic_face_dataset(rng: np.random.Generator, count: int, side: int) -> tuple[np.ndarray, np.ndarray]:
    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []

    for _ in range(count):
        img = _make_face_template(side, rng=rng).astype(np.float32)
        img += rng.normal(0.0, 0.05, img.shape).astype(np.float32)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        positives.append(np.clip(img, 0.0, 1.0).reshape(-1))

    for _ in range(count):
        img = np.zeros((side, side), dtype=np.float32)
        mode = int(rng.integers(0, 3))
        if mode == 0:
            img = rng.random((side, side), dtype=np.float32)
        elif mode == 1:
            for _ in range(int(rng.integers(3, 8))):
                x1, y1 = int(rng.integers(0, side)), int(rng.integers(0, side))
                x2, y2 = int(rng.integers(0, side)), int(rng.integers(0, side))
                color = float(rng.uniform(0.1, 1.0))
                cv2.line(img, (x1, y1), (x2, y2), color, int(rng.integers(1, 3)))
        else:
            for _ in range(int(rng.integers(2, 7))):
                cx, cy = int(rng.integers(0, side)), int(rng.integers(0, side))
                r = int(rng.integers(2, max(3, side // 3)))
                color = float(rng.uniform(0.1, 1.0))
                cv2.circle(img, (cx, cy), r, color, int(rng.choice([-1, 1, 2])))
        img = cv2.GaussianBlur(img, (3, 3), 0)
        negatives.append(np.clip(img, 0.0, 1.0).reshape(-1))

    return np.asarray(positives, dtype=np.float32), np.asarray(negatives, dtype=np.float32)


def _find_face_like_regions(image: np.ndarray, classifier_model) -> list[tuple[int, int, int, int, float]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    min_w = max(24, int(w * 0.20))
    min_h = max(24, int(h * 0.20))
    scale = min(1.0, 900.0 / max(h, w))
    small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else gray

    small_h, small_w = small.shape[:2]
    min_side = max(18, int(min(small_h, small_w) * 0.20))
    max_side = max(min_side + 2, int(min(small_h, small_w) * 0.95))
    scales = np.geomspace(min_side, max_side, num=8).astype(np.int32)

    candidates: list[tuple[int, int, int, int, float]] = []
    for template in _scaled_face_templates(scales):
        if template.shape[0] >= small.shape[0] or template.shape[1] >= small.shape[1]:
            continue

        response = cv2.matchTemplate(small, template, cv2.TM_CCOEFF_NORMED)
        if response.size == 0:
            continue

        flat = response.reshape(-1)
        top_k = min(24, flat.size)
        top_idx = np.argpartition(flat, -top_k)[-top_k:]
        for idx in top_idx:
            ry, rx = np.unravel_index(int(idx), response.shape)
            score = float(flat[idx])
            if score < 0.11:
                continue
            tw, th = template.shape[1], template.shape[0]
            x = int(round(rx / scale))
            y = int(round(ry / scale))
            rw = int(round(tw / scale))
            rh = int(round(th / scale))
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            rw = max(16, min(rw, w - x))
            rh = max(16, min(rh, h - y))
            if rw < min_w or rh < min_h:
                continue
            candidates.append((x, y, rw, rh, score))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[4], reverse=True)
    selected: list[tuple[int, int, int, int, float]] = []
    for cand in candidates:
        if len(selected) >= 20:
            break
        if all(_iou(cand, s) < 0.45 for s in selected):
            selected.append(cand)

    regions: list[tuple[int, int, int, int, float]] = []
    for x, y, rw, rh, template_score in selected:
        patch = gray[y : y + rh, x : x + rw]
        if patch.size == 0:
            continue
        sample = cv2.resize(patch, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        sample = sample.reshape(1, -1)
        _, pred = classifier_model.predict(sample)
        cls_score = float((pred[0, 0] + 1.0) / 2.0)
        combined = float(np.clip((0.45 * cls_score) + (0.55 * template_score), 0.0, 1.0))
        if combined >= 0.24:
            regions.append((x, y, rw, rh, combined))

    if not regions:
        for x, y, rw, rh, score in selected[:8]:
            regions.append((x, y, rw, rh, float(np.clip(score, 0.22, 0.52))))

    return regions[:12]


def _fallback_regions(image: np.ndarray, count: int) -> list[tuple[int, int, int, int, float]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(gray, maxCorners=max(count * 3, 20), qualityLevel=0.01, minDistance=24)
    h, w = gray.shape[:2]
    min_w = max(24, int(w * 0.20))
    min_h = max(24, int(h * 0.20))
    regions: list[tuple[int, int, int, int, float]] = []

    if points is not None:
        for p in points[: count * 2]:
            px, py = int(p[0][0]), int(p[0][1])
            rw = int(max(min_w, min(w * 0.34, w - 1)))
            rh = int(max(min_h, min(h * 0.36, h - 1)))
            x = max(0, min(px - rw // 2, w - rw))
            y = max(0, min(py - rh // 2, h - rh))
            regions.append((x, y, rw, rh, 0.32))

    if not regions:
        rng = np.random.default_rng(123)
        for _ in range(count):
            rw = int(rng.integers(min_w, max(min_w + 1, int(w * 0.95))))
            rh = int(rng.integers(min_h, max(min_h + 1, int(h * 0.95))))
            x = int(rng.integers(0, max(1, w - rw)))
            y = int(rng.integers(0, max(1, h - rh)))
            regions.append((x, y, rw, rh, 0.28))
    return regions[:count]


def _post_psychedelic_phase(image: np.ndarray, effect_strength: float, volume_strength: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_boost = 1.08 + 0.10 * min(1.0, effect_strength / 3.2)
    val_boost = 1.03 + 0.05 * min(1.0, volume_strength / 3.0)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_boost, 0.0, 255.0)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_boost, 0.0, 255.0)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _scaled_face_templates(scales: np.ndarray) -> list[np.ndarray]:
    templates: list[np.ndarray] = []
    for base in _face_prototypes():
        base_h, base_w = base.shape[:2]
        for side in scales:
            ratio = side / max(base_h, base_w)
            w = max(14, int(round(base_w * ratio)))
            h = max(14, int(round(base_h * ratio)))
            templates.append(cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR))
    return templates


def _face_prototypes() -> list[np.ndarray]:
    global _FACE_PROTOTYPE_CACHE
    if _FACE_PROTOTYPE_CACHE is not None:
        return _FACE_PROTOTYPE_CACHE

    prototypes: list[np.ndarray] = []
    rng = np.random.default_rng(7)
    sizes = (52, 68, 88)
    for side in sizes:
        for _ in range(4):
            base = _make_face_template(side, rng=rng)
            for angle in (-34, -20, -8, 0, 8, 20, 34):
                for sx, sy in ((1.0, 1.0), (1.3, 0.78), (0.78, 1.3), (1.15, 0.9), (0.9, 1.15)):
                    tpl = _affine_face_variant(base, angle=angle, sx=sx, sy=sy)
                    prototypes.append(np.clip(tpl * 255.0, 0, 255).astype(np.uint8))

    _FACE_PROTOTYPE_CACHE = prototypes
    return prototypes


def _make_face_template(side: int, rng: np.random.Generator) -> np.ndarray:
    t = np.zeros((side, side), dtype=np.float32)
    cx = side // 2 + int(rng.integers(-2, 3))
    cy = side // 2 + int(rng.integers(-2, 3))

    head_w = int(rng.uniform(0.58, 0.80) * side)
    head_h = int(rng.uniform(0.66, 0.92) * side)
    cv2.ellipse(t, (cx, cy), (head_w // 2, head_h // 2), 0, 0, 360, 0.35, -1)

    eye_y = int(cy - head_h * rng.uniform(0.08, 0.21))
    eye_dx = int(head_w * rng.uniform(0.14, 0.24))
    eye_r = max(1, int(side * rng.uniform(0.04, 0.08)))
    cv2.circle(t, (cx - eye_dx, eye_y), eye_r, 0.95, -1)
    cv2.circle(t, (cx + eye_dx, eye_y), eye_r, 0.95, -1)

    nose_len = int(head_h * rng.uniform(0.12, 0.24))
    cv2.line(t, (cx, eye_y), (cx + int(rng.integers(-1, 2)), eye_y + nose_len), 0.55, 1)

    mouth_y = int(cy + head_h * rng.uniform(0.16, 0.30))
    mouth_w = int(head_w * rng.uniform(0.20, 0.36))
    cv2.ellipse(t, (cx, mouth_y), (mouth_w // 2, max(1, int(side * 0.06))), 0, 5, 175, 0.86, 1)

    return cv2.GaussianBlur(t, (5, 5), 0)


def _affine_face_variant(template: np.ndarray, angle: float, sx: float, sy: float) -> np.ndarray:
    h, w = template.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    matrix[0, 0] *= sx
    matrix[0, 1] *= sx
    matrix[1, 0] *= sy
    matrix[1, 1] *= sy
    return cv2.warpAffine(template, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def _soft_face_mask(h: int, w: int, blur_scale: float = 0.08) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = w // 2, h // 2
    axes = (max(2, int(0.46 * w)), max(2, int(0.48 * h)))
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1.0, -1)
    sigma_x = max(1.0, blur_scale * w)
    sigma_y = max(1.0, blur_scale * h)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
    return np.clip(mask, 0.0, 1.0)


def _face_volume_map(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    depth = np.zeros((h, w), dtype=np.float32)
    cx = w // 2 + int(rng.integers(-max(1, w // 16), max(2, w // 16)))
    cy = h // 2 + int(rng.integers(-max(1, h // 18), max(2, h // 18)))

    cv2.ellipse(depth, (cx, cy), (max(3, int(0.36 * w)), max(3, int(0.45 * h))), 0, 0, 360, 0.55, -1)
    cv2.ellipse(depth, (cx - int(0.16 * w), int(0.56 * h)), (max(2, int(0.12 * w)), max(2, int(0.19 * h))), 0, 0, 360, 0.28, -1)
    cv2.ellipse(depth, (cx + int(0.16 * w), int(0.56 * h)), (max(2, int(0.12 * w)), max(2, int(0.19 * h))), 0, 0, 360, 0.28, -1)
    cv2.ellipse(depth, (cx, int(0.54 * h)), (max(2, int(0.08 * w)), max(2, int(0.20 * h))), 0, 0, 360, 0.62, -1)

    eye_r = max(1, int(0.09 * min(w, h)))
    cv2.circle(depth, (cx - int(0.18 * w), int(0.40 * h)), eye_r, -0.62, -1)
    cv2.circle(depth, (cx + int(0.18 * w), int(0.40 * h)), eye_r, -0.62, -1)
    cv2.ellipse(depth, (cx, int(0.72 * h)), (max(2, int(0.18 * w)), max(1, int(0.07 * h))), 0, 0, 360, -0.42, -1)

    return cv2.GaussianBlur(depth, (0, 0), sigmaX=max(1.0, 0.05 * w), sigmaY=max(1.0, 0.05 * h))


def _warp_face_volume(
    image: np.ndarray, regions: list[tuple[int, int, int, int, float]], volume_strength: float, effect_strength: float
) -> np.ndarray:
    if not regions:
        return image

    h, w = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x.copy()
    map_y = grid_y.copy()

    for idx, (x, y, rw, rh, score) in enumerate(regions):
        y1, y2 = y, min(y + rh, h)
        x1, x2 = x, min(x + rw, w)
        roi_h = y2 - y1
        roi_w = x2 - x1
        if roi_h <= 0 or roi_w <= 0:
            continue

        seed = int((x * 73856093) ^ (y * 19349663) ^ (idx * 83492791))
        depth = _face_volume_map(roi_h, roi_w, seed=seed)
        grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        face_mask = _soft_face_mask(roi_h, roi_w, blur_scale=0.10)

        xx = np.linspace(-1.0, 1.0, roi_w, dtype=np.float32)[None, :]
        yy = np.linspace(-1.0, 1.0, roi_h, dtype=np.float32)[:, None]
        radial = np.exp(-(xx * xx * 2.8 + yy * yy * 2.2))
        phase = (seed % 97) / 97.0 * np.pi
        twist = np.sin((xx * 2.9 + phase) * np.pi) * np.cos((yy * 2.1 - phase) * np.pi)

        amp = float(np.clip((1.4 + 4.2 * score) * volume_strength * (0.7 + 0.35 * effect_strength), 0.9, 8.5))
        dx = (grad_x * 0.75 + xx * radial * (0.11 * roi_w) + twist * (0.018 * roi_w)) * face_mask
        dy = (grad_y * 0.75 + yy * radial * (0.11 * roi_h) - twist * (0.018 * roi_h)) * face_mask
        map_x[y1:y2, x1:x2] -= dx * amp
        map_y[y1:y2, x1:x2] -= dy * amp

    map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, h - 1).astype(np.float32)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)


def _face_feature_maps(h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]

    head = np.exp(-(xx * xx * 2.1 + yy * yy * 1.8))
    eyes = (
        np.exp(-(((xx + 0.24) ** 2) * 55.0 + ((yy + 0.12) ** 2) * 95.0))
        + np.exp(-(((xx - 0.24) ** 2) * 55.0 + ((yy + 0.12) ** 2) * 95.0))
    )
    nose = np.exp(-((xx * xx) * 170.0 + ((yy - 0.02) ** 2) * 20.0))
    mouth_lips = np.exp(-((xx * xx) * 18.0 + ((yy - 0.46) ** 2) * 70.0))
    mouth_line = np.exp(-((xx * xx) * 34.0 + ((yy - 0.46) ** 2) * 230.0))
    cheek = (
        np.exp(-(((xx + 0.35) ** 2) * 25.0 + ((yy - 0.15) ** 2) * 22.0))
        + np.exp(-(((xx - 0.35) ** 2) * 25.0 + ((yy - 0.15) ** 2) * 22.0))
    )

    relief = 0.34 * head + 0.32 * nose + 0.20 * cheek - 0.34 * eyes + 0.30 * mouth_lips - 0.22 * mouth_line
    relief = cv2.GaussianBlur(relief.astype(np.float32), (0, 0), sigmaX=max(1.0, 0.04 * w), sigmaY=max(1.0, 0.04 * h))

    feature_mask = np.clip(0.9 * head + 0.2 * cheek, 0.0, 1.0).astype(np.float32)
    mouth_region = np.clip(mouth_lips, 0.0, 1.0).astype(np.float32)
    mouth_opening = np.clip(mouth_line, 0.0, 1.0).astype(np.float32)
    return relief.astype(np.float32), feature_mask, mouth_region, mouth_opening


def _apply_face_like_render(
    image: np.ndarray, regions: list[tuple[int, int, int, int, float]], effect_strength: float
) -> np.ndarray:
    out = image.astype(np.float32) / 255.0

    for _, (x, y, rw, rh, score) in enumerate(regions):
        y1, y2 = y, min(y + rh, image.shape[0])
        x1, x2 = x, min(x + rw, image.shape[1])
        roi_h = y2 - y1
        roi_w = x2 - x1
        if roi_h <= 0 or roi_w <= 0:
            continue

        face_mask = _soft_face_mask(roi_h, roi_w, blur_scale=0.12)
        relief, feature_mask, mouth_region, mouth_opening = _face_feature_maps(roi_h, roi_w)
        patch_mask = face_mask * feature_mask

        roi = out[y1:y2, x1:x2].copy()
        gray = cv2.cvtColor((roi * 255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        local_mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(1.0, 0.10 * roi_w), sigmaY=max(1.0, 0.10 * roi_h))

        strength = float(np.clip((0.20 + 0.55 * score) * effect_strength, 0.18, 0.80))
        tone_delta = (relief * strength) * patch_mask
        tone_delta += (0.18 * mouth_region * strength - 0.16 * mouth_opening * strength) * face_mask
        gray_mod = np.clip(gray + tone_delta, 0.0, 1.0)
        gray_mod = np.clip(local_mean + (gray_mod - local_mean) * (1.0 + 0.35 * strength), 0.0, 1.0)

        chroma_boost = 1.0 + (0.04 + 0.12 * score) * strength
        roi *= (gray_mod[:, :, None] / np.maximum(gray[:, :, None], 1e-4))
        roi = np.clip(roi, 0.0, 1.0)
        roi_mean = roi.mean(axis=2, keepdims=True)
        roi = np.clip(roi_mean + (roi - roi_mean) * chroma_boost, 0.0, 1.0)

        mouth_color = (0.06 + 0.10 * score) * strength
        roi[:, :, 2] = np.clip(roi[:, :, 2] + mouth_region * mouth_color * face_mask, 0.0, 1.0)
        roi[:, :, 1] = np.clip(roi[:, :, 1] - mouth_region * mouth_color * 0.45 * face_mask, 0.0, 1.0)
        roi[:, :, 0] = np.clip(roi[:, :, 0] - mouth_region * mouth_color * 0.55 * face_mask, 0.0, 1.0)

        blend = patch_mask[:, :, None]
        out[y1:y2, x1:x2] = out[y1:y2, x1:x2] * (1.0 - blend) + roi * blend

    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _iou(a: tuple[int, int, int, int, float], b: tuple[int, int, int, int, float]) -> float:
    ax, ay, aw, ah, _ = a
    bx, by, bw, bh, _ = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = float((x2 - x1) * (y2 - y1))
    union = float((aw * ah) + (bw * bh) - inter)
    return inter / union if union > 0 else 0.0


def fit_to_preview_size(image: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    if new_w == w and new_h == h:
        return image
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
