import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn


def _gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr


def _crop_to_match(a, b):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def compare_ssim(gray_a, gray_b):
    ga, gb = _crop_to_match(_gray(gray_a), _gray(gray_b))
    win = 7
    if ga.shape[0] < win or ga.shape[1] < win:
        return None, None
    mean, full = ssim_fn(
        ga, gb, data_range=255.0, win_size=win, full=True, gaussian_weights=True
    )
    return float(mean), np.clip(full * 0.5 + 0.5, 0.0, 1.0)


def block_ssim_minimum(gray_a, gray_b, block=48):
    """Small-region SSIM minimum: flags localized tampering that the global
    average can hide (e.g., a modified logo or altered detail in one area)."""
    ga, gb = _crop_to_match(_gray(gray_a), _gray(gray_b))
    h, w = ga.shape[:2]
    mins = []
    for y in range(0, h, block):
        for x in range(0, w, block):
            a = ga[y : y + block, x : x + block]
            b = gb[y : y + block, x : x + block]
            if a.shape != b.shape or a.shape[0] < 4 or a.shape[1] < 4:
                continue
            try:
                mins.append(ssim_fn(a, b, data_range=255.0, win_size=7))
            except ValueError:
                continue
    return float(np.min(mins)) if mins else 1.0


def structural_score(query_bgr, ref_bgr):
    gray_mean, gray_map = compare_ssim(query_bgr, ref_bgr)
    if gray_mean is None:
        return {"gray_ssim": None, "score": None, "map": None}

    channel_means = []
    for c in range(3):
        m, _ = compare_ssim(query_bgr[:, :, c], ref_bgr[:, :, c])
        if m is not None:
            channel_means.append(m)
    color_mean = float(np.mean(channel_means)) if channel_means else None

    block_min = block_ssim_minimum(query_bgr, ref_bgr)
    base = color_mean if color_mean is not None else gray_mean
    score = 0.30 * base + 0.70 * block_min

    return {
        "gray_ssim": round(gray_mean, 4),
        "color_ssim": round(color_mean, 4) if color_mean is not None else None,
        "block_ssim_min": round(block_min, 4),
        "score": round(float(np.clip(score, 0.0, 1.0) * 100.0), 2),
        "map": gray_map,
    }