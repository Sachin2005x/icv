import cv2
import numpy as np
from src.io_utils import to_gray


def match_roi(query_bgr, ref_bgr, roi=None):
    qg = to_gray(query_bgr)
    rg = to_gray(ref_bgr)
    if roi is None:
        h, w = rg.shape[:2]
        roi = (0, 0, w, h)
    x, y, w, h = roi
    template = rg[y : y + h, x : x + w]
    if template.size == 0:
        return None, 0.0

    best_cc = -1.0
    best_loc = None
    best_scale = 1.0
    for scale in (1.0, 0.95, 0.9, 1.05, 1.1):
        if scale != 1.0:
            qs = cv2.resize(qg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            qs = qg
        if qs.shape[0] < template.shape[0] or qs.shape[1] < template.shape[1]:
            continue
        result = cv2.matchTemplate(qs, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_cc:
            best_cc = float(max_val)
            best_loc = max_loc
            best_scale = scale

    if best_loc is None:
        return None, 0.0
    fx, fy = best_loc
    tx, ty = int(fx / best_scale), int(fy / best_scale)
    tw, th = int(w / best_scale), int(h / best_scale)
    aligned = np.clip(best_cc * 100.0, 0.0, 100.0)
    return ((tx, ty, tw, th), aligned)


def roi_alignment_score(query_bgr, ref_bgr, roi=None):
    _, score = match_roi(query_bgr, ref_bgr, roi)
    return score