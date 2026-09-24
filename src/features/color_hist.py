import cv2
import numpy as np


def compare_color(query_bgr, ref_bgr):
    qi = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2HSV)
    ri = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV)

    qh = cv2.calcHist([qi], [0, 1], None, [50, 60], [0, 180, 0, 256])
    rh = cv2.calcHist([ri], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(qh, qh)
    cv2.normalize(rh, rh)
    hsv_inter = cv2.compareHist(qh, rh, cv2.HISTCMP_INTERSECT)

    qb = cv2.calcHist([query_bgr], [0, 1, 2], None, [32, 32, 32], [0, 256] * 3)
    rb = cv2.calcHist([ref_bgr], [0, 1, 2], None, [32, 32, 32], [0, 256] * 3)
    cv2.normalize(qb, qb)
    cv2.normalize(rb, rb)
    bgr_inter = cv2.compareHist(qb, rb, cv2.HISTCMP_INTERSECT)

    score = 0.5 * hsv_inter + 0.5 * bgr_inter
    return {
        "hsv_inter": float(np.clip(hsv_inter, 0.0, 1.0)),
        "bgr_inter": float(np.clip(bgr_inter, 0.0, 1.0)),
        "score": float(np.clip(score * 100.0, 0.0, 100.0)),
    }


def color_suspicion_map(query_bgr, ref_bgr):
    ql = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2LAB)
    rl = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB)
    if ql.shape[:2] != rl.shape[:2]:
        rl = cv2.resize(rl, ql.shape[:2][::-1])
    diff = cv2.absdiff(ql, rl).astype(np.float32) / 255.0
    dist = np.mean(diff, axis=2)
    return np.uint8(np.clip(dist * 255.0, 0, 255))