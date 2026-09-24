import cv2
import numpy as np


def laplacian_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def edge_overlap(gray_a, gray_b):
    ea = cv2.Canny(gray_a, 60, 160)
    eb = cv2.Canny(gray_b, 60, 160)
    a_count = int(ea.sum() / 255.0)
    b_count = int(eb.sum() / 255.0)
    if a_count == 0 or b_count == 0:
        return 0.0, 0.0
    intersection = int(cv2.bitwise_and(ea, eb).sum() / 255.0)
    denom = min(a_count, b_count)
    return (intersection / float(denom)) if denom else 0.0, min(1.0, a_count / float(b_count))


def sharpness_from_ratio(ratio):
    """Map the query/reference high-frequency energy ratio to a tolerance-aware
    sharpness score. Ratios near 1.0 (matching or sharper) score ~1.0 while
    strongly blurred prints (ratio well below 0.5) collapse toward 0."""
    return 1.0 / (1.0 + np.exp(-10.0 * (ratio - 0.55)))


def print_quality_score(query_bgr, ref_bgr):
    qg = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

    q_lap = laplacian_var(qg)
    r_lap = laplacian_var(rg)
    energy_ratio = q_lap / max(r_lap, 1e-6)
    sharp = sharpness_from_ratio(energy_ratio)

    ov, edge_intensity = edge_overlap(qg, rg)

    score = 100.0 * (0.75 * sharp + 0.25 * edge_intensity)
    return {
        "query_laplacian_var": round(q_lap, 2),
        "ref_laplacian_var": round(r_lap, 2),
        "sharpness_ratio": round(float(sharp), 3),
        "edge_overlap": round(float(ov), 3),
        "edge_intensity": round(float(edge_intensity), 3),
        "score": round(float(np.clip(score, 0.0, 100.0)), 2),
    }