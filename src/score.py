import numpy as np
import cv2
from src.model import AuthResult, verdict_for
from src.features.color_hist import color_suspicion_map

DEFAULT_WEIGHTS = {
    "template": 0.20,
    "color": 0.25,
    "ssim": 0.10,
    "print_quality": 0.45,
}


def fuse_components(components, weights=None):
    weights = weights or DEFAULT_WEIGHTS
    acc = 0.0
    tots = 0.0
    per = {}
    for key, weight in weights.items():
        val = components.get(key)
        if val is None or val.get("score") is None:
            continue
        acc += weight * val["score"]
        tots += weight
        per[key] = val["score"]
    if tots == 0.0:
        return 50.0, per
    return acc / tots, per


def build_heatmap(query_bgr, ref_bgr, components):
    ssim_map = components.get("ssim", {}).get("map")
    color_map = color_suspicion_map(query_bgr, ref_bgr)

    suspicion = np.zeros(query_bgr.shape[:2], dtype=np.float32)
    if ssim_map is not None:
        ssim_resized = cv2.resize(
            ssim_map.astype(np.float32), query_bgr.shape[:2][::-1]
        )
        suspicion += (1.0 - ssim_resized) * 0.6
    suspicion += (color_map.astype(np.float32) / 255.0) * 0.4
    suspicion = np.clip(suspicion * 255.0, 0, 255).astype(np.uint8)

    heat = cv2.applyColorMap(suspicion, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(query_bgr, 0.6, heat, 0.4, 0)
    return blended


def explain(components, overall):
    warnings = []
    if components.get("color", {}).get("score") is not None:
        if components["color"]["score"] < 60:
            warnings.append("print colors deviate from the reference (fading or color-shift)")
    if components.get("ssim", {}).get("score") is not None:
        if components["ssim"]["score"] < 60:
            warnings.append("logo/packaging structure differs from the reference (misalignment or altered graphics)")
    if components.get("print_quality", {}).get("score") is not None:
        pq = components["print_quality"]
        if pq["score"] < 60:
            if pq.get("edge_overlap", 1.0) < 0.7:
                warnings.append("edge/print geometry is inconsistent with the reference")
            if pq.get("sharpness_ratio", 1.0) < 0.7:
                warnings.append("print appears blurred/soft, consistent with low-quality replication")
    if components.get("template", {}).get("score") is not None:
        if components["template"]["score"] < 55:
            warnings.append("stamp/logo region could not be reliably aligned to the reference")
    if not warnings:
        warnings.append("all checked cues are consistent with the reference")
    return warnings


def score_and_explain(query_bgr, ref_bgr, components, weights=None):
    overall, per = fuse_components(components, weights)
    hm = build_heatmap(query_bgr, ref_bgr, components)
    warnings = explain(components, overall)
    spread = per.values()
    confidence = 100.0
    if spread:
        confidence = float(np.clip(100.0 - np.std(list(spread)) * 1.5, 0.0, 100.0))
    return AuthResult(
        overall_score=round(overall, 1),
        verdict=verdict_for(overall),
        confidence=round(confidence, 1),
        components=components,
        warnings=warnings,
        heatmap=hm,
        aligned_query=query_bgr,
        reference=ref_bgr,
        metadata={"weights": weights or DEFAULT_WEIGHTS, "per_component": per},
    )