import cv2
import numpy as np


def align_to_reference(query_bgr, ref_bgr, max_dim=1600):
    q = _downscale(query_bgr, max_dim)
    r = _downscale(ref_bgr, max_dim)
    if q.shape[:2] == r.shape[:2]:
        return q, np.eye(3)

    H, mask = _homography_orbit(query_bgr, ref_bgr)
    if H is None:
        return _fallback_align(q, r)

    h, w = r.shape[:2]
    aligned = cv2.warpPerspective(query_bgr, H, (w, h))
    return aligned, H


def _downscale(bgr, max_dim):
    h, w = bgr.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale >= 1.0:
        return bgr
    return cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _fallback_align(query_bgr, ref_bgr):
    return cv2.resize(
        query_bgr,
        (ref_bgr.shape[1], ref_bgr.shape[0]),
        interpolation=cv2.INTER_AREA,
    ), np.eye(3)


def _homography_orbit(query_bgr, ref_bgr):
    qg = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=4000)
    kq, dq = orb.detectAndCompute(qg, None)
    kr, dr = orb.detectAndCompute(rg, None)
    if dq is None or dr is None or len(kq) < 8 or len(kr) < 8:
        return None, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(dq, dr)
    matches = sorted(matches, key=lambda m: m.distance)[:120]
    if len(matches) < 8:
        return None, None
    src = np.float32([kq[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kr[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(status.sum()) if status is not None else 0
    if H is None or inliers < 8 or inliers / float(len(matches)) < 0.3:
        return None, None
    return H, inliers


def clahe_gray(gray):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)