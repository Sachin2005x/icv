"""Check one product image against a known genuine reference image.

Example:
    python scripts/check_image.py samples/genuine/genuine_001.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Allow this script to be run directly from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.features.color_hist import compare_color
from src.features.print_quality import print_quality_score
from src.features.structural import compare_ssim
from src.features.template import roi_alignment_score
from src.io_utils import load_image, save_image, to_gray
from src.preprocessing import align_to_reference
from src.score import score_and_explain


def evaluate_images(query, reference):
    """Align, score, and explain already-loaded BGR images."""
    aligned_query, _ = align_to_reference(query, reference)

    ssim_value, ssim_map = compare_ssim(to_gray(aligned_query), to_gray(reference))
    components = {
        "template": {"score": roi_alignment_score(aligned_query, reference)},
        "color": compare_color(aligned_query, reference),
        "ssim": {
            "score": None if ssim_value is None else float(np.clip(ssim_value * 100, 0, 100)),
            "value": ssim_value,
            "map": ssim_map,
        },
        "print_quality": print_quality_score(aligned_query, reference),
    }
    return score_and_explain(aligned_query, reference, components)


def evaluate_image(image_path: Path, reference_path: Path):
    """Load, align, score, and explain one candidate image."""
    return evaluate_images(load_image(str(image_path)), load_image(str(reference_path)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authenticate one product image against a reference package image."
    )
    parser.add_argument("image", type=Path, help="path to the image to check")
    parser.add_argument(
        "--reference",
        type=Path,
        default=PROJECT_ROOT / "samples" / "reference.jpg",
        help="path to the known genuine reference image",
    )
    parser.add_argument(
        "--heatmap",
        type=Path,
        help="optional path for a PNG highlighting suspicious regions",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path, label in ((args.image, "Image"), (args.reference, "Reference image")):
        if not path.is_file():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            return 2

    try:
        result = evaluate_image(args.image, args.reference)
    except (OSError, ValueError, cv2.error) as error:
        print(f"Error: could not evaluate image: {error}", file=sys.stderr)
        return 1

    print(f"Image: {args.image}")
    print(f"Reference: {args.reference}")
    print(f"Overall score: {result.overall_score:.1f}/100")
    print(f"Verdict: {result.verdict}")
    print(f"Confidence: {result.confidence:.1f}%")
    print("Component scores:")
    for name in ("template", "color", "ssim", "print_quality"):
        score = result.components[name].get("score")
        display = "unavailable" if score is None else f"{score:.1f}/100"
        print(f"  - {name.replace('_', ' ').title()}: {display}")
    print("Notes:")
    for warning in result.warnings:
        print(f"  - {warning}")

    if args.heatmap:
        args.heatmap.parent.mkdir(parents=True, exist_ok=True)
        save_image(str(args.heatmap), result.heatmap)
        print(f"Heatmap saved: {args.heatmap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
