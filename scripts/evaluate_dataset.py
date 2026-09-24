"""Evaluate a folder of genuine and counterfeit samples against one reference.

Reports per-image results plus classification measurements (confusion matrix,
accuracy, precision, recall, F1, and ROC AUC).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_image import evaluate_image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the included sample dataset.")
    parser.add_argument("--reference", type=Path, default=PROJECT_ROOT / "samples" / "reference.jpg")
    parser.add_argument("--genuine-dir", type=Path, default=PROJECT_ROOT / "samples" / "genuine")
    parser.add_argument("--counterfeit-dir", type=Path, default=PROJECT_ROOT / "samples" / "counterfeit")
    parser.add_argument("--threshold", type=float, default=70.0,
                        help="score at or above which an image is counted as predicted genuine (default 70)")
    parser.add_argument("--csv", type=Path, help="optional CSV path for individual results")
    return parser.parse_args()


def evaluate_group(folder: Path, expected: str, reference: Path) -> list[dict]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Sample folder not found: {folder}")
    rows = []
    for image_path in image_files(folder):
        result = evaluate_image(image_path, reference)
        row = {
            "file": str(image_path),
            "expected": expected,
            "score": result.overall_score,
            "verdict": result.verdict,
            "confidence": result.confidence,
        }
        rows.append(row)
        print(f"{expected:11} {image_path.name:20} {result.overall_score:5.1f}  {result.verdict}")
    return rows


def confusion(rows, threshold: float) -> tuple:
    tp = fp = tn = fn = 0
    for row in rows:
        predicted = row["score"] >= threshold
        if row["expected"] == "genuine":
            tp += predicted
            fn += not predicted
        else:
            fp += predicted
            tn += not predicted
    return tp, fp, tn, fn


def classification_report(rows, threshold: float) -> dict:
    tp, fp, tn, fn = confusion(rows, threshold)
    acc = (tp + tn) / len(rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def roc_auc(rows) -> float:
    scores = [row["score"] for row in rows]
    labels = [1 if row["expected"] == "genuine" else 0 for row in rows]
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = {i: rank for rank, i in enumerate(order, start=1)}
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    rank_sum = sum(ranks[i] for i in range(len(scores)) if labels[i])
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def best_threshold(rows) -> tuple:
    best = (0.0, 0.0, None)
    candidates = sorted({row["score"] for row in rows}) + [100.0]
    for th in candidates:
        report = classification_report(rows, th)
        key = (report["accuracy"], report["f1"], th)
        if best[2] is None or key[0] > best[0] or (key[0] == best[0] and key[1] > best[1]):
            best = key
    return best[2], best[0], best[1]


def main() -> int:
    args = parse_args()
    if not args.reference.is_file():
        print(f"Error: Reference image not found: {args.reference}", file=sys.stderr)
        return 2
    try:
        rows = evaluate_group(args.genuine_dir, "genuine", args.reference)
        rows += evaluate_group(args.counterfeit_dir, "counterfeit", args.reference)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"\nEvaluated {len(rows)} images.")
    for expected in ("genuine", "counterfeit"):
        group = [row for row in rows if row["expected"] == expected]
        average = sum(row["score"] for row in group) / len(group) if group else 0.0
        print(f"Average {expected} score: {average:.1f}/100 ({len(group)} images)")

    report = classification_report(rows, args.threshold)
    print(f"\nMeasurements (predicted-genuine threshold: {args.threshold:.1f})")
    print(f"  Confusion matrix (tp/fp/tn/fn): {report['tp']}/{report['fp']}/{report['tn']}/{report['fn']}")
    print(f"  Accuracy : {report['accuracy']:.3f}")
    print(f"  Precision: {report['precision']:.3f}")
    print(f"  Recall   : {report['recall']:.3f}")
    print(f"  F1 score : {report['f1']:.3f}")

    auc = roc_auc(rows)
    opt_th, opt_acc, opt_f1 = best_threshold(rows)
    print(f"  ROC AUC  : {auc:.3f}")
    print(f"  Best threshold by F1: {opt_th:.1f} (accuracy {opt_acc:.3f}, F1 {opt_f1:.3f})")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("file", "expected", "score", "verdict", "confidence"))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV saved: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
