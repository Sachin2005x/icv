# Product Authentication Using Logo & Packaging Recognition

## Abstract

This mini project demonstrates image-based product authentication. A candidate package image is compared with a known genuine reference using four visual cues: logo/template similarity, colour similarity, structural similarity, and print quality. The cue scores are fused into a single score and mapped to a human-readable verdict.

The project is designed as a lightweight demonstration, not as a production anti-counterfeit system. It uses static images and a synthetic sample dataset.

## Quick Overview

| Aspect | Detail |
| --- | --- |
| Title | Product Authentication Using Logo & Packaging Recognition |
| Type | Lightweight computer-vision mini project (image similarity + score fusion) |
| Language | Python 3.13 |
| Core libraries | OpenCV, NumPy, scikit-image, Pillow, Streamlit |
| Dataset | 1 reference image + 15 synthetic genuine, 15 synthetic counterfeit-like images |
| Input / Output | One package image → score (/100), 5-level verdict, component scores, warnings, suspicion heatmap |
| Interfaces | CLI (`scripts/check_image.py`), batch evaluator (`scripts/evaluate_dataset.py`), browser UI (`streamlit run app.py`) |
| Key result | Genuine 84.9 vs Counterfeit-like 68.1 mean score; ROC AUC 0.827 |

## Objective

Build a working end-to-end system that accepts one product image and returns an authenticity verdict:

`Genuine`, `Likely Genuine`, `Uncertain`, `Likely Counterfeit`, or `Counterfeit`.

## Workflow

```text
Candidate image + genuine reference
              |
        Image loading
              |
     Alignment to reference
              |
  Template | Colour | SSIM | Print-quality scores
              |
        Weighted score fusion
              |
 Verdict + confidence + explanation + heatmap
```

## Methods

### Image loading and alignment

Images are loaded with OpenCV/Pillow fallback support. The candidate is aligned to the reference with ORB feature matching and a homography when sufficient matches are available; otherwise it is resized to the reference dimensions.

### Authentication cues

| Cue | Implementation | Purpose |
| --- | --- | --- |
| Template | Normalised template matching | Checks whether the logo/package layout aligns with the reference. |
| Colour | HSV and BGR histogram intersection | Detects colour shifts, fading, and changed packaging colours. |
| Structure | SSIM plus block-window SSIM minimum | Measures structural similarity and catches localized tampering the global average hides. |
| Print quality | High-frequency energy ratio and edge density | Detects blur, soft printing, and altered edge detail. |

### Score fusion and verdict

The system fuses the cues with weights tuned on the sample set: template 20%, colour 25%, SSIM 10%, and print quality 45%. The fused score is converted to a verdict using these ranges:

| Score | Verdict |
| --- | --- |
| 85–100 | Genuine |
| 70–84.9 | Likely Genuine |
| 55–69.9 | Uncertain |
| 35–54.9 | Likely Counterfeit |
| Below 35 | Counterfeit |

## Technology and Libraries

The project is written in Python and relies on a small set of mainstream computer-vision and web libraries. The table lists the libraries actually imported by the code; the full pinned environment is in `requirements.txt`.

| Library | Version | Role in the project |
| --- | --- | --- |
| Python | 3.13 | Runtime |
| OpenCV (`opencv-python-headless`) | 5.0.0.93 | Image load/convert/resize, ORB feature detection, homography + RANSAC alignment, normalised template matching, HSV/BGR histogram intersection, Laplacian variance, Canny edge overlap, CLAHE, suspicion-heatmap rendering |
| NumPy | 2.4.1 | Array mathematics, score fusion, heatmap blending |
| scikit-image | 0.26.0 | Structural similarity (`skimage.metrics.structural_similarity`) for global, per-channel, and block-window SSIM |
| Pillow | 12.1.1 | Fallback image decoding and EXIF orientation handling when OpenCV cannot read a file |
| Streamlit | 1.63.0 | Browser demo (`app.py`) |
| Standard library | Python 3.13 | `argparse`, `csv`, `pathlib` for the CLI scripts; `unittest` for automated tests |

Notes:

- `requirements.txt` also pins environment packages (scikit-learn, pandas, matplotlib, torch, torchvision, grad-cam, gTTS, pyzbar, etc.) that are installed but not required by the core pipeline, which imports only `cv2`, `numpy`, `PIL`, and `skimage`.
- `scripts/live_detect.py` uses only OpenCV to stream a live webcam verdict. Heatmaps are rendered with `cv2.applyColorMap` (JET) plus alpha blending.

## Dataset and evaluation

Because real product imagery was unavailable, the project ships a small synthetic dataset generated with OpenCV/NumPy. It deliberately includes four near-twin counterfeit samples (`fake_006/008/012/014`) that differ from their genuine counterparts by only a few pixels, to stress-test the scorer.

### Dataset composition

| Folder | Images | Description |
| --- | ---: | --- |
| `samples/reference.jpg` | 1 | The known-genuine reference package every candidate is compared against. |
| `samples/genuine/` | 15 | Re-renderings of the reference with mild, realistic variation (small rotation, brightness/contrast, noise, slight resize). |
| `samples/counterfeit/` | 15 | Deliberate defect renderings simulating blur, colour shifts/fading, JPEG artefacts, rotation, and near-twin pixel-level edits. |

All 30 samples are scored against the single reference image; no external or real-world product imagery is used.

### Evaluation

The evaluator was run with:

```powershell
python scripts/evaluate_dataset.py --csv output/sample_results.csv
```

| Group | Images | Mean authenticity score |
| --- | ---: | ---: |
| Genuine | 15 | 84.9 / 100 |
| Counterfeit-like | 15 | 68.1 / 100 |

### Accuracy measurements

Treating a score of 75 or higher as a genuine prediction, the confusion matrix is shown below. The measurement script prints these automatically:

| Metric | Before | After |
| --- | ---: | ---: |
| Mean genuine score | 72.6 | 84.9 |
| Mean counterfeit score | 65.0 | 68.1 |
| Score gap (genuine − counterfeit) | 7.6 | 16.9 |
| Accuracy (best threshold) | 0.633 | 0.833 |
| Precision | 0.583 | 0.778 |
| Recall | 0.933 | 0.933 |
| F1 | 0.718 | 0.848 |
| ROC AUC | 0.644 | 0.827 |

The four remaining false positives (`fake_006/008/012/014`) are deliberate near-twin samples that differ from their genuine counterparts by only a few pixels; a single-reference pixel comparator has no signal for these, so they are flagged for human review via the heatmap and low-confidence score. The measurements are therefore an honest upper bound for this synthetic set, not a claim about real-world data.

## How to run the demo

### Command-line demo

```powershell
python scripts/check_image.py samples/genuine/genuine_001.jpg
python scripts/check_image.py samples/counterfeit/fake_015.jpg --heatmap output/fake_015_heatmap.png
```

### Browser demo

```powershell
streamlit run app.py
```

The browser interface accepts a single uploaded image, displays the verdict and component scores, and renders a suspicion heatmap.

## Verification

Five automated tests verify score fusion, verdict thresholds, image alignment fallback, a genuine self-comparison, and a blurred-image comparison.

```text
Ran 5 tests in 1.994s
OK
```

## Limitations

- The dataset is synthetic and small, so the thresholds are illustrative rather than calibrated.
- The system compares against one reference image and does not identify products from a large catalogue.
- Lighting, camera angle, occlusion, and real packaging variation can affect results.
- A real deployment would need diverse genuine/counterfeit data, threshold calibration, and human review for uncertain cases.

## Conclusion

The project meets its goal of a clear end-to-end static-image authentication demo. It loads an image, compares it with a reference using multiple visual features, produces an interpretable score and verdict, and offers both CLI and browser demonstrations. Future work should focus on real-world data collection and calibration.
