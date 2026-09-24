# Product Authentication Using Logo & Packaging Recognition

A lightweight mini project that compares one product-package image with a known genuine reference. It combines template matching, colour-histogram similarity, structural similarity (SSIM), and print-quality checks, then reports an authenticity verdict.

## Setup

Use Python 3.10+ and install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

The included synthetic images live in `samples/`. `samples/reference.jpg` is the genuine reference.

## Check one image

```powershell
python scripts/check_image.py samples/genuine/genuine_001.jpg
python scripts/check_image.py samples/counterfeit/fake_015.jpg --heatmap output/result.png
```

Use `--reference path/to/reference.jpg` to compare against a different genuine package. The result includes an overall score, one of five verdicts, component scores, and plain-language notes. The optional heatmap highlights areas that differ from the reference.

## Evaluate the sample set

```powershell
python scripts/evaluate_dataset.py
python scripts/evaluate_dataset.py --csv output/sample_results.csv
```

This prints a row per image, the average score for the genuine and counterfeit groups, and classification measurements (confusion matrix, accuracy, precision, recall, F1, ROC AUC). The recommended decision threshold is printed automatically. The synthetic dataset is for demonstrating the workflow only; scores and thresholds are not calibrated for real-world authentication.

## Browser demo

```powershell
streamlit run app.py
```

Upload one image. The app compares it with the bundled reference and shows the verdict, component scores, explanatory notes, and a suspicion heatmap.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Project layout

```text
src/       image loading, preprocessing, feature scoring, score fusion
scripts/   command-line image check and dataset evaluation
samples/   generated reference, genuine, and counterfeit examples
tests/     automated checks for core pipeline behavior
output/    optional generated heatmaps and CSV files
```
