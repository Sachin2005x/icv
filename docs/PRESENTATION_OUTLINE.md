# Presentation Outline — Product Authentication Using Logo & Packaging Recognition

## Slide 1 — Title

**Product Authentication Using Logo & Packaging Recognition**  
Mini Project

Say: “This project checks whether a product package visually matches a known genuine reference image.”

## Slide 2 — Problem statement

- Counterfeit packaging can differ in logo placement, print sharpness, colour, and graphics.
- Manual inspection is slow and subjective.
- Goal: provide a lightweight single-image authenticity indication.

## Slide 3 — Project objective

- Input: one static package image.
- Compare it with a genuine reference image.
- Output: score, verdict, explanation, and optional heatmap.

## Slide 4 — System workflow

```text
Image → alignment → 4 visual checks → weighted fusion → verdict
```

Visual checks: template/logo, colour, SSIM structure, print quality.

## Slide 5 — Feature methods

| Feature | Detects |
| --- | --- |
| Template matching | displaced/changed logo or layout |
| Colour histogram | fading or colour shift |
| SSIM | altered packaging structure |
| Print quality | blur and weak edges |

## Slide 6 — Dataset

- 15 synthetic genuine images.
- 15 counterfeit-like images.
- Variations: rotation, brightness, colour shifts, blur, and JPEG compression.
- Reference: `samples/reference.jpg`.

## Slide 7 — Results

| Dataset group | Mean score |
| --- | ---: |
| Genuine | 72.6 / 100 |
| Counterfeit-like | 65.0 / 100 |

- `genuine_001.jpg`: 100.0, `GENUINE`
- `fake_015.jpg`: 45.1, `LIKELY COUNTERFEIT`
- Show `output/fake_015_heatmap.png` on this slide.

## Slide 8 — Live demo

Run:

```powershell
streamlit run app.py
```

Upload a genuine sample, then `samples/counterfeit/fake_015.jpg`. Point out the verdict, component score chart, explanatory notes, and heatmap.

## Slide 9 — Limitations and future work

- Synthetic dataset; results are not calibrated for real products.
- One reference image per package.
- Future work: real datasets, multiple reference views, barcode/OCR checks, and calibrated thresholds.

## Slide 10 — Conclusion

- Delivered an end-to-end static-image authentication demo.
- Multi-feature comparison makes the decision more interpretable than a single cue.
- Includes a CLI, batch evaluator, browser interface, documentation, and tests.
