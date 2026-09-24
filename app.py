"""A small browser UI for the product-authentication demonstration."""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from scripts.check_image import evaluate_images
from src.io_utils import load_image, to_rgb


PROJECT_ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = PROJECT_ROOT / "samples" / "reference.jpg"


st.set_page_config(page_title="Product Authentication", page_icon="🔎")
st.title("Product Authentication")
st.caption("Upload a package image to compare its logo, colors, structure, and print quality with the genuine reference.")

uploaded = st.file_uploader("Product image", type=["jpg", "jpeg", "png", "bmp", "webp"])
if uploaded is None:
    st.info("Choose an image to begin. The bundled reference image is used automatically.")
    st.stop()

data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
query = cv2.imdecode(data, cv2.IMREAD_COLOR)
if query is None:
    st.error("This file could not be read as an image.")
    st.stop()

try:
    reference = load_image(str(REFERENCE_PATH))
    result = evaluate_images(query, reference)
except (OSError, ValueError, cv2.error) as error:
    st.error(f"Could not evaluate the image: {error}")
    st.stop()

st.subheader(result.verdict)
first, second = st.columns(2)
first.metric("Authenticity score", f"{result.overall_score:.1f}/100")
second.metric("Confidence", f"{result.confidence:.1f}%")

st.subheader("Component scores")
scores = {name.replace("_", " ").title(): values.get("score") for name, values in result.components.items()}
st.bar_chart(scores, horizontal=True)

st.subheader("What was checked")
for warning in result.warnings:
    st.write(f"- {warning}")

st.subheader("Visual comparison")
left, right, heat = st.columns(3)
left.image(to_rgb(query), caption="Uploaded image")
right.image(to_rgb(result.reference), caption="Genuine reference")
heat.image(to_rgb(result.heatmap), caption="Suspicion heatmap")
