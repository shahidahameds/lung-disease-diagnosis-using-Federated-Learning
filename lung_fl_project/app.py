import json
import os
import io
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from config import RESULTS_DIR
from utils.inference import individual_predictions, predict, prepare_image
from utils.model_loader import load_json, load_proposed_model
from utils.visualization import custom_cnn_shap, gradcam_custom, overlay, shap_to_heatmap

st.set_page_config(page_title="FedLung-XAI Research Console", page_icon="🫁", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { --ink:#17232d; --muted:#60717d; --accent:#0b7285; --soft:#e8f3f5; }
.block-container { max-width: 1280px; padding-top: 2rem; }
.hero { padding: 2rem 2.2rem; border-radius: 14px; background: linear-gradient(120deg,#102a43,#0b7285); color: white; margin-bottom: 1.5rem; }
.hero h1 { margin: 0; font-size: 2.2rem; }
.hero p { color: #d7eef1; margin-bottom: 0; }
.disclaimer { border-left: 4px solid #f0ad4e; padding: .8rem 1rem; background: #fff8e8; color: #5a461b; }
</style>
""", unsafe_allow_html=True)

ROOT = os.path.dirname(os.path.abspath(__file__))

def path(name):
    return os.path.join(ROOT, "results", name)


def read_csv(name):
    file_path = path(name)
    return pd.read_csv(file_path) if os.path.exists(file_path) else pd.DataFrame()


def dataset_ground_truth(filename):
    for split in ("train", "val", "test"):
        for class_name in ("NORMAL", "PNEUMONIA"):
            candidate = os.path.join(ROOT, "data", split, class_name, filename)
            if os.path.isfile(candidate):
                return class_name
    return None


def confidence_label(confidence):
    if confidence < 0.60:
        return "Low confidence"
    if confidence <= 0.80:
        return "Moderate confidence"
    return "High confidence"


@st.cache_data(show_spinner=False, max_entries=32)
def cached_custom_shap(image_bytes, target_class, checkpoint_version):
    model, device, _ = load_proposed_model()
    image = Image.open(io.BytesIO(image_bytes))
    _, tensor = prepare_image(image)
    return custom_cnn_shap(model, tensor, device, target_class)


def header(title, subtitle):
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


def page_diagnosis():
    header("Diagnosis", "Real four-model ensemble inference from the saved Proposed Framework checkpoint")
    st.markdown("<div class='disclaimer'>Research prototype only. This output is not a clinical diagnosis or clinical validation.</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
    if not uploaded:
        st.info("Upload an image to run local inference.")
        return
    try:
        image = Image.open(uploaded)
        display_image, tensor = prepare_image(image)
        model, device, checkpoint = load_proposed_model()
        mapping = load_json(path("class_mapping.json"))
        result = predict(model, tensor, device, mapping)
        individual = individual_predictions(model, tensor, device, mapping)
        known_label = dataset_ground_truth(uploaded.name)
        left, right = st.columns([1, 1])
        with left:
            st.image(display_image, caption="Uploaded image", use_container_width=True)
        with right:
            st.metric("Ensemble Prediction", result["label"])
            confidence = result["probabilities"][result["label"]]
            st.metric("Model Confidence", f"{confidence:.1%}")
            st.caption(confidence_label(confidence))
            if known_label:
                st.write(f"Ground truth: **{known_label}**")
                st.write("Result: **Correct**" if known_label == result["label"] else "Result: **Misclassified**")
            else:
                st.caption("Ground truth is not available for this arbitrary upload.")
            st.bar_chart(pd.Series(result["probabilities"], name="Probability"))
        st.subheader("Individual model predictions")
        st.dataframe(pd.DataFrame(individual).T, use_container_width=True)
    except Exception as error:
        st.error(f"The upload could not be processed as an X-ray image: {error}")


def page_xai():
    header("Explainable AI", "Actual Grad-CAM and SHAP explanations generated from the saved ensemble")
    uploaded = st.file_uploader("Upload an image for explanation", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], key="xai")
    if not uploaded:
        st.info("Upload an image to generate explanations.")
        return
    try:
        image = Image.open(uploaded)
        display_image, tensor = prepare_image(image)
        model, device, checkpoint = load_proposed_model()
        mapping = load_json(path("class_mapping.json"))
        result = predict(model, tensor, device, mapping)
        target = result["class_index"]
        gradcam, shape_log = gradcam_custom(model, tensor, device, target, (display_image.height, display_image.width))
        st.caption(f"XAI shapes: original={shape_log['original_shape']}, model_input={shape_log['model_input_shape']}, CAM={shape_log['cam_shape']}, resized_CAM={shape_log['resized_cam_shape']}")
        columns = st.columns(3)
        columns[0].image(display_image, caption="Original", use_container_width=True)
        columns[1].image(overlay(display_image, gradcam), caption="Grad-CAM: Custom CNN convolutional evidence", use_container_width=True)
        try:
            checkpoint_path = os.path.join(ROOT, "checkpoints", "proposed", "latest.pt")
            checkpoint_version = f"{os.path.getmtime(checkpoint_path)}:{checkpoint.get('current_round', 'unknown')}"
            with st.spinner("Generating SHAP explanation..."):
                started = time.perf_counter()
                selected_shap, shap_metadata = cached_custom_shap(uploaded.getvalue(), target, checkpoint_version)
                heat, shap_shape_log = shap_to_heatmap(selected_shap, (display_image.height, display_image.width))
                elapsed = time.perf_counter() - started
            columns[2].image(overlay(display_image, heat), caption="SHAP: Custom CNN branch explanation", use_container_width=True)
            st.caption(f"SHAP: Pixel-level contribution to predicted class • {elapsed:.2f}s")
            st.caption(f"SHAP shapes: type={shap_metadata['shap_values_type']}, raw={shap_metadata['shap_values_shape']}, selected={shap_shape_log['shap_shape']}, resized={shap_shape_log['resized_shap_shape']}, overlay={(display_image.height, display_image.width, 3)}")
            st.info("Red/warm regions indicate features contributing positively toward the selected prediction, while cool regions indicate lower or opposing contribution. These regions are not clinically diagnostic.")
        except Exception as error:
            with columns[2].expander("Developer details", expanded=False):
                st.code(f"{type(error).__name__}: {error}")
            columns[2].warning("SHAP explanation unavailable for this image/model configuration")
    except Exception as error:
        st.error(f"Explanation generation failed: {error}")


def page_federated():
    header("Federated Learning", "Topology and observed convergence from the controlled real-data experiments")
    config = load_json(path("experiment_config.json"))
    a, b, c = st.columns(3)
    a.metric("Hospitals", config.get("num_hospitals", "NOT AVAILABLE"))
    b.metric("Departments / hospital", config.get("departments_per_hospital", "NOT AVAILABLE"))
    c.metric("Virtual clients", config.get("total_clients", "NOT AVAILABLE"))
    st.code("Hospital 1..4\n  Department 1 + Department 2\n        -> hospital aggregation\n        -> central aggregation", language="text")
    convergence = read_csv("convergence.csv")
    if not convergence.empty:
        selected = st.multiselect("Experiments", sorted(convergence["Experiment"].unique()), default=sorted(convergence["Experiment"].unique()))
        frame = convergence[convergence["Experiment"].isin(selected)]
        st.line_chart(frame.pivot(index="Round", columns="Experiment", values="Validation Accuracy"))
        st.dataframe(frame, use_container_width=True)


def page_privacy():
    header("Privacy & Security", "Configured protections and explicit accounting limitations")
    privacy = read_csv("privacy_results.csv")
    if not privacy.empty:
        st.dataframe(privacy, use_container_width=True)
    st.warning("Formal epsilon and delta guarantees are NOT AVAILABLE because no privacy accountant was implemented.")
    reconstruction = read_csv("reconstruction_results.csv")
    st.subheader("Gradient reconstruction")
    st.dataframe(reconstruction, use_container_width=True)


def page_audit():
    header("Audit & Anomaly Detection", "Observed hash-chain integrity and hospital update screening")
    audit = read_csv("audit_results.csv")
    anomaly = read_csv("anomaly_results.csv")
    left, right = st.columns(2)
    with left:
        st.subheader("Hash-chain audit")
        st.dataframe(audit, use_container_width=True)
    with right:
        st.subheader("Hospital updates")
        st.dataframe(anomaly, use_container_width=True)
    st.info("Anomaly precision, recall, F1, and accuracy are NOT AVAILABLE because no controlled anomaly ground truth was recorded.")


def page_performance():
    header("Model Performance", "Artifact-backed comparison of the five completed experiments")
    summary = read_csv("experiment_summary.csv")
    if summary.empty:
        st.warning("No consolidated experiment summary is available.")
        return
    st.dataframe(summary, use_container_width=True)
    metric = st.selectbox("Metric", ["Accuracy", "Precision", "Recall", "F1", "AUC", "runtime_seconds"])
    st.bar_chart(summary.set_index("Experiment")[[metric]])
    convergence = read_csv("convergence.csv")
    if not convergence.empty:
        st.subheader("Validation convergence")
        st.line_chart(convergence.pivot(index="Round", columns="Experiment", values="Validation Accuracy"))


def page_about():
    header("About", "FedLung-XAI research prototype")
    st.write("This application loads the saved Phase 3 experiment artifacts and checkpoints for a reproducible research demonstration.")
    st.write("The verified dataset currently contains NORMAL and PNEUMONIA only. No COVID-19 or tuberculosis images were used.")
    st.markdown("**Medical disclaimer:** This software is for research and education. It is not a medical device, does not provide clinical validation, and must not be used for patient care decisions.")


pages = {"Diagnosis": page_diagnosis, "Explainable AI": page_xai, "Federated Learning": page_federated, "Privacy & Security": page_privacy, "Audit & Anomaly Detection": page_audit, "Model Performance": page_performance, "About": page_about}
st.sidebar.title("FedLung-XAI")
st.sidebar.caption("Research console")
selection = st.sidebar.radio("Module", list(pages))
st.sidebar.divider()
st.sidebar.caption("Real artifacts only • CPU-friendly cached inference")
pages[selection]()
