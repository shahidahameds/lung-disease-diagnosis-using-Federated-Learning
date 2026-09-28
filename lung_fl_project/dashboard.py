import streamlit as st
import pandas as pd
import os
import json
from PIL import Image

# Config
RESULTS_DIR = 'results'

st.set_page_config(page_title="Lung Disease FL Dashboard", layout="wide")

st.title("Blockchain-Audited Federated Ensemble Learning with XAI")
st.markdown("### Experimental Prototype Dashboard")

# 1. Dataset & Config
st.header("1. Dataset & Configuration")
col1, col2 = st.columns([1, 2])

with col1:
    config_path = os.path.join(RESULTS_DIR, 'experiment_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        st.json(config)
    else:
        st.warning("Config not found. Run experiments first.")

with col2:
    summary_path = os.path.join(RESULTS_DIR, 'dataset_summary.csv')
    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
        st.dataframe(df_summary, use_container_width=True)
    
    dist_img_path = os.path.join(RESULTS_DIR, 'class_distribution.png')
    if os.path.exists(dist_img_path):
        st.image(Image.open(dist_img_path), caption="Class Distribution", use_container_width=True)

# 2. Diagnostic Performance
st.header("2. Diagnostic Performance")
diag_path = os.path.join(RESULTS_DIR, 'diagnostic_results.csv')
if os.path.exists(diag_path):
    df_diag = pd.read_csv(diag_path)
    st.dataframe(df_diag.style.format(precision=4), use_container_width=True)
    
    # Bar chart for Accuracy and AUC
    df_chart = df_diag[['Method', 'Accuracy', 'AUC']].set_index('Method')
    st.bar_chart(df_chart)

conv_img_path = os.path.join(RESULTS_DIR, 'accuracy_convergence.png')
if os.path.exists(conv_img_path):
    st.image(Image.open(conv_img_path), caption="Accuracy Convergence", use_container_width=True)

# 3. Privacy, Security, & Communication
st.header("3. Privacy, Security & Overhead")
col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    st.subheader("Gradient Reconstruction")
    recon_path = os.path.join(RESULTS_DIR, 'reconstruction_results.csv')
    if os.path.exists(recon_path):
        df_recon = pd.read_csv(recon_path)
        st.dataframe(df_recon.style.format(precision=4))
    else:
        st.info("No reconstruction results.")

with col_p2:
    st.subheader("Hash-Chain Audit")
    audit_path = os.path.join(RESULTS_DIR, 'audit_results.csv')
    if os.path.exists(audit_path):
        df_audit = pd.read_csv(audit_path)
        st.dataframe(df_audit)
    else:
        st.info("No audit results.")

with col_p3:
    st.subheader("Communication Overhead")
    comm_path = os.path.join(RESULTS_DIR, 'communication_results.csv')
    if os.path.exists(comm_path):
        df_comm = pd.read_csv(comm_path)
        st.dataframe(df_comm.style.format(precision=2))
    else:
        st.info("No communication results.")

# 4. Explainable AI
st.header("4. Explainable AI (XAI)")
xai_img_path = os.path.join(RESULTS_DIR, 'xai_results.png')
if os.path.exists(xai_img_path):
    st.image(Image.open(xai_img_path), caption="Grad-CAM and SHAP Explanations", use_container_width=True)
else:
    st.info("XAI results not found.")

st.markdown("---")
st.markdown("*Research Prototype Dashboard generated automatically.*")
