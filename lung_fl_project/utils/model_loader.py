import json
import os

import streamlit as st
import torch

from config import CHECKPOINTS_DIR, DEVICE, NUM_CLASSES
from models.ensemble import DiagnosticEnsemble


@st.cache_resource(show_spinner="Loading the proposed ensemble checkpoint...")
def load_proposed_model():
    device = torch.device("cuda" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu")
    model = DiagnosticEnsemble(num_classes=NUM_CLASSES, freeze_backbone=True).to(device)
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, "proposed", "latest.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Proposed checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    return model, device, checkpoint


def load_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)
