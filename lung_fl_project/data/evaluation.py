import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def calculate_metrics(labels, predictions, probabilities, num_classes):
    """Calculate weighted metrics and report why AUC is unavailable when invalid."""
    metrics = {
        "Accuracy": accuracy_score(labels, predictions),
        "Precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "Recall": recall_score(labels, predictions, average="weighted", zero_division=0),
        "F1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "AUC": np.nan,
    }
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    unique_labels = np.unique(labels)
    auc_reason = None
    if num_classes < 2:
        auc_reason = "AUC requires at least two detected classes."
    elif len(unique_labels) < num_classes:
        auc_reason = f"AUC requires labels for all {num_classes} classes; observed {len(unique_labels)}."
    elif probabilities.ndim == 1 and num_classes == 2 and len(probabilities) == len(labels):
        try:
            metrics["AUC"] = roc_auc_score(labels, probabilities)
        except ValueError as error:
            auc_reason = f"Binary AUC failed: {error}"
    elif probabilities.ndim == 2 and probabilities.shape == (len(labels), num_classes):
        try:
            if num_classes == 2:
                auc = roc_auc_score(labels, probabilities[:, 1])
            else:
                auc = roc_auc_score(labels, probabilities, multi_class="ovr", average="weighted")
            metrics["AUC"] = auc
        except ValueError as error:
            auc_reason = f"AUC failed: {error}"
    else:
        auc_reason = f"AUC probabilities have invalid shape {probabilities.shape} for {num_classes} classes."
    if auc_reason:
        print(f"AUC not calculated: {auc_reason}")
    return metrics


def evaluate_model(model, loader, device, num_classes):
    model.eval()
    predictions, labels, probabilities = [], [], []
    import torch
    with torch.no_grad():
        for inputs, batch_labels in loader:
            output = model(inputs.to(device))
            predictions.extend(torch.argmax(output, dim=1).cpu().numpy())
            labels.extend(batch_labels.numpy())
            probabilities.extend(output.cpu().numpy())
    return calculate_metrics(labels, predictions, probabilities, num_classes)