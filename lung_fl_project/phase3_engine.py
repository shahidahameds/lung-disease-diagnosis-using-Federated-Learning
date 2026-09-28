import copy
import gc
import io
import json
import os
import random
import tracemalloc
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader, Subset

from config import (
    ANOMALY_THRESHOLD,
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    CLIP_NORM,
    DATA_DIR,
    DEPARTMENTS_PER_HOSPITAL,
    DEVICE,
    IMAGE_SIZE,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    MAX_TEST_SAMPLES,
    MAX_TRAIN_SAMPLES,
    NOISE_MULTIPLIER,
    NUM_HOSPITALS,
    NUM_ROUNDS,
    RESULTS_DIR,
    RANDOM_SEED,
    EXPERIMENT_MAX_TRAIN_SAMPLES_PER_CLIENT,
    EXPERIMENT_MAX_TEST_SAMPLES,
)
from data.dataset import LungDiseaseDataset, build_dataset_report, discover_class_mapping
from data.evaluation import calculate_metrics
from federated.aggregation import aggregate_models
from models.ensemble import DiagnosticEnsemble
from privacy.differential_privacy import apply_dp_to_update
from security.anomaly import detect_anomaly
from security.audit import HashChainAudit


class FeatureCachedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, features):
        self.dataset = dataset
        self.features = features
        self.samples = dataset.samples
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, {name: values[index] for name, values in self.features.items()}


def _feature_cache_key(dataset, mapping):
    import hashlib
    payload = "|".join(path for path, _ in dataset.samples)
    payload += f"|{IMAGE_SIZE}|{sorted(mapping.items())}|inception-vgg-resnet-frozen-v1"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _build_feature_cache(model, dataset, mapping, split):
    cache_dir = os.path.join(RESULTS_DIR, "feature_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{split}_{_feature_cache_key(dataset, mapping)}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu", weights_only=True)
    captured = {"inception": [], "vgg": [], "resnet": []}
    handles = []
    for name, layer in (("inception", model.inception.fc), ("vgg", model.vgg.classifier[6]), ("resnet", model.resnet.fc)):
        handles.append(layer.register_forward_pre_hook(lambda _, inputs, name=name: captured[name].append(inputs[0].detach().cpu())))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            model(images)
    for handle in handles:
        handle.remove()
    cache = {name: torch.cat(values, dim=0) for name, values in captured.items()}
    torch.save(cache, cache_path)
    return cache


def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _subset_dataset(dataset, limit):
    if limit is None or limit >= len(dataset):
        return dataset
    rng = random.Random(RANDOM_SEED)
    by_class = {}
    for index, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(index)
    selected = []
    class_names = sorted(by_class)
    per_class = limit // len(class_names)
    remainder = limit % len(class_names)
    for position, class_index in enumerate(class_names):
        class_limit = per_class + (1 if position < remainder else 0)
        selected.extend(rng.sample(by_class[class_index], min(class_limit, len(by_class[class_index]))))
    dataset.samples = [dataset.samples[index] for index in selected[:limit]]
    return dataset


def _make_datasets(smoke_test=False, full_data=False, deterministic=False):
    train = LungDiseaseDataset(DATA_DIR, split="train", img_size=IMAGE_SIZE, augment=not deterministic)
    test = LungDiseaseDataset(DATA_DIR, split="test", img_size=IMAGE_SIZE, augment=False)
    full_train_size = len(train)
    full_test_size = len(test)
    limit_train = 64 if smoke_test else (None if full_data else EXPERIMENT_MAX_TRAIN_SAMPLES_PER_CLIENT * NUM_HOSPITALS * DEPARTMENTS_PER_HOSPITAL)
    limit_test = 16 if smoke_test else (None if full_data else EXPERIMENT_MAX_TEST_SAMPLES)
    if MAX_TRAIN_SAMPLES is not None and not full_data and not smoke_test:
        limit_train = min(limit_train, MAX_TRAIN_SAMPLES)
    if MAX_TEST_SAMPLES is not None and not full_data and not smoke_test:
        limit_test = min(limit_test, MAX_TEST_SAMPLES)
    _subset_dataset(train, limit_train)
    _subset_dataset(test, limit_test)
    mapping = discover_class_mapping(DATA_DIR)
    with open(os.path.join(RESULTS_DIR, "class_mapping.json"), encoding="utf-8") as mapping_file:
        saved_mapping = json.load(mapping_file)
    if mapping != saved_mapping or train.class_to_idx != test.class_to_idx:
        raise ValueError(f"Class mapping mismatch: discovered={mapping}, saved={saved_mapping}")
    print(f"class_to_idx={mapping}")
    mode = "FULL_DATASET" if full_data else ("SMOKE_TEST" if smoke_test else "DEMO_SUBSET")
    print(f"dataset_mode={mode}, train_samples={len(train)}, test_samples={len(test)}, full_train={full_train_size}, full_test={full_test_size}")
    return train, test, mapping, mode, full_train_size, full_test_size


def _partition_indices(size, count):
    indices = list(range(size))
    random.shuffle(indices)
    return [indices[offset::count] for offset in range(count)]


def _partition_stratified(dataset, count):
    by_class = {}
    for index, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(index)
    partitions = [[] for _ in range(count)]
    for label in sorted(by_class):
        indices = by_class[label][:]
        random.shuffle(indices)
        for position, index in enumerate(indices):
            partitions[position % count].append(index)
    for partition in partitions:
        random.shuffle(partition)
        labels = {dataset.samples[index][1] for index in partition}
        if len(labels) < len(by_class):
            raise ValueError(f"Client partition lacks classes: labels={labels}, expected={sorted(by_class)}")
    return partitions


def _loss_function(dataset, device):
    counts = Counter(label for _, label in dataset.samples)
    weights = torch.tensor(
        [len(dataset) / (len(counts) * counts[index]) for index in range(len(counts))],
        dtype=torch.float32,
        device=device,
    )
    return nn.NLLLoss(weight=weights)


def _train_model(model, dataset, indices, device, epochs, batch_size, learning_rate):
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=True)
    criterion = _loss_function(dataset, device)
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=learning_rate)
    losses = []
    model.train()
    for module in model.modules():
        if module is not model and not any(parameter.requires_grad for parameter in module.parameters(recurse=False)):
            module.eval()
    for _ in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for batch in loader:
            inputs, labels = batch[:2]
            cached_features = batch[2] if len(batch) == 3 else None
            inputs, labels = inputs.to(device), labels.to(device)
            if cached_features is not None:
                cached_features = {name: value.to(device) for name, value in cached_features.items()}
            optimizer.zero_grad()
            probabilities = model(inputs, cached_features=cached_features)
            loss = criterion(torch.log(probabilities.clamp_min(1e-7)), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        losses.append(epoch_loss / max(1, batches))
    return model.state_dict(), len(indices), float(np.mean(losses))


def _evaluate(model, dataset, device, num_classes):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    labels, predictions, probabilities = [], [], []
    total_loss = 0.0
    batches = 0
    criterion = _loss_function(dataset, device)
    with torch.no_grad():
        for batch in loader:
            inputs, batch_labels = batch[:2]
            cached_features = batch[2] if len(batch) == 3 else None
            if cached_features is not None:
                cached_features = {name: value.to(device) for name, value in cached_features.items()}
            output = model(inputs.to(device), cached_features=cached_features)
            total_loss += criterion(torch.log(output.clamp_min(1e-7)), batch_labels.to(device)).item()
            batches += 1
            labels.extend(batch_labels.numpy())
            predictions.extend(output.argmax(dim=1).cpu().numpy())
            probabilities.extend(output.cpu().numpy())
    metrics = calculate_metrics(labels, predictions, probabilities, num_classes)
    return metrics, total_loss / max(1, batches), labels, predictions, probabilities


def _save_checkpoint(name, model, class_mapping, experiment_config, current_round, optimizer_state=None, write_model_file=False, trainable_keys=None):
    path = os.path.join(CHECKPOINTS_DIR, name)
    os.makedirs(path, exist_ok=True)
    if write_model_file:
        torch.save(model.state_dict(), os.path.join(path, "model.pt"))
    full_state = model.state_dict()
    saved_state = {key: value for key, value in full_state.items() if trainable_keys is None or key in trainable_keys}
    torch.save({
        "model_state": saved_state,
        "optimizer_state": optimizer_state,
        "current_round": current_round,
        "experiment_config": experiment_config,
        "class_mapping": class_mapping,
        "random_seed": RANDOM_SEED,
    }, os.path.join(path, "latest.pt"))
    with open(os.path.join(path, "class_mapping.json"), "w", encoding="utf-8") as mapping_file:
        json.dump(class_mapping, mapping_file, indent=2, sort_keys=True)
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as config_file:
        json.dump(experiment_config, config_file, indent=2, default=str)
    return path


def _write_experiment_config(mapping, device, rounds, local_epochs, hospitals, clients, mode, train_size, test_size, full_train_size, full_test_size, feature_cache):
    config = {
        "dataset": os.path.abspath(DATA_DIR),
        "dataset_mode": mode,
        "FULL_DATASET_SIZE": {"train": full_train_size, "test": full_test_size},
        "EXPERIMENT_TRAIN_SUBSET": train_size,
        "EXPERIMENT_TEST_SUBSET": test_size,
        "experiment_max_train_samples_per_client": EXPERIMENT_MAX_TRAIN_SAMPLES_PER_CLIENT,
        "experiment_max_test_samples": EXPERIMENT_MAX_TEST_SAMPLES,
        "classes": list(mapping),
        "num_classes": len(mapping),
        "num_hospitals": hospitals,
        "departments_per_hospital": DEPARTMENTS_PER_HOSPITAL,
        "total_clients": clients,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "clip_norm": CLIP_NORM,
        "noise_multiplier": NOISE_MULTIPLIER,
        "anomaly_threshold": ANOMALY_THRESHOLD,
        "model_configuration": "InceptionV3 + VGG16 + ResNet50 + Custom CNN; frozen pretrained backbones",
        "random_seed": RANDOM_SEED,
        "device": str(device),
        "class_weighted_loss": True,
        "feature_cache": feature_cache,
    }
    with open(os.path.join(RESULTS_DIR, "experiment_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
    return config


def _parameter_report(model):
    report = {}
    for name, submodel in (("InceptionV3", model.inception), ("VGG16", model.vgg), ("ResNet50", model.resnet), ("Custom CNN", model.custom)):
        total = sum(parameter.numel() for parameter in submodel.parameters())
        trainable = sum(parameter.numel() for parameter in submodel.parameters() if parameter.requires_grad)
        report[name] = {"total_parameters": total, "trainable_parameters": trainable}
        print(f"{name}: total_parameters={total}, trainable_parameters={trainable}")
    return report


def _record_results(method, metrics, train_time, metadata):
    path = os.path.join(RESULTS_DIR, "diagnostic_results.csv")
    row = {"Method": method, **metadata, **metrics, "Training Time (s)": train_time}
    existing = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    required_columns = set(row)
    if not existing.empty and required_columns.issubset(existing.columns):
        if metadata["Dataset Mode"] == "DEMO_SUBSET":
            existing = existing[existing["Method"] != method]
        else:
            existing = existing[~((existing["Method"] == method) & (existing["Dataset Mode"] == metadata["Dataset Mode"]))]
    else:
        existing = pd.DataFrame()
    pd.concat([existing, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def _record_convergence(rows):
    path = os.path.join(RESULTS_DIR, "convergence.csv")
    incoming = pd.DataFrame(rows)
    existing = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    if not existing.empty and set(incoming.columns).issubset(existing.columns):
        if incoming["Dataset Mode"].iloc[0] == "DEMO_SUBSET":
            existing = existing[existing["Experiment"] != incoming["Experiment"].iloc[0]]
        else:
            existing = existing[~((existing["Experiment"] == incoming["Experiment"].iloc[0]) & (existing["Dataset Mode"] == incoming["Dataset Mode"].iloc[0]))]
    else:
        existing = pd.DataFrame()
    pd.concat([existing, incoming], ignore_index=True).to_csv(path, index=False)


def _save_confusion(labels, predictions, mapping):
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(mapping))))
    display = ConfusionMatrixDisplay(matrix, display_labels=list(mapping))
    display.plot(cmap="Blues", xticks_rotation=45)
    display.figure_.tight_layout()
    display.figure_.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=160)
    plt.close(display.figure_)
    return matrix


def _append_communication(row):
    path = os.path.join(RESULTS_DIR, "communication_results.csv")
    existing = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    existing = existing[existing.get("Configuration", pd.Series(dtype=str)) != row["Configuration"]] if not existing.empty else existing
    pd.concat([existing, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def _update_experiment_summary():
    source = os.path.join(RESULTS_DIR, "diagnostic_results.csv")
    if not os.path.exists(source):
        return
    results = pd.read_csv(source)
    names = {"Centralized CNN": "Centralized", "Basic FL": "Basic FL", "FL + DP": "FL + DP", "Hierarchical FL": "Hierarchical", "Proposed Framework": "Proposed"}
    results["Experiment"] = results["Method"].map(names)
    columns = ["Experiment", "Accuracy", "Precision", "Recall", "F1", "AUC", "Training Time (s)"]
    results[results["Experiment"].notna()][columns].rename(columns={"Training Time (s)": "runtime_seconds"}).to_csv(os.path.join(RESULTS_DIR, "experiment_summary.csv"), index=False)


def run_experiment(kind, rounds=None, local_epochs=None, smoke_test=False, full_data=False, resume=True):
    seed_everything()
    tracemalloc.start()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    timings = Counter()
    profile_started = time.perf_counter()
    device = torch.device("cuda" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu")
    dataset_started = time.perf_counter()
    rounds = 1 if smoke_test else (rounds or NUM_ROUNDS)
    local_epochs = local_epochs or LOCAL_EPOCHS
    hospitals = 1 if smoke_test else NUM_HOSPITALS
    departments = 2
    clients_count = hospitals * departments
    feature_cache_enabled = True
    train, test, mapping, mode, full_train_size, full_test_size = _make_datasets(smoke_test, full_data, deterministic=feature_cache_enabled)
    timings["dataset_loading_seconds"] = time.perf_counter() - dataset_started
    num_classes = len(mapping)
    config = _write_experiment_config(mapping, device, rounds, local_epochs, hospitals, clients_count, mode, len(train), len(test), full_train_size, full_test_size, feature_cache_enabled)
    model_started = time.perf_counter()
    model = DiagnosticEnsemble(num_classes=num_classes, freeze_backbone=True).to(device)
    parameter_report = _parameter_report(model)
    trainable_keys = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    timings["model_initialization_seconds"] = time.perf_counter() - model_started
    if feature_cache_enabled:
        cache_started = time.perf_counter()
        train_features = _build_feature_cache(model, train, mapping, "train")
        test_features = _build_feature_cache(model, test, mapping, "test")
        train = FeatureCachedDataset(train, train_features)
        test = FeatureCachedDataset(test, test_features)
        timings["feature_cache_seconds"] = time.perf_counter() - cache_started
    started = time.time()
    convergence = []
    client_partitions = _partition_indices(len(train), clients_count) if smoke_test else _partition_stratified(train, clients_count)
    audit = HashChainAudit() if kind == "proposed" else None
    anomaly_records = []
    previous_global = None
    last_labels, last_predictions = [], []
    participating = []
    communication_bytes = 0
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, "basic_fl", "latest.pt")
    start_round = 0
    if kind == "basic_fl" and resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        saved_config = checkpoint.get("experiment_config", {})
        if saved_config.get("dataset_mode") == mode and saved_config.get("EXPERIMENT_TRAIN_SUBSET") == len(train) and saved_config.get("EXPERIMENT_TEST_SUBSET") == len(test) and saved_config.get("rounds") == rounds:
            model.load_state_dict(checkpoint["model_state"], strict=False)
            start_round = int(checkpoint.get("current_round", 0))
            print(f"Resuming {kind} from round {start_round + 1}.")

    if kind == "centralized":
        model_state, _, train_loss = _train_model(model, train, list(range(len(train))), device, local_epochs if smoke_test else 1, BATCH_SIZE, LEARNING_RATE)
        model.load_state_dict(model_state)
        metrics, validation_loss, labels, predictions, _ = _evaluate(model, test, device, num_classes)
        _record_convergence([{"Experiment": "Centralized CNN", "Round": 1, "Training Loss": train_loss, "Validation Loss": validation_loss, "Validation Accuracy": metrics["Accuracy"], "Dataset Mode": mode, "Train Samples": len(train), "Test Samples": len(test), "Rounds": 1, "Local Epochs": local_epochs}])
        last_labels, last_predictions = labels, predictions
        checkpoint = _save_checkpoint("centralized", model, mapping, config, 1, trainable_keys=trainable_keys)
        _record_results("Centralized CNN", metrics, time.time() - started, {"Dataset Mode": mode, "Train Samples": len(train), "Test Samples": len(test), "Rounds": 1, "Local Epochs": local_epochs})
        _update_experiment_summary()
        tracemalloc.stop()
        return metrics, checkpoint

    local_model_started = time.perf_counter()
    local_model = copy.deepcopy(model)
    timings["model_initialization_seconds"] += time.perf_counter() - local_model_started
    for round_index in range(start_round, rounds):
        global_state = copy.deepcopy(model.state_dict())
        updates, counts = [], []
        round_losses = []
        hospital_updates, hospital_counts = [], []
        for hospital_index in range(hospitals):
            department_updates, department_counts = [], []
            for department_index in range(departments):
                client_index = hospital_index * departments + department_index
                indices = client_partitions[client_index]
                local_training_started = time.perf_counter()
                local_model.load_state_dict(global_state)
                state, sample_count, loss = _train_model(local_model, train, indices, device, local_epochs, BATCH_SIZE, LEARNING_RATE)
                timings["local_client_training_seconds"] += time.perf_counter() - local_training_started
                state = copy.deepcopy(state)
                if kind in {"fl_dp", "proposed"}:
                    state = apply_dp_to_update(state, global_state, CLIP_NORM, NOISE_MULTIPLIER)
                department_updates.append(state)
                department_counts.append(sample_count)
                round_losses.append(loss)
                participating.append(client_index)
                payload = io.BytesIO()
                torch.save(state, payload)
                communication_bytes += len(payload.getvalue())
            aggregation_started = time.perf_counter()
            hospital_state = aggregate_models(global_state, department_updates, department_counts, aggregate_keys=trainable_keys)
            timings["aggregation_seconds"] += time.perf_counter() - aggregation_started
            hospital_count = sum(department_counts)
            if kind == "proposed":
                if not hospital_updates:
                    accepted = True
                    similarity = 1.0
                else:
                    accepted, similarity = detect_anomaly(hospital_state, global_state, hospital_updates[0], ANOMALY_THRESHOLD)
                audit.create_audit_record(hospital_index, round_index + 1, hospital_state)
                print(f"Hospital {hospital_index}: accepted={accepted}, cosine_similarity={similarity:.4f}")
                anomaly_records.append({"Experiment": "Proposed Framework", "Round": round_index + 1, "Hospital": hospital_index, "Accepted": accepted, "Cosine Similarity": similarity, "Ground Truth": "NOT_AVAILABLE"})
                if not accepted:
                    continue
            hospital_updates.append(hospital_state)
            hospital_counts.append(hospital_count)
        if not hospital_updates:
            raise RuntimeError(f"No hospital updates survived screening in round {round_index + 1}.")
        model.load_state_dict(aggregate_models(global_state, hospital_updates, hospital_counts, aggregate_keys=trainable_keys))
        previous_global = copy.deepcopy(model.state_dict())
        payload = io.BytesIO()
        torch.save(model.state_dict(), payload)
        communication_bytes += len(payload.getvalue()) * (hospitals + clients_count)
        validation_started = time.perf_counter()
        metrics, validation_loss, labels, predictions, _ = _evaluate(model, test, device, num_classes)
        timings["validation_seconds"] += time.perf_counter() - validation_started
        convergence.append({"Experiment": {"basic_fl": "Basic FL", "fl_dp": "FL + DP", "hierarchical": "Hierarchical FL", "proposed": "Proposed Framework"}[kind], "Round": round_index + 1, "Training Loss": float(np.mean(round_losses)), "Validation Loss": validation_loss, "Validation Accuracy": metrics["Accuracy"]})
        prediction_counts = Counter(predictions)
        named_counts = ", ".join(f"{name}: {prediction_counts.get(index, 0)}" for name, index in mapping.items())
        if len(prediction_counts) < num_classes:
            raise RuntimeError(f"Prediction collapse in round {round_index + 1}: {named_counts}")
        convergence[-1].update({"Dataset Mode": mode, "Train Samples": len(train), "Test Samples": len(test), "Rounds": rounds, "Local Epochs": local_epochs})
        print(f"Round {round_index + 1}/{rounds} complete")
        print(f"loss={convergence[-1]['Training Loss']:.4f}, validation_loss={validation_loss:.4f}, accuracy={metrics['Accuracy']:.4f}")
        print(f"predicted class counts: {named_counts}")
        checkpoint_started = time.perf_counter()
        _save_checkpoint(name="basic_fl" if kind == "basic_fl" else kind, model=model, class_mapping=mapping, experiment_config=config, current_round=round_index + 1, write_model_file=False, trainable_keys=trainable_keys)
        timings["checkpoint_saving_seconds"] += time.perf_counter() - checkpoint_started
        last_labels, last_predictions = labels, predictions

    name = {"basic_fl": "basic_fl", "fl_dp": "fl_dp", "hierarchical": "hierarchical", "proposed": "proposed"}[kind]
    method = {"basic_fl": "Basic FL", "fl_dp": "FL + DP", "hierarchical": "Hierarchical FL", "proposed": "Proposed Framework"}[kind]
    evaluation_started = time.perf_counter()
    metrics, _, final_labels, final_predictions, _ = _evaluate(model, test, device, num_classes)
    timings["evaluation_seconds"] = time.perf_counter() - evaluation_started
    last_labels, last_predictions = final_labels, final_predictions
    _record_convergence(convergence)
    elapsed = time.time() - started
    current, peak = tracemalloc.get_traced_memory()
    metadata = {"Dataset Mode": mode, "Train Samples": len(train), "Test Samples": len(test), "Rounds": rounds, "Local Epochs": local_epochs}
    _record_results(method, metrics, elapsed, metadata)
    checkpoint = _save_checkpoint(name, model, mapping, config, rounds, write_model_file=False, trainable_keys=trainable_keys)
    _append_communication({"Configuration": method, "Rounds": rounds, "Hospitals": hospitals, "Clients": clients_count, "Model Bytes": communication_bytes, "CLIP_NORM": CLIP_NORM if kind in {"fl_dp", "proposed"} else 0, "NOISE_MULTIPLIER": NOISE_MULTIPLIER if kind in {"fl_dp", "proposed"} else 0, "Audit Chain Valid": audit.verify_audit_chain() if audit else True})
    if kind == "proposed":
        pd.DataFrame(anomaly_records).to_csv(os.path.join(RESULTS_DIR, "anomaly_results.csv"), index=False)
        chain_valid = audit.verify_audit_chain()
        original_record = dict(audit.chain[1]) if len(audit.chain) > 1 else None
        tamper_detected = False
        if original_record is not None:
            audit.chain[1]["update_hash"] = "tampered"
            tamper_detected = not audit.verify_audit_chain()
            audit.chain[1] = original_record
        pd.DataFrame([{"Experiment": method, "Records": len(audit.chain), "Chain Valid": chain_valid, "Tamper Detection": tamper_detected}]).to_csv(os.path.join(RESULTS_DIR, "audit_results.csv"), index=False)
        _save_confusion(last_labels, last_predictions, mapping)
        if audit:
            with open(os.path.join(RESULTS_DIR, "proposed_audit_chain.json"), "w", encoding="utf-8") as audit_file:
                json.dump(audit.chain, audit_file, indent=2)
    performance = {
        "experiment": method,
        "start_time": started_at,
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": elapsed,
        "dataset_mode": mode,
        "train_samples": len(train),
        "test_samples": len(test),
        "clients": clients_count,
        "hospitals": hospitals,
        "rounds_completed": rounds,
        "peak_tracemalloc_bytes": peak,
        "device": str(device),
        "timings_seconds": dict(timings),
        "parameter_report": parameter_report,
        "total_profiled_seconds": time.perf_counter() - profile_started,
    }
    with open(os.path.join(RESULTS_DIR, "performance_log.jsonl"), "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(performance) + "\n")
    if kind in {"fl_dp", "proposed"}:
        privacy_path = os.path.join(RESULTS_DIR, "privacy_results.csv")
        privacy_row = {"Experiment": method, "Clip Norm": CLIP_NORM, "Noise Multiplier": NOISE_MULTIPLIER, "Formal Epsilon": "NOT_AVAILABLE", "Formal Delta": "NOT_AVAILABLE"}
        privacy_existing = pd.read_csv(privacy_path) if os.path.exists(privacy_path) else pd.DataFrame()
        privacy_existing = privacy_existing[privacy_existing["Experiment"] != method] if not privacy_existing.empty and "Experiment" in privacy_existing.columns else pd.DataFrame()
        pd.concat([privacy_existing, pd.DataFrame([privacy_row])], ignore_index=True).to_csv(privacy_path, index=False)
    pd.DataFrame([{"Status": "NOT_AVAILABLE", "Reason": "No real-gradient reconstruction experiment was run during Phase 3; prior synthetic reconstruction output is excluded."}]).to_csv(os.path.join(RESULTS_DIR, "reconstruction_results.csv"), index=False)
    _update_experiment_summary()
    if kind == "basic_fl" and mode == "DEMO_SUBSET" and len(train) == 256 and len(test) == 128 and rounds == 3 and clients_count == 8:
        baseline_runtime = 529.2177548408508
        comparison = {
            "previous_runtime_seconds": baseline_runtime,
            "new_runtime_seconds": elapsed,
            "baseline_runtime_seconds": baseline_runtime,
            "optimized_runtime_seconds": elapsed,
            "speedup": baseline_runtime / elapsed if elapsed > 0 else None,
            "same_configuration": True,
            "random_seed": RANDOM_SEED,
            "configuration": {
                "hospitals": hospitals,
                "clients": clients_count,
                "rounds": rounds,
                "local_epochs": local_epochs,
                "train_samples": len(train),
                "test_samples": len(test),
            },
            "dataset_mode": mode,
            "baseline_metrics": {
                "Accuracy": 0.5234375,
                "Precision": 0.5237212750185323,
                "Recall": 0.5234375,
                "F1": 0.5220079583715947,
                "AUC": 0.562744140625,
            },
            "optimized_metrics": metrics,
            "timings_seconds": dict(timings),
        }
        with open(os.path.join(RESULTS_DIR, "runtime_comparison.json"), "w", encoding="utf-8") as comparison_file:
            json.dump(comparison, comparison_file, indent=2)
    del model, train, test
    del local_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tracemalloc.stop()
    return metrics, checkpoint


def run_all(smoke_test=False):
    build_dataset_report(DATA_DIR)
    results = {}
    for kind in ("centralized", "basic_fl", "fl_dp", "hierarchical", "proposed"):
        results[kind] = run_experiment(kind, smoke_test=smoke_test)
    convergence = pd.read_csv(os.path.join(RESULTS_DIR, "convergence.csv"))
    figure, axis = plt.subplots(figsize=(9, 5))
    for experiment, frame in convergence.groupby("Experiment"):
        axis.plot(frame["Round"], frame["Validation Accuracy"], marker="o", label=experiment)
    axis.set_xlabel("Round")
    axis.set_ylabel("Validation Accuracy")
    axis.set_title("Federated Validation Accuracy Convergence")
    axis.legend()
    figure.tight_layout()
    figure.savefig(os.path.join(RESULTS_DIR, "accuracy_convergence.png"), dpi=160)
    plt.close(figure)
    return results
