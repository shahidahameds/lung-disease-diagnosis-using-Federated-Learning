import cv2
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def _rgb_uint8(image):
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Display image must be HxWx3 after RGB conversion; got {array.shape}.")
    return array


def _normalize_cam(cam):
    cam = np.asarray(cam, dtype=np.float32)
    cam = np.squeeze(cam)
    if cam.ndim != 2:
        raise ValueError(f"CAM must be a 2D spatial array before resizing; got {cam.shape}.")
    cam = np.maximum(cam, 0)
    minimum, maximum = float(cam.min()), float(cam.max())
    if maximum > minimum:
        cam = (cam - minimum) / (maximum - minimum)
    else:
        cam = np.zeros_like(cam, dtype=np.float32)
    return cam


def gradcam_custom(model, tensor, device, target_class, display_shape):
    activations = []
    gradients = []
    layer = model.custom.conv3
    forward_handle = layer.register_forward_hook(lambda _, __, output: activations.append(output))
    backward_handle = layer.register_full_backward_hook(lambda _, __, output: gradients.append(output[0]))
    try:
        model.zero_grad(set_to_none=True)
        output = model(tensor.to(device))
        output[0, target_class].backward()
        activation = activations[-1][0]
        gradient = gradients[-1][0]
        weights = gradient.mean(dim=(1, 2), keepdim=True)
        cam = _normalize_cam(torch.relu((weights * activation).sum(dim=0)).detach().cpu().numpy())
        resized_cam = cv2.resize(cam, (display_shape[1], display_shape[0]), interpolation=cv2.INTER_LINEAR)
        resized_cam = _normalize_cam(resized_cam)
        logger.info("XAI shapes original_shape=%s model_input_shape=%s cam_shape=%s resized_cam_shape=%s", display_shape, tuple(tensor.shape), cam.shape, resized_cam.shape)
        return resized_cam, {"original_shape": tuple(display_shape), "model_input_shape": tuple(tensor.shape), "cam_shape": tuple(cam.shape), "resized_cam_shape": tuple(resized_cam.shape)}
    finally:
        forward_handle.remove()
        backward_handle.remove()
        model.zero_grad(set_to_none=True)


def overlay(image, heatmap, alpha=0.45):
    base = _rgb_uint8(image)
    normalized = _normalize_cam(heatmap)
    if normalized.shape != base.shape[:2]:
        raise ValueError(f"Overlay shape mismatch after explicit resize: original={base.shape}, heatmap={normalized.shape}.")
    heat = cv2.applyColorMap(np.uint8(255 * normalized), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    if heat.shape != base.shape:
        raise ValueError(f"Overlay requires identical HxWx3 arrays: original={base.shape}, heatmap={heat.shape}.")
    result = cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)
    logger.info("XAI shapes heatmap_shape=%s overlay_shape=%s", heat.shape, result.shape)
    return result


def shap_to_heatmap(shap_output, display_shape):
    values = shap_output
    if isinstance(values, list):
        raise ValueError("SHAP output list requires the selected class to be extracted before conversion.")
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    values = np.asarray(values)
    values = np.squeeze(values)
    if values.ndim == 4:
        if values.shape[0] == 3:
            values = np.abs(values).sum(axis=0)
        elif values.shape[-1] == 3:
            values = np.abs(values).sum(axis=-1)
        else:
            raise ValueError(f"Unsupported SHAP channel shape: {values.shape}.")
    if values.ndim == 3:
        values = np.abs(values).sum(axis=-1 if values.shape[-1] == 3 else 0)
    if values.ndim != 2:
        raise ValueError(f"SHAP output must become 2D spatial data; got {values.shape}.")
    heatmap = _normalize_cam(values)
    resized = cv2.resize(heatmap, (display_shape[1], display_shape[0]), interpolation=cv2.INTER_LINEAR)
    resized = _normalize_cam(resized)
    logger.info("XAI shapes shap_shape=%s resized_shap_shape=%s", tuple(np.asarray(shap_output).shape), resized.shape)
    return resized, {"shap_shape": tuple(np.asarray(shap_output).shape), "resized_shap_shape": tuple(resized.shape)}


def select_shap_class(shap_values, target_class):
    if isinstance(shap_values, list):
        if target_class >= len(shap_values):
            raise ValueError(f"SHAP class index {target_class} is outside list length {len(shap_values)}.")
        return shap_values[target_class]
    values = shap_values
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    values = np.asarray(values)
    if values.ndim >= 5 and values.shape[-1] > target_class:
        return values[..., target_class]
    return values


def custom_cnn_shap(model, tensor, device, target_class):
    """Generate real SHAP attributions for the differentiable Custom CNN branch."""
    import shap

    branch = model.custom
    branch.eval()
    input_tensor = tensor.to(device)
    background = torch.cat((torch.zeros_like(input_tensor), input_tensor.detach()), dim=0)
    explainer = shap.GradientExplainer(branch, background)
    raw_values = explainer.shap_values(input_tensor)
    raw_shape = tuple(raw_values.shape) if hasattr(raw_values, "shape") else None
    logger.info("SHAP method=GradientExplainer branch=Custom CNN type=%s shape=%s model_input=%s class=%s", type(raw_values).__name__, raw_shape, tuple(tensor.shape), target_class)
    selected = select_shap_class(raw_values, target_class)
    return selected, {"method": "GradientExplainer", "branch": "Custom CNN", "shap_values_type": type(raw_values).__name__, "shap_values_shape": raw_shape, "model_input_shape": tuple(tensor.shape), "selected_class": target_class, "background_shape": tuple(background.shape)}
