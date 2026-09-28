import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import IMAGE_SIZE


TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def prepare_image(image):
    if image.mode not in {"RGB", "RGBA", "L", "LA", "I", "F"}:
        raise ValueError(f"Unsupported image mode: {image.mode}")
    rgb_image = image.convert("RGB")
    return rgb_image, TRANSFORM(rgb_image).unsqueeze(0)


def predict(model, tensor, device, class_mapping):
    with torch.no_grad():
        probabilities = model(tensor.to(device))[0].cpu().numpy()
    order = sorted(class_mapping.items(), key=lambda item: item[1])
    prediction_index = int(np.argmax(probabilities))
    return {
        "label": next(name for name, index in order if index == prediction_index),
        "class_index": prediction_index,
        "probabilities": {name: float(probabilities[index]) for name, index in order},
    }


def individual_predictions(model, tensor, device, class_mapping):
    model.eval()
    with torch.no_grad():
        outputs = {
            "InceptionV3": model.inception(tensor.to(device)),
            "VGG16": model.vgg(tensor.to(device)),
            "ResNet50": model.resnet(tensor.to(device)),
            "Custom CNN": model.custom(tensor.to(device)),
        }
    predictions = {}
    for name, output in outputs.items():
        if isinstance(output, tuple):
            output = output[0]
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        index = int(np.argmax(probs))
        label = next(class_name for class_name, class_index in class_mapping.items() if class_index == index)
        predictions[name] = {"label": label, "confidence": float(probs[index])}
    return predictions
