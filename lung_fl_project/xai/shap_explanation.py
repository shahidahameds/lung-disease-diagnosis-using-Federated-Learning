import shap
import torch
import numpy as np

def generate_shap(model, background_images, test_image):
    """
    Generates SHAP values for the test_image given a set of background_images.
    """
    model.eval()
    
    # We use GradientExplainer for deep learning models in PyTorch
    # SHAP expects the model to output a tensor, which our Ensemble does (probabilities)
    explainer = shap.GradientExplainer(model, background_images)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(test_image)
    
    # shap_values is typically a list of arrays (one for each class).
    # If it's a list, we return the values for the predicted class.
    # Sometimes for PyTorch it returns a tensor or list.
    with torch.no_grad():
        preds = model(test_image)
        pred_class = torch.argmax(preds, dim=1).item()
        
    # Get SHAP values for predicted class
    class_shap_values = shap_values[pred_class]
    
    # Format the SHAP values to be visualized as an image (sum across channels for simplicity)
    shap_image = np.abs(class_shap_values[0]).sum(axis=0)
    
    # Normalize
    shap_image /= np.max(shap_image) + 1e-7
    
    return shap_image
