import cv2
import numpy as np
import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook the target layer
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate(self, input_image, target_class=None):
        self.model.eval()
        
        # Forward pass
        model_output = self.model(input_image)
        
        if target_class is None:
            target_class = torch.argmax(model_output, dim=1).item()
            
        # Clear gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        # (Assuming model_output are probabilities, we want to backpropagate from the log probability)
        loss = torch.log(model_output[0, target_class] + 1e-7)
        loss.backward()
        
        # Get pooled gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the channels by corresponding gradients
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the activations
        heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
        
        # ReLU on heatmap
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        heatmap /= np.max(heatmap) + 1e-7
        
        # Resize to input image size
        heatmap = cv2.resize(heatmap, (input_image.shape[3], input_image.shape[2]))
        
        return heatmap
