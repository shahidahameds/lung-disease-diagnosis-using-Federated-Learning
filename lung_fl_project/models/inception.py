import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights

def get_inception(num_classes=2, freeze_backbone=True):
    # Using the standard pretrained weights
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights)
    model.AuxLogits = None
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    # Replace the final classification layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model
