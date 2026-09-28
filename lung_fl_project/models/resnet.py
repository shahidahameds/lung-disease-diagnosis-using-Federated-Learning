import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_resnet(num_classes=2, freeze_backbone=True):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    # Replace the final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    # Unfreeze the final layer (just in case freeze_backbone caught it)
    for param in model.fc.parameters():
        param.requires_grad = True
        
    return model
