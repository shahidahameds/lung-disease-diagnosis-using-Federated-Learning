import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

def get_vgg(num_classes=2, freeze_backbone=True):
    weights = VGG16_Weights.DEFAULT
    model = vgg16(weights=weights)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    # Replace final classification layer
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    for param in model.classifier[6].parameters():
        param.requires_grad = True
    
    return model
