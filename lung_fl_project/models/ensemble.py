import torch
import torch.nn as nn
import torch.nn.functional as F

from models.inception import get_inception
from models.vgg import get_vgg
from models.resnet import get_resnet
from models.custom_cnn import get_custom_cnn

class DiagnosticEnsemble(nn.Module):
    def __init__(self, num_classes=2, weights=None, freeze_backbone=True):
        super(DiagnosticEnsemble, self).__init__()
        
        if weights is None:
            weights = {'inception': 0.25, 'vgg': 0.25, 'resnet': 0.25, 'custom': 0.25}
            
        self.weights = weights
        
        self.inception = get_inception(num_classes, freeze_backbone)
        self.vgg = get_vgg(num_classes, freeze_backbone)
        self.resnet = get_resnet(num_classes, freeze_backbone)
        self.custom = get_custom_cnn(num_classes, freeze_backbone)

        self.num_classes = num_classes

    def forward(self, x, cached_features=None):
        # Forward passes
        # Inception has aux outputs during training, but we only want the main output for probabilities
        if cached_features is None:
            inception_output = self.inception(x)
            out_inc = inception_output[0] if isinstance(inception_output, tuple) else inception_output
            out_vgg = self.vgg(x)
            out_res = self.resnet(x)
        else:
            out_inc = self.inception.fc(cached_features["inception"])
            out_vgg = self.vgg.classifier[6](cached_features["vgg"])
            out_res = self.resnet.fc(cached_features["resnet"])
        out_cus = self.custom(x)
        
        # Convert logits to probabilities
        p_inc = F.softmax(out_inc, dim=1)
        p_vgg = F.softmax(out_vgg, dim=1)
        p_res = F.softmax(out_res, dim=1)
        p_cus = F.softmax(out_cus, dim=1)
        
        # Weighted averaging
        p_ens = (self.weights['inception'] * p_inc +
                 self.weights['vgg'] * p_vgg +
                 self.weights['resnet'] * p_res +
                 self.weights['custom'] * p_cus)
                 
        return p_ens

    def load_submodel_state(self, model_name, state_dict):
        if model_name == 'inception':
            self.inception.load_state_dict(state_dict)
        elif model_name == 'vgg':
            self.vgg.load_state_dict(state_dict)
        elif model_name == 'resnet':
            self.resnet.load_state_dict(state_dict)
        elif model_name == 'custom':
            self.custom.load_state_dict(state_dict)
            
    def get_submodel_state(self, model_name):
        if model_name == 'inception':
            return self.inception.state_dict()
        elif model_name == 'vgg':
            return self.vgg.state_dict()
        elif model_name == 'resnet':
            return self.resnet.state_dict()
        elif model_name == 'custom':
            return self.custom.state_dict()
