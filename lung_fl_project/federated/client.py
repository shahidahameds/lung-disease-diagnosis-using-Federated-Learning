import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

class DepartmentClient:
    def __init__(self, client_id, dataset, indices, batch_size=16, local_epochs=1, lr=1e-4, device='cpu'):
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = device
        
        # Create local dataset subset
        self.dataset = Subset(dataset, indices)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.num_samples = len(self.dataset)
        
        # Loss function for probabilities (NLLLoss with log)
        class EnsembleLoss(nn.Module):
            def __init__(self):
                super(EnsembleLoss, self).__init__()
                self.nll = nn.NLLLoss()
                
            def forward(self, probs, target):
                probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
                return self.nll(torch.log(probs), target)
                
        self.criterion = EnsembleLoss()

    def train(self, global_model):
        """
        Trains locally on the global model and returns the updated model state and sample count.
        """
        # Deep copy the model to simulate local isolated training
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()
        
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, local_model.parameters()), lr=self.lr)
        
        for epoch in range(self.local_epochs):
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                probs = local_model(inputs)
                loss = self.criterion(probs, labels)
                loss.backward()
                optimizer.step()
                
        # Move back to CPU to save GPU memory during simulation
        local_model.to('cpu')
        
        return local_model.state_dict(), self.num_samples
