import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from config import RESULTS_DIR
from models.custom_cnn import CustomCNN

def dummy_reconstruction_experiment(original_image, dummy_model, original_gradient, steps=100):
    device = original_image.device
    # Start from random noise
    dummy_image = torch.randn_like(original_image, requires_grad=True).to(device)
    optimizer = optim.Adam([dummy_image], lr=0.1)
    
    for step in range(steps):
        optimizer.zero_grad()
        dummy_model.zero_grad()
        
        dummy_pred = dummy_model(dummy_image)
        loss = F.cross_entropy(dummy_pred, torch.tensor([0]).to(device))
        
        dummy_gradient = torch.autograd.grad(loss, dummy_model.parameters(), create_graph=True)
        
        grad_diff = 0
        for dg, og in zip(dummy_gradient, original_gradient):
            grad_diff += ((dg - og) ** 2).sum()
            
        grad_diff.backward()
        optimizer.step()
        
    return dummy_image.detach()

def run_reconstruction():
    print("\n--- PHASE 10: MINIMAL Gradient Reconstruction Experiment ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CustomCNN(num_classes=2).to(device)
    model.eval()
    
    # 1. Unprotected
    original_image = torch.rand(1, 3, 299, 299).to(device)
    model.zero_grad()
    pred = model(original_image)
    loss = F.cross_entropy(pred, torch.tensor([0]).to(device))
    original_gradient = [g.detach() for g in torch.autograd.grad(loss, model.parameters())]
    
    recon_unprotected = dummy_reconstruction_experiment(original_image, model, original_gradient, steps=50)
    
    # 2. Gradient Clipping (simulate by scaling down the norm)
    clip_C = 1.0
    clipped_gradient = []
    total_norm = sum((g**2).sum() for g in original_gradient) ** 0.5
    scale = min(1.0, clip_C / (total_norm.item() + 1e-6))
    for g in original_gradient:
        clipped_gradient.append(g * scale)
        
    recon_clipped = dummy_reconstruction_experiment(original_image, model, clipped_gradient, steps=50)
    
    # 3. DP (Clipping + Noise)
    dp_gradient = []
    noise_scale = 0.1
    for g in clipped_gradient:
        noise = torch.randn_like(g) * noise_scale * clip_C
        dp_gradient.append(g + noise)
        
    recon_dp = dummy_reconstruction_experiment(original_image, model, dp_gradient, steps=50)
    
    # Measure SSIM and PSNR
    def calc_metrics(orig, recon):
        orig_np = orig.squeeze().cpu().numpy().transpose(1, 2, 0)
        recon_np = recon.squeeze().cpu().numpy().transpose(1, 2, 0)
        
        # Normalize to 0-1 range for metrics
        orig_np = (orig_np - np.min(orig_np)) / (np.max(orig_np) - np.min(orig_np) + 1e-8)
        recon_np = (recon_np - np.min(recon_np)) / (np.max(recon_np) - np.min(recon_np) + 1e-8)
        
        s = ssim(orig_np, recon_np, data_range=1.0, channel_axis=-1)
        p = psnr(orig_np, recon_np, data_range=1.0)
        return s, p
        
    s_unp, p_unp = calc_metrics(original_image, recon_unprotected)
    s_clp, p_clp = calc_metrics(original_image, recon_clipped)
    s_dp, p_dp = calc_metrics(original_image, recon_dp)
    
    results = [
        {'Condition': 'Unprotected FL', 'SSIM': s_unp, 'PSNR': p_unp},
        {'Condition': 'Gradient Clipping', 'SSIM': s_clp, 'PSNR': p_clp},
        {'Condition': 'Gradient Clipping + DP', 'SSIM': s_dp, 'PSNR': p_dp}
    ]
    
    df = pd.DataFrame(results)
    path = os.path.join(RESULTS_DIR, 'reconstruction_results.csv')
    df.to_csv(path, index=False)
    print(f"Reconstruction test complete. Saved to {path}")
    print(df)

if __name__ == "__main__":
    run_reconstruction()
