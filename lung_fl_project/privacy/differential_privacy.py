import torch
import copy

def apply_dp_to_update(local_state, global_state, clip_threshold=1.0, noise_scale=0.1):
    """
    Applies Differential Privacy (gradient clipping and Gaussian noise)
    to the client's model update before sending it to the server.
    
    delta_w = local_state - global_state
    clipped_delta_w = delta_w * min(1, C / ||delta_w||_2)
    dp_delta_w = clipped_delta_w + N(0, sigma^2 C^2)
    """
    dp_state = copy.deepcopy(local_state)
    
    # 1. Calculate the L2 norm of the entire update (delta_w)
    total_norm = 0.0
    total_elements = 0
    deltas = {}
    
    for key in local_state.keys():
        if local_state[key].is_floating_point() and not key.endswith(("running_mean", "running_var")):
            delta = local_state[key] - global_state[key]
            if torch.count_nonzero(delta).item() > 0:
                deltas[key] = delta
                total_norm += torch.sum(delta ** 2).item()
                total_elements += delta.numel()
            
    total_norm = total_norm ** 0.5
    
    # 2. Calculate scaling factor
    clip_factor = min(1.0, clip_threshold / (total_norm + 1e-6))
    
    # 3. Apply clipping and noise
    for key in deltas.keys():
        # Clip
        clipped_delta = deltas[key] * clip_factor
        
        # Add Gaussian noise
        # std_dev = noise_scale * clip_threshold
        std_dev = noise_scale * clip_threshold / max(1, total_elements) ** 0.5
        noise = torch.randn_like(clipped_delta) * std_dev
        
        dp_delta = clipped_delta + noise
        
        # Reconstruct the state
        dp_state[key] = global_state[key] + dp_delta

    for key in local_state.keys():
        if key.endswith(("running_mean", "running_var")):
            dp_state[key] = local_state[key]
        
    return dp_state
