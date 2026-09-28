import copy
import torch

def aggregate_models(global_model_state, local_updates, sample_counts, aggregate_keys=None):
    """
    Performs standard FedAvg: weighted average of model updates.
    """
    total_samples = sum(sample_counts)
    
    # Initialize the aggregated state with float tensors
    aggregate_keys = set(aggregate_keys or global_model_state.keys())
    aggregated_state = {key: value.clone() for key, value in global_model_state.items() if key not in aggregate_keys}
    for key in aggregate_keys:
        aggregated_state[key] = torch.zeros_like(global_model_state[key], dtype=torch.float32)
        
    # Add weighted updates
    for i, local_state in enumerate(local_updates):
        weight = sample_counts[i] / total_samples
        for key in aggregate_keys:
            aggregated_state[key] += local_state[key].to(torch.float32) * weight
            
    # Convert back to original types
    for key in aggregate_keys:
        if not global_model_state[key].is_floating_point():
            aggregated_state[key] = aggregated_state[key].round().to(global_model_state[key].dtype)
            
    return aggregated_state
