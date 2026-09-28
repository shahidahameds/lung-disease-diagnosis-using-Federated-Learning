import torch
import torch.nn.functional as F

def calculate_cosine_similarity(delta_m, delta_g):
    """
    Calculates cosine similarity between a client/hospital update and the reference global update.
    """
    # Flatten both into 1D vectors
    vec_m = []
    vec_g = []
    
    for key in delta_m.keys():
        if delta_m[key].is_floating_point():
            vec_m.append(delta_m[key].view(-1))
            vec_g.append(delta_g[key].view(-1))
            
    vec_m = torch.cat(vec_m)
    vec_g = torch.cat(vec_g)
    
    if len(vec_m) == 0 or len(vec_g) == 0:
        return 0.0
    return F.cosine_similarity(vec_m.unsqueeze(0), vec_g.unsqueeze(0), eps=1e-8).item()

def detect_anomaly(update_state, global_state, reference_update_state, threshold=0.7):
    """
    Returns True if accepted, False if flagged as anomaly.
    """
    delta_m = {k: update_state[k] - global_state[k] for k in update_state.keys() if update_state[k].is_floating_point()}
    delta_g = {k: reference_update_state[k] - global_state[k] for k in reference_update_state.keys() if reference_update_state[k].is_floating_point()}
    
    sim = calculate_cosine_similarity(delta_m, delta_g)
    
    if sim >= threshold:
        return True, sim
    else:
        return False, sim
