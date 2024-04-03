
from collections import defaultdict
import numpy as np
import torch

from utils import pairwise_distance_matrix


@torch.no_grad()
def calc_metrics(model, dataloader, num_steps, slots_state=True, logits=False, hits_at_seq=(1)):
    model.eval()
    
    pred_states = []
    next_states = []
    
    hits_at = defaultdict(int)
    num_samples = 0
    rr_sum = 0
    
    for batch_idx, data_batch in enumerate(dataloader):

        observations, actions, *_ = data_batch
        observations.cuda()
        actions.cuda()
        
        obs = observations[0]
        next_obs = observations[-1]
        
        state = model.extract_state(obs, slots_state)
        next_state = model.extract_state(next_obs, slots_state)
        
        pred_state = state
        for i in range(num_steps):
            pred_state = model.next_state(pred_state, actions[i], slots_state)
        
        
        pred_states.append(pred_state.cpu())
        next_states.append(next_state.cpu())
    
    pred_state_cat = torch.cat(pred_states, dim=0)
    next_state_cat = torch.cat(next_states, dim=0)
    
    full_size = pred_state_cat.size(0)

    # Flatten object/feature dimensions
    next_state_flat = next_state_cat.view(full_size, -1)
    pred_state_flat = pred_state_cat.view(full_size, -1)

    dist_matrix = pairwise_distance_matrix(next_state_flat, pred_state_flat)

    dist_matrix_diag = torch.diag(dist_matrix).unsqueeze(-1)
    dist_matrix_augmented = torch.cat(
        [dist_matrix_diag, dist_matrix], dim=1)
    
    # Workaround to get a stable sort in numpy.
    dist_np = dist_matrix_augmented.numpy()
    indices = []
    for row in dist_np:
        keys = (np.arange(len(row)), row)
        indices.append(np.lexsort(keys))
    indices = np.stack(indices, axis=0)
    indices = torch.from_numpy(indices).long()
    labels = torch.zeros(
        indices.size(0), device=indices.device,
        dtype=torch.int64).unsqueeze(-1)
    num_samples += full_size
    for k in hits_at_seq:
        match = (indices[:, :k] == labels).any(dim=1)
        num_matches = match.sum()
        hits_at[k] += num_matches.item()

    match = indices == labels
    _, ranks = match.max(1)

    reciprocal_ranks = torch.reciprocal(ranks.double() + 1)
    rr_sum += reciprocal_ranks.sum().item()

    res = dict()

    for k in hits_at_seq:
        result_hits = hits_at[k] / float(num_samples)
        res[f"HITS_at_{k}"] = result_hits
    
    result_mrr = rr_sum / float(num_samples)
    res['MRR'] = result_mrr
    return res

        
        
        
        
        

    