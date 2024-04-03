from collections import defaultdict
import os.path
import argparse
import math
import numpy as np
import torch
import torchvision.utils as vutils
from datetime import datetime
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import get_env, match_slots, pairwise_distance_matrix, slots_distance_matrix
import wandb

from shapes_3d import Shapes3D
from slate import SLATE
from causal_world_push import CausalWorldPush
from env_data import EnvDataset, EnvTestDataset

parser = argparse.ArgumentParser()

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--patience', type=int, default=4)
parser.add_argument('--clip', type=float, default=1.0)
parser.add_argument('--image_size', type=int, default=64)

parser.add_argument('--action_size', type=int, default=6)
parser.add_argument('--num_episodes', type=int, default=1000)
parser.add_argument('--steps_per_episode', type=int, default=101)
parser.add_argument('--one_hot_actions', type=bool, default=True)

parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
parser.add_argument('--log_path', default='logs')
parser.add_argument('--data_path', default=None)
parser.add_argument('--env_name', default='pong')
parser.add_argument('--name', default='default')

parser.add_argument('--lr_dvae', type=float, default=3e-4)
parser.add_argument('--lr_main', type=float, default=1e-4)
parser.add_argument('--lr_warmup_steps', type=int, default=30000)

parser.add_argument('--num_dec_blocks', type=int, default=6)
parser.add_argument('--vocab_size', type=int, default=512)
parser.add_argument('--d_model', type=int, default=192)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--num_iterations', type=int, default=3)
parser.add_argument('--num_slots', type=int, default=3)
parser.add_argument('--num_slot_heads', type=int, default=1)
parser.add_argument('--slot_size', type=int, default=192)
parser.add_argument('--mlp_hidden_size', type=int, default=192)
parser.add_argument('--img_channels', type=int, default=3)
parser.add_argument('--pos_channels', type=int, default=4)

parser.add_argument('--tau_start', type=float, default=1.0)
parser.add_argument('--tau_final', type=float, default=0.1)
parser.add_argument('--tau_steps', type=int, default=30000) 
parser.add_argument('--hard', action='store_true')
parser.add_argument('--weights', required=True)
parser.add_argument('--test_steps', default='1,5,10,50,100')
parser.add_argument('--slots_state', default=True, type=bool)
parser.add_argument('--max_steps_per_batch', default=20, type=int)
parser.add_argument('--max_batches', default=20, type=int)


args = parser.parse_args()

torch.manual_seed(args.seed)

arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
arg_str = '__'.join(arg_str_list)
log_dir = os.path.join(args.log_path, datetime.today().isoformat())
writer = SummaryWriter(log_dir)
writer.add_text('hparams', arg_str)

loader_kwargs = {
    'batch_size': 1,
    'shuffle': True,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': True,
}

env = get_env(args.env_name, dict(image_size=args.image_size))


model = SLATE(args)
state = torch.load(args.weights, map_location='cpu')
model.load_state_dict(state)
model.cuda()
model.zero_action.cuda()
model.eval()

test_steps = map(int, args.test_steps.split(','))
test_sampler = None


def hit_at_k(predictions: torch.Tensor, ground_truth_idx: torch.Tensor, device: torch.device, k: int = 10) -> int:
    """Calculates number of hits@k.

    :param predictions: BxN tensor of prediction values where B is batch size and N number of classes. Predictions
    must be sorted in class ids order
    :param ground_truth_idx: Bx1 tensor with index of ground truth class
    :param device: device on which calculations are taking place
    :param k: number of top K results to be considered as hits
    :return: Hits@K score
    """
    assert predictions.size(0) == ground_truth_idx.size(0)

    zero_tensor = torch.tensor([0], device=device)
    one_tensor = torch.tensor([1], device=device)
    _, indices = predictions.topk(k=k, largest=False)
    print(indices)
    return torch.where(indices == ground_truth_idx, one_tensor, zero_tensor).sum().item()


def mrr(predictions: torch.Tensor, ground_truth_idx: torch.Tensor) -> float:
    """Calculates mean reciprocal rank (MRR) for given predictions and ground truth values.

    :param predictions: BxN tensor of prediction values where B is batch size and N number of classes. Predictions
    must be sorted in class ids order
    :param ground_truth_idx: Bx1 tensor with index of ground truth class
    :return: Mean reciprocal rank score
    """
    assert predictions.size(0) == ground_truth_idx.size(0)

    indices = predictions.argsort()
    return (1.0 / (indices == ground_truth_idx).nonzero()[:, 1].float().add(1.0)).sum().item()

    



@torch.no_grad()
def calc_metrics(model, dataloader, num_steps, slots_state=True, logits=False, hits_at_seq=(1,), max_batches=30):
    model.eval()
    
    pred_states = []
    next_states = []
    
    hits_at = defaultdict(int)
    num_samples = 0
    rr_sum = 0
    steps = 0
    res = dict()
        
    for batch_idx, data_batch in enumerate(dataloader):
        if batch_idx == max_batches:
            break
        observations, actions = data_batch
        observations = observations.cuda()
        actions = actions.cuda()
        _, T, *_ = observations.shape
        
        start_step = np.random.randint(0, T - num_steps - 1)
        obs = observations[:, start_step]
        next_obs = observations[:, start_step+num_steps]
        
        state = model.extract_state(obs, slots_state, logits)
        next_state = model.extract_state(next_obs, slots_state, logits)
        
        pred_state = state
        for i in range(num_steps):
            pred_state = model.next_state(pred_state, actions[:, i], slots_state, logits)
        
        pred_state = pred_state[0]
        next_state = next_state[0]
        
        if not slots_state:
            next_state = next_state.argmax(dim=1)
        
        pred_states.append(pred_state.cpu())
        next_states.append(next_state.cpu())
        steps += 1
        
    print('Calculating metrics')
    
    pred_state_cat = torch.cat(pred_states, dim=0)
    next_state_cat = torch.cat(next_states, dim=0)
    
    full_size = pred_state_cat.size(0)
    
    if not slots_state:
        # Flatten object/feature dimensions
        _, vocab, _, _ = pred_state_cat.shape
        next_state_flat = next_state_cat.flatten().unsqueeze(1)
        pred_state_flat = pred_state_cat.reshape(-1, vocab)
        for k in hits_at_seq:
            result_hits = hit_at_k(pred_state_flat, next_state_flat, device=torch.device('cpu'), k=k)
            res[f"HITS_at_{k}"] = result_hits
        res['MRR'] = mrr(pred_state_flat, next_state_flat)
        return res
    
    # pred_state_flat = match_slots(pred_state_cat, next_state_cat)
    # next_state_flat = next_state_cat.flatten(start_dim=1)
    # pred_state_flat = pred_state_cat.flatten(start_dim=1)

    # Flatten object/feature dimensions    writer.add_h
    dist_matrix = slots_distance_matrix(next_state_cat, pred_state_cat)

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


    for k in hits_at_seq:
        result_hits = hits_at[k] / float(num_samples)
        res[f"HITS_at_{k}"] = result_hits
    
    result_mrr = rr_sum / float(num_samples)
    res['MRR'] = result_mrr
    return res


test_dataset = EnvTestDataset(env, args.num_episodes, args.steps_per_episode,
                           one_hot_actions=True, data_path=os.path.join(args.data_path, 'test'))
test_loader = DataLoader(test_dataset, sampler=test_sampler, **loader_kwargs)

results = {}

with torch.no_grad():
    for k in test_steps:
        
        print(f'Evaluation {k} steps...')
        metrics = calc_metrics(model, test_loader, k, args.slots_state)
        print(f"Eval results at {k} steps")
        for key, val in metrics.items():
            print(f'{key}: {val}')
            writer.add_text(f'test_{k}_step', f'{key}: {val}')
        
    