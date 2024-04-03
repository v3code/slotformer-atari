"""Testing script for the video prediction task."""

import os
import sys
import numpy as np
import importlib
import argparse
from tqdm import tqdm

import torch

from nerv.utils import AverageMeter, save_video
from nerv.training import BaseDataModule

from vp_utils import pred_eval_step, postproc_mask, masks_to_boxes, \
    PALETTE_torch
from vp_vis import make_video, batch_draw_bbox
from models import build_model
from datasets import build_dataset

import lpips

# loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

def adjust_params(params, batch_size):
    """Adjust config files for testing."""
    if batch_size > 0:
        params.val_batch_size = batch_size
    else:
        params.val_batch_size = 12 if 'obj3d' in params.dataset.lower() else 8

    # rollout the model until 50 steps for OBJ3D dataset
    if 'obj3d' in params.dataset.lower():
        num_frames = 50
    # rollout the model until 48 steps for CLEVRER dataset
    elif 'clevrer' in params.dataset.lower():
        num_frames = 48
        params.load_mask = True  # test mask/bbox
    else:
        raise NotImplementedError('Unknown dataset: {}'.format(params.dataset))

    params.n_sample_frames = num_frames
    if params.model == 'SlotFormer':
        params.loss_dict['rollout_len'] = num_frames - params.input_frames
    else:
        raise NotImplementedError(f'Unknown model: {params.model}')

    # setup rollout image
    if params.model == 'SlotFormer':
        params.loss_dict['use_img_recon_loss'] = True
    params.load_img = True

    return params



@torch.no_grad()
def main(params):
    params = adjust_params(params, args.batch_size)

    val_set = build_dataset(params, val_only=True)
    datamodule = BaseDataModule(
        params, train_set=val_set, val_set=val_set, use_ddp=False)
    val_loader = datamodule.val_loader

    model = build_model(params).eval().cuda()
    model.load_state_dict(
        torch.load(args.weight, map_location='cpu')['state_dict'])

    history_len = params.input_frames
    rollout_len = params.n_sample_frames - history_len
    
    # pred_states, gt_states = [], []
    steps = params.steps
    
    mrr_metrics = {
        f"{step}_step": AverageMeter() for step in steps
    }
    
    hits_metrics = {
        f"{step}_step": AverageMeter() for step in steps
    }
    
    # pred_states, gt_states = [], []
    
    for data_dict in tqdm(val_loader):
        out_dict = model(data_dict)
        
        gt_slots = data_dict['slots'][:, history_len:]
        pred_slots = out_dict['pred_slots']
        
        # pred_states.append(build_state_dict(pred_slots, steps))
        # gt_states.append(build_state_dict(gt_slots, steps))
        

        for step in steps:
            gt_slots_on_step = gt_slots[:, step]
            pred_slots_on_step = pred_slots[:, step]
            
            
        # if agrs.batch_size * 
    

def build_state_dict(slots, steps):
    state_dict = {}
    for step in steps:
        state_dict[f"{step}_steps"] = slots[:, step]
    
    return state_dict
        
        
     


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate video prediction')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument(
        '--weight', type=str, required=True, help='load weight')
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--save_num', type=int, default=-1)
    parser.add_argument('--steps', type=str, default="5, 10, 20")
    args = parser.parse_args()
    args.steps = list(map(int, args.steps.split(",")))
    
    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(os.path.dirname(args.params))
    args.params = os.path.basename(args.params)
    params = importlib.import_module(args.params)
    params = params.SlotFormerParams()
    params.ddp = False

    torch.backends.cudnn.benchmark = True
    main(params)
