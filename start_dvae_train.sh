#!/bin/bash
export PYTHONPATH=.
python ./slotformer/rl/collect.py pong 
python ./slotformer/rl/collect.py pong --split val --seed 44 --episodes 1000

# python ./scripts/train.py --task base_slots --params slotformer/base_slots/configs/dvae_pong_params.py