import argparse
import copy
from pathlib import Path
from typing import Optional, List, Literal

import numpy as np
import torch
from tap import Tap
from tqdm import tqdm

from slotformer.atari.configs.a3c_configs import get_a3c_config
from slotformer.atari.configs.collect_configs import get_collect_config
from slotformer.atari.constants import Environments
from slotformer.atari.models.factory import get_model
from slotformer.atari.utils import get_environment, init_lib_seed, \
    construct_blacklist, crop_normalize, check_duplication, save_actions, \
    save_state_ids, save_obs, reset_rnn_state, select_action, preprocess_state


class AtariCollectArgs(Tap):
    environment: Environments
    steps: int = 50
    seed: int = 42
    episodes: int = 1000
    split: Literal['train', 'test', 'val'] = 'train'
    black_list: Optional[List[Path]] = None
    device: Optional[str] = None

    def configure(self) -> None:
        self.add_argument('environment',
                          type=str,
                          choices=[e.value for e in Environments])


def collect(args: AtariCollectArgs):
    env = get_environment(args.environment)
    init_lib_seed(args.seed)
    collect_config = get_collect_config(args.environment, args.split)
    a3c_config = get_a3c_config(args.environment)
    model = get_model(args.environment, collect_config.a3c_weights)
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    model.to(device)

    blacklist = construct_blacklist(collect_config.blacklist_paths)

    ep_idx = 0
    with tqdm(total=args.episodes) as pbar:
        while ep_idx < args.episodes:
            burnin_steps = np.random.randint(collect_config.min_burnin,
                                             collect_config.max_burnin)
            hx = reset_rnn_state(device, a3c_config.memsize)
            episode_states = []
            episode_actions = []

            prev_obs = env.reset()[0]
            step_idx = 0
            for _ in range(burnin_steps):
                action = select_action(preprocess_state(prev_obs, device),
                                       model, hx, collect_config.eps)
                prev_obs, _, _, _, _ = env.step(action)
            after_warmup = True

            while True:
                if after_warmup:
                    action = 0
                else:
                    # select random action
                    action = env.action_space.sample()

                obs, _, truncated, terminated, _ = env.step(action)

                done = truncated or terminated

                state = copy.deepcopy(
                    np.array(env.ale.getRAM(), dtype=np.int32))

                if after_warmup:
                    after_warmup = False
                    if check_duplication(blacklist, state):
                        continue

                if collect_config.crop:
                    obs = crop_normalize(obs, collect_config.crop)
                episode_actions.append(action)
                step_idx += 1
                save_obs(
                    ep_idx, step_idx,
                    obs, collect_config.save_path
                )

                episode_states.append(state)

                if step_idx >= args.steps:
                    done = True

                if done:
                    if step_idx < args.steps:
                        continue
                    save_actions(episode_actions, ep_idx, collect_config.save_path)
                    save_state_ids(episode_states, ep_idx, collect_config.save_path)
                    ep_idx += 1
                    pbar.update(n=1)

                    break


if __name__ == "__main__":
    args = AtariCollectArgs().parse_args()
    collect(args)
