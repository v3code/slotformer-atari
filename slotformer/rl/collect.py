import argparse
import copy
from pathlib import Path
from typing import Optional, List, Literal

import numpy as np
import torch
from tap import Tap
from tqdm import tqdm

from slotformer.rl.agents import get_agent
from slotformer.rl.configs.a3c_configs import get_a3c_config
from slotformer.rl.configs.collect_configs import get_collect_config
from slotformer.rl.constants import Environments
from slotformer.rl.models.factory import get_model
from slotformer.rl.utils import get_environment, init_lib_seed, \
    construct_blacklist, crop_normalize, check_duplication, save_actions, \
    save_state_ids, save_obs, reset_rnn_state, select_action, preprocess_state, delete_episode_observations


class CollectArgs(Tap):
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


def collect(args: CollectArgs):
    env = get_environment(args.environment)
    init_lib_seed(args.seed)
    collect_config = get_collect_config(args.environment, args.split)
    agent = get_agent(env, args.environment, collect_config, args.device)
    blacklist = construct_blacklist(collect_config.blacklist_paths)

    ep_idx = 0
    atari_env = args.environment != Environments.CRAFTER
    with tqdm(total=args.episodes) as pbar:
        while ep_idx < args.episodes:
            burnin_steps = np.random.randint(collect_config.min_burnin,
                                             collect_config.max_burnin)
            episode_states = []
            episode_actions = []
            agent.reset()
            prev_obs = env.reset()[0]
            step_idx = 0
            for _ in range(burnin_steps):
                action = agent.get_action(prev_obs, True)
                prev_obs, *_ = env.step(action)
            after_warmup = True

            while True:

                # TODO Refactor this

                if after_warmup:
                    action = 0
                else:
                    # select random action
                    action = agent.get_action(prev_obs)

                step_state = env.step(action)
                if len(step_state) == 5:
                    obs, _, truncated, terminated, _ = step_state
                    done = terminated
                else:
                    obs, _, done, _ = step_state

                state = None

                if atari_env:
                    state = copy.deepcopy(
                        np.array(env.ale.getRAM(), dtype=np.int32))

                if after_warmup:
                    after_warmup = False
                    if atari_env and check_duplication(blacklist, state):
                        break

                if collect_config.crop:
                    obs = crop_normalize(obs, collect_config.crop)
                episode_actions.append(action)
                step_idx += 1
                save_obs(
                    ep_idx, step_idx,
                    obs, collect_config.save_path
                )
                if state:
                    episode_states.append(state)

                if step_idx >= args.steps:
                    done = True

                if done:
                    if step_idx < args.steps:
                        delete_episode_observations(collect_config.save_path, ep_idx)
                        break
                    save_actions(episode_actions, ep_idx, collect_config.save_path)
                    save_state_ids(episode_states, ep_idx, collect_config.save_path)
                    ep_idx += 1
                    pbar.update(n=1)

                    break


if __name__ == "__main__":
    args = CollectArgs().parse_args()
    collect(args)
