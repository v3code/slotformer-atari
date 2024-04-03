import os
import pickle
import crafter
import slotformer.rl.envs
import shutil
from pathlib import Path
from typing import Tuple, Optional, List, Set, Union, AnyStr

import gym
import torch.nn.functional as F
import numpy.random
import numpy as np
import torch
from PIL import Image
from torch import nn
from skimage.transform import resize

from slotformer.rl.constants import Environments, STATE_IDS_TEMPLATE, \
    STATE_FOLDER_TEMPLATE, STATE_TEMPLATE, ACTIONS_TEMPLATE, \
    STATE_IDS_FOLDER_TEMPLATE, ACTIONS_FOLDER_TEMPLATE


def get_torch_device(device_option: Optional[str]) -> torch.device:
    if device_option is None:
        device_option = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device_option)


def get_environment(env_str: Environments, seed=None) -> gym.Env:
    if env_str == Environments.PONG:
        return gym.make("PongDeterministic-v4")
    elif env_str == Environments.SPACE_INVADERS:
        return gym.make("SpaceInvadersDeterministic-v4", seed=seed)
    elif env_str == Environments.CRAFTER:
        return gym.make("CrafterReward-v1", apply_api_compatibility=True, seed=seed)
    elif env_str == Environments.NAVIGATION_5x5:
        return gym.make("Navigation5x5-v0", seed=seed)
    elif env_str == Environments.PUSHING_5x5:
        return gym.make("Pushing5x5-v0", seed=seed)
    elif env_str == Environments.CUBES_3D:
        return gym.make("Cubes-v0", seed=seed)
    elif env_str == Environments.SHAPES_2D:
        return gym.make("Shapes-v0", seed=seed)
    else:
        raise NotImplementedError(f"Environment {env_str} is not supported")


def init_lib_seed(seed: int):
    torch.manual_seed(seed)
    numpy.random.seed(seed)


def raise_env_not_implemented_error(env: Environments):
    raise NotImplementedError(f'Config for "{env}" environment '
                              f'is not implemented')


def crop_normalize(img: np.ndarray,
                   crop_ratio: Tuple[int, int],
                   size: Optional[Tuple[int, int]] = None):
    img = img[crop_ratio[0]:crop_ratio[1]]
    img = Image.fromarray(img)
    if size:
        img = img.resize(size, Image.ANTIALIAS)
    return np.array(img) / 255


def construct_blacklist(
        black_list_folders: Optional[List[Path]] = None,
        is_numpy = False,
        ignore_blacklist = True
) -> Optional[Set[bytes]]:
    blacklist = set()

    if not black_list_folders or ignore_blacklist:
        return blacklist

    for path in black_list_folders:
        for dir_it in os.scandir(path):
            if dir_it.is_dir():
                file_path = os.path.join(dir_it.path, STATE_IDS_TEMPLATE)
                state_ids = load_state_id_from_path(file_path)
                first_id = state_ids[0]
                if not is_numpy:
                    first_id = np.array(first_id)
                blacklist.add(first_id.tobytes())

    return blacklist


def load_state_id_from_path(load_path: Union[Path, str]) -> np.ndarray:
    if not os.path.isfile(load_path):
        raise ValueError("State ids not found.")
    with open(load_path, "rb") as f:
        state_ids = pickle.load(f)
    return state_ids


def load_state_ids(path: Union[Path, bytes, str], episode: int) -> np.ndarray:
    load_path = os.path.join(path, STATE_IDS_FOLDER_TEMPLATE.format(episode))
    return load_state_id_from_path(load_path)


def delete_episode_observations(save_path: Path, episode: int):
    save_path = os.path.join(save_path,
                             STATE_FOLDER_TEMPLATE.format(episode))
    if not os.path.isdir(save_path):
        return
    shutil.rmtree(save_path)


def check_duplication(blacklist: Set[bytes], state_id: np.ndarray):
    if not blacklist:
        return False
    return state_id.tobytes() in blacklist


def clear_blacklist(blacklist: Set[bytes], episode_states: np.ndarray):
    for state in episode_states:
        blacklist.remove(state.tobytes())


def save_obs(ep: int, step: int, obs: np.ndarray, save_path: Path, normalized=True):
    save_path = os.path.join(save_path,
                             STATE_TEMPLATE.format(ep, step))
    maybe_create_dirs(get_dir_name(save_path))
    if normalized:
        obs = np.round(obs * 225).astype('uint8')
    # print(obs)
    image = Image.fromarray(obs)
    image.save(save_path)


def save_actions(actions: List[Union[np.ndarray, int]],
                 ep_index: int,
                 save_path: Path):
    save_path = os.path.join(save_path, ACTIONS_FOLDER_TEMPLATE.format(ep_index))
    maybe_create_dirs(get_dir_name(save_path))
    np.save(save_path, actions)


def load_action_from_path(load_path: Union[Path, str]) -> np.ndarray:
    if not os.path.isfile(load_path):
        raise ValueError("Actions not found.")
    actions = np.load(load_path)
    return actions


def load_actions(path: Union[Path, str, bytes], episode: int):
    load_path = os.path.join(path, ACTIONS_FOLDER_TEMPLATE.format(episode))
    return load_action_from_path(load_path)


def save_state_ids(state_ids: List[np.ndarray], ep_idx: int, save_path: Union[Path, str, bytes]):
    save_path = os.path.join(save_path, STATE_IDS_FOLDER_TEMPLATE.format(ep_idx))
    maybe_create_dirs(get_dir_name(save_path))
    np.save(save_path, state_ids)


def get_dir_name(path: Union[bytes, str, os.PathLike]) -> AnyStr:
    return os.path.dirname(path)


def maybe_create_dirs(dir_path: Union[bytes, str, os.PathLike]):
    if len(dir_path) == 0:
        return

    if not os.path.isdir(dir_path):
        if os.path.isfile(dir_path):
            raise ValueError(
                "File of the same name as target directory found.")
        os.makedirs(dir_path)


def select_action(state: np.ndarray, model: nn.Module, hx: torch.Tensor, eps: Optional[float]):
    # select an action using either an epsilon greedy or softmax policy
    value, logit, hx = model((state.view(1, 1, 80, 80), hx))
    logp = F.log_softmax(logit, dim=-1)

    if eps is not None:
        # use epsilon greedy
        if np.random.uniform(0, 1) < eps:
            # random action
            return np.random.randint(logp.size(1))
        else:
            return torch.argmax(logp, dim=1).cpu().numpy()[0]
    else:
        # sample from softmax
        action = torch.exp(logp).multinomial(num_samples=1).data[0]
        return action.cpu().numpy()[0]


def preprocess_state(state: np.ndarray, device: torch.device):
    state = resize(state[35:195].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255
    return torch.tensor(state, device=device)


def reset_rnn_state(device: torch.device, memsize: int):
    # reset the hidden state of an rnn
    return torch.zeros(1, memsize, device=device)
