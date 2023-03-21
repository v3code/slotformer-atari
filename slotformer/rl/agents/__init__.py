from typing import Optional

import gym

from slotformer.rl.utils import raise_env_not_implemented_error

from .a3c import A3CAtariAgent
from .random import RandomAgent
from ..configs.collect_configs import BaseCollectConfig
from ..constants import Environments


def get_agent(env: gym.Env,
              env_name: Environments,
              collect_config: BaseCollectConfig,
              device: Optional[str] = None):
    if env_name in (Environments.PONG, Environments.SPACE_INVADERS):
        return A3CAtariAgent(env, env_name, collect_config, device)
    elif env_name in (Environments.CRAFTER, Environments.CUBES_3D, Environments.SHAPES_2D):
        return RandomAgent(env, env_name, collect_config, device)
    else:
        raise_env_not_implemented_error(env_name)
