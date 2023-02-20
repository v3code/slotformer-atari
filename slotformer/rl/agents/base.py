from abc import ABC, abstractmethod
from typing import Optional

import gym
from gym.core import ActType, ObsType

from slotformer.rl.configs.collect_configs import BaseCollectConfig
from slotformer.rl.constants import Environments
from slotformer.rl.utils import get_torch_device


class BaseAgent(ABC):

    def __init__(self,
                 env: gym.Env,
                 env_name: Environments,
                 collect_config: BaseCollectConfig,
                 device: Optional[str] = None):
        self.env = env
        self.action_space = env.action_space
        self.torch_device = get_torch_device(device)
        self.collect_config = collect_config
        self.env_name = env_name
    @abstractmethod
    def get_action(self, obs: ObsType, is_burnin_phase=False) -> ActType:
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()