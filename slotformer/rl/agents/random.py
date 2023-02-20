from typing import Optional

import gym
from gym.core import ObsType, ActType

from slotformer.rl.agents.base import BaseAgent
from slotformer.rl.configs.collect_configs import BaseCollectConfig
from slotformer.rl.constants import Environments


class RandomAgent(BaseAgent):
    def __init__(self, env: gym.Env, env_name: Environments, collect_config: BaseCollectConfig,
                 device: Optional[str] = None):
        super().__init__(env, env_name, collect_config, device)

    def get_action(self, obs: ObsType, is_burnin_phase=False) -> ActType:
        return self.action_space.sample()

    def reset(self):
        pass