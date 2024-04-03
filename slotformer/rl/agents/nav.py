import random
from typing import Optional

from gym import Env
import numpy as np
from slotformer.rl.agents.base import BaseAgent
from slotformer.rl.configs.collect_configs import BaseCollectConfig
from slotformer.rl.constants import Environments
from slotformer.rl.envs.shapes import Shapes2d


class AdHocNavigationAgent(BaseAgent):
    # def __init__(self, env: Shapes2d,  random_action_proba=0.5):
    #     self.env = None
    #     self.random_action_proba = random_action_proba
    #     self.set_env(env)
    
    def __init__(self, env: Env, env_name: Environments, collect_config: BaseCollectConfig, device: Optional[str] = None):
        assert env_name == Environments.NAVIGATION_5x5
        super().__init__(env, env_name, collect_config, device)
        self.random_action_proba = collect_config.random_action_proba 
    
    # def set_env(self, env: Shapes2d):
    #     self.env = env
    #     assert not env.embodied_agent
    #     assert env.static_goals
    #     assert len(env.goal_ids) == 1
    #     assert len(env.static_box_ids) == 0
        
    def act(self):
        if random.random() < self.random_action_proba:
            return self.env.action_space.sample()

        box_pos_in_game = [(idx, box_pos) for idx, box_pos in enumerate(self.env.box_pos)
                           if idx not in self.env.goal_ids and idx not in self.env.static_box_ids and box_pos[0] != -1]
        idx, box_pos = random.choice(box_pos_in_game)
        goal_pos = self.env.box_pos[next(iter(self.env.goal_ids))]
        delta = goal_pos - box_pos
        if np.abs(delta)[0] >= np.abs(delta)[1]:
            direction = (int(delta[0] > 0) * 2 - 1, 0)
        else:
            direction = (0, int(delta[1] > 0) * 2 - 1)

        return idx * 4 + self.env.direction2action[direction]
    
    def get_action(self, obs, is_burnin_phase=False):
        return self.act()
    
    def reset(self):
        pass