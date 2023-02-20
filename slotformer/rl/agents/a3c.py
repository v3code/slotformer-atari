from typing import Optional

import gym
import numpy as np
import torch
from gym.core import ActType, ObsType
from skimage.transform import resize

import torch.functional as F

from slotformer.rl.agents.base import BaseAgent
from slotformer.rl.configs.a3c_configs import get_a3c_config
from slotformer.rl.configs.collect_configs import BaseCollectConfig
from slotformer.rl.models.factory import get_model


class A3CAtariAgent(BaseAgent):


    def __init__(self,
                 env: gym.Env,
                 env_name: str,
                 collect_config: BaseCollectConfig,
                 device: Optional[str] = None):
        super().__init__(env, env_name, collect_config, device)
        self.a3c_config = get_a3c_config(env_name)
        self.model = get_model(env_name, collect_config.a3c_weights, device)
        self.hx = self.reset_rnn_state()

    def select_action(self, state: torch.Tensor):
        # select an action using either an epsilon greedy or softmax policy
        value, logit, hx = self.model((state.view(1, 1, 80, 80), self.hx))
        logp = F.log_softmax(logit, dim=-1)
        eps = self.collect_config.eps

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

    def preprocess_state(self, state: np.ndarray):
        state = resize(state[35:195].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255
        return torch.tensor(state, device=self.torch_device)

    def reset_rnn_state(self):
        # reset the hidden state of an rnn
        return torch.zeros(1, self.a3c_config.memsize, device=self.torch_device)

    def get_action(self, obs: ObsType, burnin_phase=False) -> ActType:
        if not burnin_phase:
            return self.action_space.sample()

        preprocessed = self.preprocess_state(obs)
        return self.select_action(preprocessed)

    def reset(self):
        self.hx = self.reset_rnn_state()

