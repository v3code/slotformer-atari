from pathlib import Path
from typing import Optional

import torch

from slotformer.rl.configs.a3c_configs import get_a3c_config
from slotformer.rl.constants import Environments
from slotformer.rl.models.a3c_baby import NNPolicy
from slotformer.rl.utils import get_torch_device


def get_model(env: Environments,
              weights_path: Optional[str] = None,
              device_option: Optional[str] = None) -> NNPolicy:

    config = get_a3c_config(env)
    model = NNPolicy(**config.__dict__)
    if weights_path:
        model.try_load(weights_path)
    device = get_torch_device(device_option)
    model.to(device)
    return model
