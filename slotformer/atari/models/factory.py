from pathlib import Path
from typing import Optional

from slotformer.atari.configs.a3c_configs import get_a3c_config
from slotformer.atari.constants import Environments
from slotformer.atari.models.a3c_baby import NNPolicy


def get_model(env: Environments,
              weights_path: Optional[str] = None) -> NNPolicy:

    config = get_a3c_config(env)
    model = NNPolicy(**config.__dict__)
    if weights_path:
        model.try_load(weights_path)
    return model
