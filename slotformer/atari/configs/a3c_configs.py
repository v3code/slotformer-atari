from dataclasses import dataclass

from slotformer.atari.constants import Environments
from slotformer.atari.utils import raise_env_not_implemented_error


@dataclass
class A3CBaseConfig:
    channels: int
    memsize: int
    num_actions: int


def get_a3c_config(env: Environments) -> A3CBaseConfig:
    if env == Environments.SPACE_INVADERS:
        return A3CBaseConfig(
            channels=1,
            memsize=256,
            num_actions=6
        )
    if env == Environments.PONG:
        return A3CBaseConfig(
            channels=1,
            memsize=256,
            num_actions=6
        )
    else:
        raise_env_not_implemented_error(env)
