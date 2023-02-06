from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List, Literal

from slotformer.atari.constants import Environments
from slotformer.atari.utils import raise_env_not_implemented_error


@dataclass
class BaseCollectConfig:
    min_burnin: int
    max_burnin: int
    save_path: Path
    a3c_weights: str

    blacklist_paths: List[Path]
    size: Optional[Tuple[int, int]]
    crop: Optional[Tuple[int, int]]
    eps: Optional[float]


def get_collect_config(env: Environments, split: Literal['train', 'test', 'val']) -> BaseCollectConfig:
    assert split in ['train', 'test', 'val']
    blacklist_paths = []
    if split != 'train':
        blacklist_paths.append(Path(f'data/${env}/train'))
        if split == 'test':
            blacklist_paths.append(Path(f'data/${env}/val'))
    if env == Environments.PONG:
        return BaseCollectConfig(
            min_burnin=50,
            max_burnin=100,
            save_path=Path(f'data/pong/{split}'),
            a3c_weights='pretrained/a3c-pong/',
            blacklist_paths=blacklist_paths,
            crop=(32, 195),
            size=(64, 64),
            eps=0.5
        )
    elif env == Environments.SPACE_INVADERS:
        return BaseCollectConfig(
            min_burnin=50,
            max_burnin=300,
            save_path=Path(f'data/spinv/{split}'),
            a3c_weights='pretrained/a3c-spinv/',
            blacklist_paths=blacklist_paths,
            crop=(30, 210),
            size=(64, 64),
            eps=0.5
        )
    else:
        raise_env_not_implemented_error(env)
