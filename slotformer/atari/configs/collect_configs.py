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
    if env == Environments.PONG:
        if split != 'train':
            blacklist_paths.append(Path('data/pong/train'))
            if split == 'test':
                blacklist_paths.append(Path('data/pong/val'))
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
    else:
        raise_env_not_implemented_error(env)