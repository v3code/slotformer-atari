from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List, Literal

from slotformer.rl.constants import Environments
from slotformer.rl.utils import raise_env_not_implemented_error


@dataclass
class BaseCollectConfig:
    min_burnin: int
    max_burnin: int
    save_path: Path

    blacklist_paths: List[Path]
    a3c_weights: Optional[str] = None
    size: Optional[Tuple[int, int]] = None
    crop: Optional[Tuple[int, int]] = None
    eps: Optional[float] = None
    
    
@dataclass
class NavAgentCollectConfig(BaseCollectConfig):
    random_action_proba: float = 0.5


def get_collect_config(env: Environments, split: Literal['train', 'test', 'val']) -> BaseCollectConfig:
    assert split in ['train', 'test', 'val']
    blacklist_paths = []
    if split != 'train' and env != Environments.CRAFTER:
        blacklist_paths.append(Path(f'data/{env}/train'))
        if split == 'test':
            blacklist_paths.append(Path(f'data/{env}/val'))
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
    elif env == Environments.NAVIGATION_5x5:
        return NavAgentCollectConfig(
            min_burnin=0,
            max_burnin=0,
            save_path=Path(f'data/nav5x5/{split}'),
            blacklist_paths=[],
            random_action_proba=0.5
        )
        
    elif env == Environments.PUSHING_5x5:
        return BaseCollectConfig(
            min_burnin=0,
            max_burnin=0,
            save_path=Path(f'data/push5x5/{split}'),
            blacklist_paths=[],
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
    elif env == Environments.CRAFTER:
        return BaseCollectConfig(
            min_burnin=10,
            max_burnin=300,
            save_path=Path(f'data/crafter/{split}'),
            blacklist_paths=blacklist_paths,
            size=(64, 64),
        )
    elif env == Environments.SHAPES_2D:
        return BaseCollectConfig(
            min_burnin=0,
            max_burnin=0,
            save_path=Path(f'data/shapes/{split}'),
            blacklist_paths=blacklist_paths,
            size=(64, 64),
        )
    elif env == Environments.CUBES_3D:
        return BaseCollectConfig(
            min_burnin=0,
            max_burnin=0,
            save_path=Path(f'data/cubes/{split}'),
            blacklist_paths=blacklist_paths,
            size=(64, 64),
        )
    else:
        raise_env_not_implemented_error(env)
