from typing import Literal

from .basic import BasicFlameIterator
from .common import FlameIterator
from .shuffler import ShufflerFlameIterator

BackendMethod = Literal['shuffler', 'basic']


def create_flame_iterator(
    method: BackendMethod,
    size: tuple[int, int],
    supersamples: int,
) -> FlameIterator:
    match method:
        case 'basic':
            return BasicFlameIterator(size, supersamples)
        case 'shuffler':
            return ShufflerFlameIterator(size, supersamples)
