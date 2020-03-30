import random
from itertools import islice
from typing import Iterator, List, TypeVar

T = TypeVar('T')

def sampling_iter(base_iter: Iterator[T], sample_size: int) -> List[T]:
    """Reservoir Sampling"""
    base_iter = iter(base_iter)
    sampled_items = list(islice(base_iter, sample_size))
    for i, element in enumerate(base_iter, start=sample_size):
        j = random.randrange(i)
        if j < sample_size:
            sampled_items[j] = element

    return sampled_items
