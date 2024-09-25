import numpy as np

from typing import Union, Hashable
import random


Seedable = Union[Hashable, random.Random, np.random.RandomState]


def get_random_state(random_state: Seedable = None):
    """
    Return a standard python random state. If the input is already one of these, just return it.
    If not, safely create a random state from the inputted random state.

    :param random_state: The hashable or random state.
    :return: A numpy random state.
    """
    if random_state is None:
        return random.Random()
    elif isinstance(random_state, random.Random):
        return random_state
    else:
        if isinstance(random_state, np.random.RandomState):
            seed = random_state.random()
        else:
            try:
                seed = hash(random_state) % (2 ** 32)
            except TypeError:
                raise ValueError(f"Provided random_state '{random_state}' cannot be used to create a random state.")

        return random.Random(seed)
