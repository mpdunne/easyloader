import pytest
import random
import numpy as np

from easyloader.utils.random import get_random_state


def test_get_random_state_none():
    rng = get_random_state(None)
    assert isinstance(rng, random.Random)
    results = [rng.random() for _ in range(10)]
    assert len(set(results)) == len(results)

    rng = get_random_state()
    assert isinstance(rng, random.Random)
    results = [rng.random() for _ in range(10)]
    assert len(set(results)) == len(results)


def test_get_random_state_python():
    rng1 = random.Random()
    rng2 = get_random_state(rng1)
    assert rng1 == rng2


def test_get_random_state_numpy():
    results = []
    for _ in range(100):
        rng1 = np.random.RandomState(100)
        rng2 = get_random_state(rng1)
        results.append(rng2.random())

    assert len(set(results)) == 1


@pytest.mark.parametrize(
    'hashable',
    (1, 'sausage', (1, 2, 3))
)
def test_get_random_state_hashable(hashable):
    results = []
    for _ in range(100):
        rng = get_random_state(hashable)
        results.append(rng.random())

    assert len(set(results)) == 1


def test_get_random_state_invalid():
    with pytest.raises(ValueError):
        get_random_state([])
