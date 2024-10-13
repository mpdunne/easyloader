import pytest
import random

from easyloader.utils.slices import merge_neighbouring_slices


@pytest.mark.parametrize('seed', range(10))
@pytest.mark.parametrize('sample_fraction', (0.6, 0.7, 0.9, 1.0))
def test_merge_neighbouring_slices(seed, sample_fraction):
    rng = random.Random(seed)
    slice_size, n_slices = 10, 100
    slices = [slice(i * slice_size, (i + 1) * slice_size) for i in range(n_slices)]

    # Sample the slices (but keep order) and expand them.
    slices = rng.sample(slices, k=int(sample_fraction * len(slices)))
    slices = sorted(slices, key=lambda s: s.start)
    slices_expanded = [ix for sl in slices for ix in range(sl.start, sl.stop)]

    # Now join them. At least two must be neighbouring.
    slices_joined = merge_neighbouring_slices(slices)
    slices_joined_expanded = [ix for sl in slices_joined for ix in range(sl.start, sl.stop)]
    assert slices_expanded == slices_joined_expanded
    assert len(slices_joined) < len(slices)

