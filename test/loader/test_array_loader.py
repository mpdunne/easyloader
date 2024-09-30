import numpy as np
import pytest

from copy import deepcopy
from unittest.mock import patch

from easyloader.loader.array import ArrayDataLoader


@pytest.fixture(scope='session')
def arrays():
    n_entries = 1000

    # Use broadcasting to add consecutive numbers to each layer, for testing sampling & shuffling
    arrays = [
        np.arange(n_entries).reshape(-1, 1, 1, 1) * np.ones(shape=(1, 3, 16, 16)), # Shape (1000, 3, 16, 16)
        np.arange(n_entries).reshape(-1, 1) * np.ones(shape=(1, 5)),  # Shape (1000, 5)
        np.arange(n_entries).reshape(-1, 1, 1) * np.ones(shape=(1, 10, 10)),  # Shape (1000, 10, 10)
    ]

    return arrays


def helper_check_shuffled_arrays_equal(arr1: np.ndarray, arr2: np.ndarray):
    assert arr1.shape == arr2.shape
    sorted_arr1 = np.sort(arr1, axis=0)
    sorted_arr2 = np.sort(arr2, axis=0)
    assert (sorted_arr1 == sorted_arr2).all()


def test_can_instantiate(arrays):
    ArrayDataLoader(arrays)


def test_args_passed_to_data_class(arrays):
    with patch('easyloader.loader.array.ArrayData') as MockArrayData:
        sample_fraction = 0.7
        sample_seed = 8675309
        shuffle_seed = 5318008
        ids = [*range(len(arrays[0]))]
        ArrayDataLoader(arrays, ids=ids, shuffle_seed=shuffle_seed, sample_fraction=sample_fraction, sample_seed=sample_seed)
        MockArrayData.assert_called_once_with(arrays, ids=ids, shuffle_seed=shuffle_seed,
                                              sample_fraction=sample_fraction, sample_seed=sample_seed)


def test_can_iterate(arrays):
    dl = ArrayDataLoader(arrays)
    for _ in dl:
        pass


def helper_iterate_all_and_concatenate(dl, expected_batch_size):
    batch_sets = []
    for i, batch in enumerate(dl):
        assert all((len(b) == expected_batch_size) for b in batch) or i == len(dl) - 1
        batch_sets.append(batch)

    batches_joined = [np.concatenate(bs) for bs in zip(*batch_sets)]
    return batches_joined


@pytest.mark.parametrize('batch_size', (1, 10, 11, 100))
def test_iterated_values_correct(arrays, batch_size):
    dl = ArrayDataLoader(arrays, batch_size=batch_size)
    batches_joined = helper_iterate_all_and_concatenate(dl, batch_size)
    assert all((b == b_expected).all() for b, b_expected in zip(batches_joined, arrays))


def test_ids_set(arrays):
    dl = ArrayDataLoader(arrays)
    assert len(dl.ids) == len(arrays[0])
    ids = [f'i_{i}' for i in range(len(arrays[0]))]
    dl = ArrayDataLoader(arrays, ids=ids)
    assert dl.ids == ids


def test_shuffle_works(arrays):
    batch_size = 11
    dl = ArrayDataLoader(arrays, batch_size=batch_size, shuffle=True)
    batches_joined = helper_iterate_all_and_concatenate(dl, batch_size)
    for array_in, array_out in zip(arrays, batches_joined):
        helper_check_shuffled_arrays_equal(array_in, array_out)


def test_shuffle_consistent(arrays):
    batch_size = 11
    dl = ArrayDataLoader(arrays, batch_size=batch_size, shuffle=True, shuffle_seed=8675309)
    batches_joined1 = deepcopy(helper_iterate_all_and_concatenate(dl, batch_size))
    dl = ArrayDataLoader(arrays, batch_size=batch_size, shuffle=True, shuffle_seed=8675309)
    batches_joined2 = deepcopy(helper_iterate_all_and_concatenate(dl, batch_size))
    for b1, b2 in zip(batches_joined1, batches_joined2):
        assert (b1 == b2).all()


def test_sample_works(arrays):
    batch_size = 11
    dl = ArrayDataLoader(arrays, batch_size=batch_size, shuffle=True, sample_fraction=0.7)
    batches_joined = helper_iterate_all_and_concatenate(dl, batch_size)
    for array_in, array_out in zip(arrays, batches_joined):
        assert len(array_out) == len(array_in) * 0.7


def test_sample_consistent(arrays):
    batch_size = 11
    dl = ArrayDataLoader(arrays, batch_size=batch_size, sample_fraction=0.7, sample_seed=8675309)
    batches_joined1 = deepcopy(helper_iterate_all_and_concatenate(dl, batch_size))
    dl = ArrayDataLoader(arrays, batch_size=batch_size, sample_fraction=0.7, sample_seed=8675309)
    batches_joined2 = deepcopy(helper_iterate_all_and_concatenate(dl, batch_size))
    for b1, b2 in zip(batches_joined1, batches_joined2):
        assert (b1 == b2).all()



def test_len_is_n_batches(arrays):
    dl = ArrayDataLoader(arrays, batch_size=99)
    assert len(dl) == 11
