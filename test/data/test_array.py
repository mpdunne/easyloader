import copy
import numpy as np
import pytest

from copy import deepcopy

from easyloader.data.array import ArrayData


@pytest.fixture(scope='session')
def arrays():
    n_entries = 1000

    # Use broadcasting to add consecutive numbers to each layer, for testing sampling & shuffling
    arrays = [
        np.arange(1000).reshape(-1, 1, 1, 1) * np.ones(shape=(1, 3, 16, 16)), # Shape (1000, 3, 16, 16)
        np.arange(1000).reshape(-1, 1) * np.ones(shape=(1, 5)),  # Shape (1000, 5)
        np.arange(1000).reshape(-1, 1, 1) * np.ones(shape=(1, 10, 10)),  # Shape (1000, 10, 10)
    ]

    return arrays


def helper_check_shuffled_arrays_equal(arr1: np.ndarray, arr2: np.ndarray):
    assert arr1.shape == arr2.shape
    sorted_arr1 = np.sort(arr1, axis=0)
    sorted_arr2 = np.sort(arr2, axis=0)
    assert (sorted_arr1 == sorted_arr2).all()


def test_can_instantiate(arrays):
    ArrayData(arrays)


def test_uneven_sized_arrays_throws_error():
    arrays = [
        np.ones(shape=(100, 3)),
        np.ones(shape=(99, 3)),
    ]
    with pytest.raises(ValueError):
        ArrayData(arrays)


@pytest.mark.parametrize(
    'sample_seed', (8675309, None)
)
def test_not_sampled_if_not_asked(arrays, sample_seed):
    original_arrays = copy.deepcopy(arrays)
    data = ArrayData(arrays, sample_seed=sample_seed)
    for array, original_array in zip(data.arrays, original_arrays):
        assert (array == original_array).all()


@pytest.mark.parametrize(
    'sample_seed', (8675309, None)
)
def test_sampled_correct_length_and_ordered(arrays, sample_seed):
    # No seed, but a sample fraction.
    original_arrays = copy.deepcopy(arrays)
    data = ArrayData(arrays, sample_fraction=0.7, sample_seed=sample_seed)
    for array, original_array in zip(data.arrays, original_arrays):
        assert array.shape[1:] == original_array.shape[1:]
        assert len(array) == len(data) == len(original_array) * 0.7
        assert all((array[i] <= array[i + 1]).all() for i in range(len(array) - 1))


@pytest.mark.parametrize(
    'seed,consistent',
    (
            (1, True),
            ('sausage', True),
            ((1, 2, 3), True),
            (None, False),
    )
)
def test_sampled_consistent(arrays, seed, consistent):
    data = ArrayData(arrays, sample_fraction=0.7, sample_seed=seed)
    ix_sets = [list(data.index)]
    for _ in range(4):
        data = ArrayData(arrays, sample_fraction=0.7, sample_seed=seed)
        ixs = list(data.index)
        assert all((ixs == ixsc) == consistent for ixsc in ix_sets)
        ix_sets.append(ixs)


def test_shuffle_works(arrays):
    data = ArrayData(arrays)
    data.shuffle()
    for array_original, array_new in zip(arrays, data.arrays):
        helper_check_shuffled_arrays_equal(array_original, array_new)
        assert array_original.shape == array_new.shape
        assert not (array_original == array_new).all()


def test_shuffle_consistent(arrays):
    data = ArrayData(arrays, shuffle_seed=8675309)
    data.shuffle()
    array_sets_first = deepcopy(data.arrays)
    for _ in range(4):
        data = ArrayData(arrays, shuffle_seed=8675309)
        data.shuffle()
        assert all((arr == arr_first).all() for arr, arr_first in zip(data.arrays, array_sets_first))


def test_shuffle_changes_index(arrays):
    data = ArrayData(arrays)
    index_orig = data.index.copy()
    data.shuffle()
    assert data.index != index_orig
    assert sorted(data.index) == sorted(index_orig)


def test_ids_specified(arrays):
    ids = [f'entry{i}' for i in range(len(arrays[0]))]
    data = ArrayData(arrays, ids=ids)
    assert ids == data.ids


def test_ids_unspecified(arrays):
    data = ArrayData(arrays)
    assert len(data.ids) == len(arrays[0])


def test_shuffle_changes_ids(arrays):
    ids = [f'entry{i}' for i in range(len(arrays[0]))]
    data = ArrayData(arrays, ids=ids)
    ids_orig = data.ids.copy()
    data.shuffle()
    assert data.ids != ids_orig
    assert sorted(data.ids) == sorted(ids_orig)
