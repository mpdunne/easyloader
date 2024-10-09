import copy
import numpy as np
import pytest

from copy import deepcopy
from torch.utils.data import DataLoader
from unittest.mock import patch

from easyloader.dataset.array import ArrayDataset


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
    ArrayDataset(arrays)


def test_uneven_sized_arrays_throws_error():
    arrays = [
        np.ones(shape=(100, 3)),
        np.ones(shape=(99, 3)),
    ]
    with pytest.raises(ValueError):
        ArrayDataset(arrays)


@pytest.mark.parametrize(
    'sample_seed', (8675309, None)
)
def test_not_sampled_if_not_asked(arrays, sample_seed):
    original_arrays = copy.deepcopy(arrays)
    data = ArrayDataset(arrays, sample_seed=sample_seed)
    for array, original_array in zip(data.arrays, original_arrays):
        assert (array == original_array).all()


@pytest.mark.parametrize(
    'sample_seed', (8675309, None)
)
def test_sampled_correct_length_and_ordered(arrays, sample_seed):
    # No seed, but a sample fraction.
    original_arrays = copy.deepcopy(arrays)
    data = ArrayDataset(arrays, sample_fraction=0.7, sample_seed=sample_seed)
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
    data = ArrayDataset(arrays, sample_fraction=0.7, sample_seed=seed)
    ix_sets = [list(data.index)]
    for _ in range(4):
        data = ArrayDataset(arrays, sample_fraction=0.7, sample_seed=seed)
        ixs = list(data.index)
        assert all((ixs == ixsc) == consistent for ixsc in ix_sets)
        ix_sets.append(ixs)


def test_shuffle_works(arrays):
    data = ArrayDataset(arrays)
    data.shuffle()
    for array_original, array_new in zip(arrays, data.arrays):
        helper_check_shuffled_arrays_equal(array_original, array_new)
        assert array_original.shape == array_new.shape
        assert not (array_original == array_new).all()


def test_shuffle_consistent(arrays):
    data = ArrayDataset(arrays, shuffle_seed=8675309)
    data.shuffle()
    array_sets_first = deepcopy(data.arrays)
    for _ in range(4):
        data = ArrayDataset(arrays, shuffle_seed=8675309)
        data.shuffle()
        assert all((arr == arr_first).all() for arr, arr_first in zip(data.arrays, array_sets_first))


def test_shuffle_changes_index(arrays):
    data = ArrayDataset(arrays)
    index_orig = data.index.copy()
    data.shuffle()
    assert data.index != index_orig
    assert sorted(data.index) == sorted(index_orig)


def test_ids_specified(arrays):
    ids = [f'entry{i}' for i in range(len(arrays[0]))]
    data = ArrayDataset(arrays, ids=ids)
    assert ids == data.ids


def test_ids_unspecified(arrays):
    data = ArrayDataset(arrays)
    assert len(data.ids) == len(arrays[0])


def test_shuffle_changes_ids(arrays):
    ids = [f'entry{i}' for i in range(len(arrays[0]))]
    data = ArrayDataset(arrays, ids=ids)
    ids_orig = data.ids.copy()
    data.shuffle()
    assert data.ids != ids_orig
    assert sorted(data.ids) == sorted(ids_orig)


def test_can_get_item(arrays):
    ds = ArrayDataset(arrays)
    entries = ds[10]
    assert isinstance(entries, tuple)
    assert all(entry.shape == array.shape[1:] for entry, array in zip(entries, arrays))
    assert all((entry == array[10]).all() for entry, array in zip(entries, arrays))


def test_cant_get_out_of_range_item(arrays):
    with pytest.raises(IndexError):
        ds = ArrayDataset(arrays)
        ds[1000000]


def test_can_be_inputted_to_torch_dataloader(arrays):
    ds = ArrayDataset(arrays)
    DataLoader(ds)


def test_slice_works(arrays):
    ds = ArrayDataset(arrays)
    slices = ds[:10]
    assert all(len(s) == 10 for s in slices)
    assert all((s == arr[:10]).all() for s, arr in zip(slices, arrays))


def test_slice_works_sampled(arrays):
    ds = ArrayDataset(arrays, sample_fraction=0.3, sample_seed=8675309)
    slices = ds[:10]
    assert all(len(s) == 10 for s in slices)
    assert all(not (s == arr[:10]).all() for s, arr in zip(slices, arrays))


def test_works_with_torch_dataloader(arrays):
    ds = ArrayDataset(arrays)
    dl = DataLoader(ds, batch_size=10)
    entries = next(iter(dl))
    expected = tuple([arr[:10] for arr in arrays])
    assert isinstance(expected, tuple)
    assert all((entry.numpy() == array).all() for entry, array in zip(entries, expected))


def test_shuffle_works_with_torch_dataloader(arrays):
    ds = ArrayDataset(arrays)
    dl = DataLoader(ds, shuffle=True, batch_size=1000000)
    all_entries = next(iter(dl))
    assert all(not (array == entry.numpy()).all() for array, entry in zip(arrays, all_entries))
    for array, entry in zip(arrays, all_entries):
        helper_check_shuffled_arrays_equal(array, entry.numpy())
