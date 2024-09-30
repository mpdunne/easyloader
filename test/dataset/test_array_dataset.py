import numpy as np
import pytest

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


def test_args_passed_to_data_class(arrays):
    with patch('easyloader.datasets.array.ArrayData') as MockArrayData:
        sample_fraction = 0.7
        sample_seed = 8675309
        ids = [f'i_{i}' for i in range(len(arrays[0]))]
        ArrayDataset(arrays, sample_fraction=sample_fraction, sample_seed=sample_seed, ids=ids)
        MockArrayData.assert_called_once_with(arrays, ids=ids,
                                              sample_fraction=sample_fraction, sample_seed=sample_seed,)


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
