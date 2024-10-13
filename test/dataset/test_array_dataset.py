import copy
import math
import numpy as np
import pytest
import torch

from copy import deepcopy
from torch.utils.data import DataLoader

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


def test_can_instantiate_with_multiple_arrays(arrays):
    ArrayDataset(arrays)


def test_can_instantiate_with_single_array(arrays):
    ArrayDataset(arrays[0])


def test_values_with_multiple_arrays(arrays):
    ds = ArrayDataset(arrays)
    assert isinstance(ds[0], tuple)
    assert all((dss == array[0]).all() for dss, array in zip(ds[0], arrays))
    assert isinstance(ds[:10], tuple)
    assert all((dss == array[:10]).all() for dss, array in zip(ds[:10], arrays))


def test_values_with_single_arrays(arrays):
    ds = ArrayDataset(arrays[0])
    assert isinstance(ds[0], np.ndarray)
    assert (ds[0] == arrays[0][0]).all()
    assert isinstance(ds[:10], np.ndarray)
    assert (ds[:10] == arrays[0][:10]).all()


def test_process_arrays_unevenly_sized_arrays_throws_error(arrays):
    arrays = [
        np.ones(shape=(100, 3)),
        np.ones(shape=(99, 3)),
    ]
    with pytest.raises(ValueError):
        ArrayDataset(arrays)


def test_process_multiple_arrays_multiple_outputs(arrays):
    ds = ArrayDataset(arrays)
    assert isinstance(ds[0], tuple)
    assert all((dss == array[0]).all() for dss, array in zip(ds[0], arrays))


def test_process_single_arrays_single_output(arrays):
    ds = ArrayDataset(arrays[0])
    assert isinstance(ds[0], np.ndarray)
    assert (ds[0] == arrays[0][0]).all()


@pytest.mark.parametrize(
    'sample_seed', (8675309, None)
)
def test_not_sampled_if_not_asked(arrays, sample_seed):
    original_arrays = copy.deepcopy(arrays)
    data = ArrayDataset(arrays, sample_seed=sample_seed)
    for array, original_array in zip(data._arrays, original_arrays):
        assert (array == original_array).all()


@pytest.mark.parametrize(
    'sample_seed', (8675309, None)
)
def test_sampled_correct_length_and_ordered(arrays, sample_seed):
    # No seed, but a sample fraction.
    original_arrays = copy.deepcopy(arrays)
    data = ArrayDataset(arrays, sample_fraction=0.7, sample_seed=sample_seed)
    for array, original_array in zip(data._arrays, original_arrays):
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


@pytest.mark.parametrize('sample_seed', [*range(10)])
@pytest.mark.parametrize('grain_size', [*range(1, 11)])
def test_sample_grained(arrays, grain_size, sample_seed):
    data = ArrayDataset(arrays, grain_size=grain_size, sample_fraction=0.7, sample_seed=sample_seed)
    assert all(data.index[i + 1] == data.index[i] + 1 for i in range(len(data) - 1) if (i + 1) % grain_size != 0)

    # Size can vary depending on whether the final grain is included in the sample.
    n_original_grains = int(math.ceil(len(arrays[0]) / grain_size))
    n_sampled_grains = int(n_original_grains * 0.7)

    lower = grain_size * (n_sampled_grains - 1)
    upper = grain_size * n_sampled_grains
    assert lower < len(data.index) == len(data) <= upper


def test_shuffle_works(arrays):
    data = ArrayDataset(arrays)
    data.shuffle()
    for array_original, array_new in zip(arrays, data._arrays):
        helper_check_shuffled_arrays_equal(array_original, array_new)
        assert array_original.shape == array_new.shape
        assert not (array_original == array_new).all()


def test_shuffle_consistent(arrays):
    data = ArrayDataset(arrays, shuffle_seed=8675309)
    data.shuffle()
    array_sets_first = deepcopy(data._arrays)
    for _ in range(4):
        data = ArrayDataset(arrays, shuffle_seed=8675309)
        data.shuffle()
        assert all((arr == arr_first).all() for arr, arr_first in zip(data._arrays, array_sets_first))


def test_shuffle_changes_index(arrays):
    data = ArrayDataset(arrays)
    index_orig = data.index.copy()
    data.shuffle()
    assert data.index != index_orig
    assert sorted(data.index) == sorted(index_orig)


@pytest.mark.parametrize('grain_size', [*range(1, 11)])
def test_shuffle_grained(arrays, grain_size):
    data = ArrayDataset(arrays, grain_size=grain_size)

    # Make a note. Should start unshuffled.
    assert all(data.grain_index[i + 1] == data.grain_index[i] + 1 for i in range(len(data.grain_index) - 1))
    grain_index_orig = data.grain_index.copy()
    index_orig = data.index.copy()

    # Check that the grain index is shuffled. Do it a few times.
    for _ in range(5):
        data.shuffle()
        assert not all(data.grain_index[i + 1] == data.grain_index[i] + 1 for i in range(len(data.grain_index) - 1))
        assert sorted(grain_index_orig) == sorted(data.grain_index)

        # Check that the index is shuffled also.
        assert sorted(index_orig) == sorted(data.index)
        assert not all(data.index[i + 1] == data.index[i] + 1 for i in range(len(data.index) - 1))

        # Check that the index is as expected.
        expected = [ix for gix in data.grain_index
                    for ix in range(gix * grain_size, (gix + 1) * grain_size) if ix < len(arrays[0])]
        assert expected == data.index


def test_ids_unspecified(arrays):
    data = ArrayDataset(arrays)
    assert len(data.ids) == len(arrays[0])


def test_ids_specified(arrays):
    ids = [f'entry{i}' for i in range(len(arrays[0]))]
    data = ArrayDataset(arrays, ids=ids)
    assert ids == data.ids


def test_ids_specified_too_short(arrays):
    ids = [f'entry{i}' for i in range(len(arrays[0]) - 1)]
    with pytest.raises(ValueError):
        ArrayDataset(arrays, ids=ids)


def test_ids_specified_wrong_type(arrays):
    ids = 4
    with pytest.raises(TypeError):
        ArrayDataset(arrays, ids=ids)


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


def test_works_with_torch_dataloader_single_array(arrays):
    ds = ArrayDataset(arrays[0])
    dl = DataLoader(ds, batch_size=10)
    entry = next(iter(dl))
    assert isinstance(entry, torch.Tensor)
    expected = arrays[0][:10]
    assert (entry.numpy() == expected).all()


def test_works_with_torch_dataloader_multi_array(arrays):
    ds = ArrayDataset(arrays)
    dl = DataLoader(ds, batch_size=10)
    entries = next(iter(dl))
    assert isinstance(entries, list)
    expected = tuple([arr[:10] for arr in arrays])
    assert all((entry.numpy() == array).all() for entry, array in zip(entries, expected))


def test_shuffle_works_with_torch_dataloader(arrays):
    ds = ArrayDataset(arrays)
    dl = DataLoader(ds, shuffle=True, batch_size=1000000)
    all_entries = next(iter(dl))
    assert all(not (array == entry.numpy()).all() for array, entry in zip(arrays, all_entries))
    for array, entry in zip(arrays, all_entries):
        helper_check_shuffled_arrays_equal(array, entry.numpy())
