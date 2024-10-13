import h5py
import os
import pytest
import numpy as np
import tempfile
import torch

from easyloader.dataset.h5 import H5Dataset
from torch.utils.data import DataLoader
from unittest.mock import patch


@pytest.fixture
def h5_file(scope='module'):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test.h5")
        data_length = 100

        # Open the HDF5 file in write mode
        with h5py.File(file_path, 'w') as h5file:
            # First dimension is consistent across all datasets, other dimensions vary
            h5file.create_dataset("key_1", data=np.random.rand(data_length, 10))
            h5file.create_dataset("key_2", data=np.random.rand(data_length, 5, 3))
            h5file.create_dataset("key_3", data=np.random.rand(data_length, 20))
            h5file.create_dataset("id_key", data=np.array([*range(data_length)]))

        # Yield the file path for use in tests
        yield file_path


@pytest.fixture
def h5_file_jagged():
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test_data_consistent_first_dim.h5")

        n_entries = 1000

        # Open the HDF5 file in write mode
        with h5py.File(file_path, 'w') as h5file:
            # First dimension is consistent across all datasets, other dimensions vary
            h5file.create_dataset("key_1", data=np.arange(n_entries).reshape(-1, 1, 1, 1) * np.ones(shape=(1, 3, 16, 16)))
            h5file.create_dataset("key_2", data=np.arange(n_entries).reshape(-1, 1) * np.ones(shape=(1, 5)))
            h5file.create_dataset("key_3", data=np.arange(n_entries).reshape(-1, 1, 1) * np.ones(shape=(1, 10, 10)))
            h5file.create_dataset("id_key", data=np.random.rand(n_entries, ))

        np.arange(n_entries).reshape(-1, 1, 1, 1) * np.ones(shape=(1, 3, 16, 16)), # Shape (1000, 3, 16, 16)
        np.arange(n_entries).reshape(-1, 1) * np.ones(shape=(1, 5)),  # Shape (1000, 5)
        np.arange(n_entries).reshape(-1, 1, 1) * np.ones(shape=(1, 10, 10)),  # Shape (1000, 10, 10)

        # Yield the file path for use in tests
        yield file_path


def test_can_instantiate_with_single_keys(h5_file):
    keys = 'key_1'
    H5Dataset(h5_file, keys=keys)


def test_can_instantiate_with_multiple_keys(h5_file):
    keys = ['key_1', 'key_2']
    H5Dataset(h5_file, keys=keys)


def test_single_key_gives_single_output(h5_file):
    keys = 'key_1'
    ds = H5Dataset(h5_file, keys=keys)
    h5 = h5py.File(h5_file)
    assert isinstance(ds[0], np.ndarray)
    assert ds[0].shape == h5['key_1'].shape[1:]
    assert isinstance(ds[:10], np.ndarray)
    assert ds[:10].shape == (10, *h5['key_1'].shape[1:])


@pytest.mark.parametrize('keys', (
        ['key_1', 'key_2'],
        ('key_1', 'key_2'),
))
def test_multiple_keys_give_multiple_outputs(h5_file, keys):
    ds = H5Dataset(h5_file, keys=keys)
    h5 = h5py.File(h5_file)

    assert isinstance(ds[0], tuple)
    assert len(ds[0]) == 2
    for i, key in enumerate(keys):
        assert isinstance(ds[0][i], np.ndarray)
        assert ds[0][i].shape == h5[key].shape[1:]

    assert isinstance(ds[:10], tuple)
    assert len(ds[:10]) == 2
    for i, key in enumerate(keys):
        assert isinstance(ds[:10][i], np.ndarray)
        assert ds[:10][i].shape == (10, *h5[key].shape[1:])


def test_cant_instantiate_without_keys(h5_file):
    with pytest.raises(TypeError):
        _ = H5Dataset(h5_file)

    with pytest.raises(ValueError):
        _ = H5Dataset(h5_file, keys=[])


def test_missing_keys_throws_error(h5_file):
    with pytest.raises(ValueError):
        _ = H5Dataset(h5_file, keys=['sausage'])


@pytest.mark.parametrize(
    'sample_seed', (8675309, None)
)
def test_not_sampled_if_not_asked(h5_file, sample_seed):
    keys = ['key_1', 'key_2']
    data = H5Dataset(h5_file, keys=keys, sample_seed=sample_seed)
    h5 = h5py.File(h5_file)
    assert data.index == [*range(len(h5[keys[0]]))]


@pytest.mark.parametrize(
    'sample_seed', (8675309, None)
)
def test_sampled_correct_length_and_ordered(h5_file, sample_seed):
    keys = ['key_1', 'key_2']
    data = H5Dataset(h5_file, keys=keys, sample_seed=sample_seed, sample_fraction=0.7)
    h5 = h5py.File(h5_file)
    assert len(data.index) == len(h5[keys[0]]) * 0.7
    assert all(data.index[i] < data.index[i + 1] for i in range(len(data.index) - 1))


@pytest.mark.parametrize(
    'seed,consistent',
    (
            (1, True),
            ('sausage', True),
            ((1, 2, 3), True),
            (None, False),
    )
)
def test_sampled_consistent(h5_file, seed, consistent):
    keys = ['key_1', 'key_2']
    data = H5Dataset(h5_file, keys=keys, sample_seed=seed, sample_fraction=0.7)
    ix_sets = [list(data.index)]
    for _ in range(4):
        data = H5Dataset(h5_file, keys=keys, sample_seed=seed, sample_fraction=0.7)
        ixs = list(data.index)
        assert all((ixs == ixsc) == consistent for ixsc in ix_sets)
        ix_sets.append(ixs)


def test_shuffle_works(h5_file):
    keys = ['key_1', 'key_2']
    data = H5Dataset(h5_file, keys=keys, shuffle_seed=8675309)
    h5_orig = h5py.File(h5_file)
    unshuffled_index = [*range(len(h5_orig[keys[0]]))]
    assert (data.index == unshuffled_index)
    data.shuffle()
    assert not (data.index == unshuffled_index)
    assert len(data) == len(unshuffled_index)


def test_shuffle_consistent(h5_file):
    keys = ['key_1', 'key_2']
    data = H5Dataset(h5_file, keys=keys, shuffle_seed=8675309)
    h5_orig = h5py.File(h5_file)
    ix_sets = [list(data.index)]
    for _ in range(4):
        data = H5Dataset(h5_file, keys=keys, shuffle_seed=8675309)
        ixs = list(data.index)
        assert all((ixs == ixsc) for ixsc in ix_sets)


def test_shuffle_changes_index(h5_file):
    keys = ['key_1', 'key_2']
    data = H5Dataset(h5_file, keys=keys, shuffle_seed=8675309)
    index_orig = data.index.copy()
    data.shuffle()
    assert data.index != index_orig
    assert sorted(data.index) == sorted(index_orig)


def test_id_key_unspecified(h5_file):
    keys = ['key_1', 'key_2']
    id_key = 'id_key'
    data = H5Dataset(h5_file, keys=keys, ids=id_key, shuffle_seed=8675309)
    assert data.ids == [*range(len(data))]


def test_id_key_specified(h5_file):
    keys = ['key_1', 'key_2']
    id_key = 'id_key'
    data = H5Dataset(h5_file, keys=keys, ids=id_key, shuffle_seed=8675309)
    h5 = h5py.File(h5_file)
    assert (data.ids == h5[id_key][:]).all()


def test_id_key_specified_bad(h5_file):
    keys = ['key_1', 'key_2']
    id_key = 'monkey'
    with pytest.raises(ValueError):
        H5Dataset(h5_file, keys=keys, ids=id_key, shuffle_seed=8675309)


def test_ids_specified_wrong_type(h5_file):
    keys = ['key_1', 'key_2']
    id_key = 4
    with pytest.raises(TypeError):
        H5Dataset(h5_file, keys=keys, ids=id_key, shuffle_seed=8675309)


def test_ids_specified_as_list(h5_file):
    keys = ['key_1', 'key_2']
    h5 = h5py.File(h5_file)
    ids = [f'ix_{i}' for i in range(len(h5[keys[0]]))]
    data = H5Dataset(h5_file, keys=keys, ids=ids, shuffle_seed=8675309)
    assert (data.ids == ids)


def test_ids_specified_as_list_wrong_size(h5_file):
    keys = ['key_1', 'key_2']
    h5 = h5py.File(h5_file)
    ids = [f'ix_{i}' for i in range(len(h5[keys[0]]) - 1)]
    with pytest.raises(ValueError):
        H5Dataset(h5_file, keys=keys, ids=ids, shuffle_seed=8675309)


def test_shuffle_changes_ids(h5_file):
    keys = ['key_1', 'key_2']
    id_key = 'id_key'
    data = H5Dataset(h5_file, keys=keys, ids=id_key, shuffle_seed=8675309)
    data.shuffle()
    assert list(data.ids) != sorted(list(data.ids))


@pytest.mark.parametrize('grain_size', (1, 2, 5, 10, 100, 100000))
def test_shuffle_grained(h5_file, grain_size):
    grain_size = 20
    data = H5Dataset(h5_file, keys=keys, grain_size=grain_size)
    data.shuffle()
    assert all(data.index[i + 1] == data.index[i] + 1 for i in range(len(data) - 1) if (i + 1) % grain_size != 0)
    assert not all(data.index[i + 1] == data.index[i] + 1 for i in range(len(data) - 1) if (i + 1) % grain_size == 0)


def test_can_get_item(h5_file):
    keys = ['key_1', 'key_2']
    ds = H5Dataset(h5_file, keys=keys)
    h5 = h5py.File(h5_file)
    entries = ds[5]
    assert isinstance(entries, tuple)
    assert (entries[0] == h5[keys[0]][5]).all()
    assert (entries[1] == h5[keys[1]][5]).all()


def test_cant_get_out_of_range_item(h5_file):
    with pytest.raises(IndexError):
        keys = ['key_1', 'key_2']
        ds = H5Dataset(h5_file, keys=keys)
        ds[1000000]


def test_can_be_inputted_to_torch_dataloader(h5_file):
    keys = ['key_1', 'key_2']
    ds = H5Dataset(h5_file, keys=keys)
    DataLoader(ds)


def test_slice_works(h5_file):
    keys = ['key_1', 'key_2']
    ds = H5Dataset(h5_file, keys=keys)
    slices = ds[:10]
    h5 = h5py.File(h5_file)
    assert all(len(s) == 10 for s in slices)
    assert all((s == h5[k][:10]).all() for s, k in zip(slices, keys))


def test_slice_works_sampled(h5_file):
    keys = ['key_1', 'key_2']
    ds = H5Dataset(h5_file, keys=keys, sample_fraction=0.3, sample_seed=8675309)
    slices = ds[:10]
    h5 = h5py.File(h5_file)
    assert all(len(s) == 10 for s in slices)
    assert all(not (s == h5[k][:10]).all() for s, k in zip(slices, keys))


def test_works_with_torch_dataloader_single_key(h5_file):
    keys = 'key_1'
    ds = H5Dataset(h5_file, keys=keys)
    dl = DataLoader(ds, batch_size=10)
    h5 = h5py.File(h5_file)
    entries = next(iter(dl))
    expected = h5['key_1'][:10]
    assert len(entries) == 10
    assert isinstance(entries, torch.Tensor)
    assert (entries.numpy() == expected).all()


def test_works_with_torch_dataloader_multi_key(h5_file):
    keys = ['key_1', 'key_2']
    ds = H5Dataset(h5_file, keys=keys)
    dl = DataLoader(ds, batch_size=10)
    h5 = h5py.File(h5_file)
    entries = next(iter(dl))
    expected = tuple([h5[k][:10] for k in keys])
    assert all(len(entry) == 10 for entry in entries)
    assert isinstance(entries, list)
    assert all((entry.numpy() == array).all() for entry, array in zip(entries, expected))


def test_shuffle_works_with_torch_dataloader(h5_file):
    keys = ['key_1', 'key_2']
    ds = H5Dataset(h5_file, keys=keys)
    dl = DataLoader(ds, shuffle=True, batch_size=1000000)
    h5 = h5py.File(h5_file)
    all_entries = next(iter(dl))
    assert all(not (entry.numpy() == h5[k][:]).all().all() for entry, k in zip(all_entries, keys))
