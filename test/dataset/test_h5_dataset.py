import h5py
import os
import pytest
import numpy as np
import tempfile

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


def test_can_instantiate(h5_file):
    keys = ['key_1', 'key_2']
    H5Dataset(h5_file, keys=keys)


def test_args_passed_to_data_class(h5_file):
    with patch('easyloader.dataset.h5.H5Data') as MockH5Data:
        sample_fraction = 0.7
        sample_seed = 8675309
        id_key = 'id_key'
        keys = ['key_1', 'key_2']
        H5Dataset(h5_file, keys=keys, id_key=id_key, grain_size=5,
                  sample_fraction=sample_fraction, sample_seed=sample_seed)
        MockH5Data.assert_called_once_with(h5_file, keys=keys, id_key=id_key, grain_size=5,
                                              sample_fraction=sample_fraction, sample_seed=sample_seed)


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


def test_works_with_torch_dataloader(h5_file):
    keys = ['key_1', 'key_2']
    ds = H5Dataset(h5_file, keys=keys)
    dl = DataLoader(ds, batch_size=10)
    h5 = h5py.File(h5_file)
    entries = next(iter(dl))
    expected = tuple([h5[k][:10] for k in keys])
    assert all(len(entry) == 10 for entry in entries)
    assert isinstance(expected, tuple)
    assert all((entry.numpy() == array).all() for entry, array in zip(entries, expected))


def test_shuffle_works_with_torch_dataloader(h5_file):
    keys = ['key_1', 'key_2']
    ds = H5Dataset(h5_file, keys=keys)
    dl = DataLoader(ds, shuffle=True, batch_size=1000000)
    h5 = h5py.File(h5_file)
    all_entries = next(iter(dl))
    assert all(not (entry.numpy() == h5[k][:]).all().all() for entry, k in zip(all_entries, keys))
