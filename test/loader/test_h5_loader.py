import h5py
import os
import pytest
import numpy as np
import tempfile

from copy import deepcopy
from unittest.mock import patch

from easyloader.loader.h5 import H5DataLoader


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
    H5DataLoader(h5_file, keys=keys)


def test_args_passed_to_dataset_class(h5_file):
    with patch('easyloader.loader.h5.H5Dataset') as MockH5Dataset:
        sample_fraction = 0.7
        sample_seed = 8675309
        shuffle_seed = 5318008
        id_key = 'id_key'
        keys = ['key_1', 'key_2']
        H5DataLoader(h5_file, keys=keys, id_key=id_key, grain_size=5, shuffle_seed=shuffle_seed,
                     sample_fraction=sample_fraction, sample_seed=sample_seed)
        MockH5Dataset.assert_called_once_with(h5_file, keys=keys, id_key=id_key, grain_size=5, shuffle_seed=shuffle_seed,
                                           sample_fraction=sample_fraction, sample_seed=sample_seed)


def test_can_iterate(h5_file):
    keys = ['key_1', 'key_2']
    dl = H5DataLoader(h5_file, keys=keys)
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
def test_iterated_values_correct(h5_file, batch_size):
    keys = ['key_1', 'key_2']
    dl = H5DataLoader(h5_file, keys=keys, batch_size=batch_size)
    h5 = h5py.File(h5_file)
    batches_joined = helper_iterate_all_and_concatenate(dl, batch_size)
    for key, batch_joined in zip(keys, batches_joined):
        assert np.isclose(h5[key][:], batch_joined, atol=1e-7).all()


def test_ids_set(h5_file):
    keys = ['key_1', 'key_2']
    dl = H5DataLoader(h5_file, keys=keys)
    h5 = h5py.File(h5_file)
    assert len(dl.ids) == len(h5[keys[0]][:])
    dl = H5DataLoader(h5_file, keys=keys, id_key='id_key')
    assert (dl.ids == h5['id_key'][:]).all()


def test_shuffle_works(h5_file):
    batch_size = 11
    keys = ['key_1', 'key_2']
    dl = H5DataLoader(h5_file, keys=keys, batch_size=batch_size, shuffle=True)
    h5 = h5py.File(h5_file)
    batches_joined = helper_iterate_all_and_concatenate(dl, batch_size)
    for key, array_out in zip(keys, batches_joined):
        np.isclose(np.sort(h5[key][:], axis=0), np.sort(array_out, axis=0), atol=1e-7).all()


def test_shuffle_consistent(h5_file):
    batch_size = 11
    keys = ['key_1', 'key_2']
    dl1 = H5DataLoader(h5_file, keys=keys, batch_size=batch_size, shuffle=True, shuffle_seed=8675309)
    dl1_batch1 = deepcopy(next(iter(dl1)))
    dl2 = H5DataLoader(h5_file, keys=keys, batch_size=batch_size, shuffle=True, shuffle_seed=8675309)
    dl2_batch1 = deepcopy(next(iter(dl2)))
    for subbatch1, subbatch1 in zip(dl1_batch1, dl2_batch1):
        assert (subbatch1 == subbatch1).all().all()


def test_sample_works(h5_file):
    batch_size = 11
    keys = ['key_1', 'key_2']
    h5 = h5py.File(h5_file)
    dl = H5DataLoader(h5_file, keys=keys, batch_size=batch_size, sample_fraction=0.7)
    batches_joined = helper_iterate_all_and_concatenate(dl, batch_size)
    for array_out in batches_joined:
        assert len(array_out) == len(h5[keys[0]]) * 0.7


def test_sample_consistent(h5_file):
    batch_size = 11
    keys = ['key_1', 'key_2']
    dl1 = H5DataLoader(h5_file, keys=keys, batch_size=batch_size, sample_fraction=0.7, sample_seed=4)
    dl1_batch1 = deepcopy(next(iter(dl1)))
    dl2 = H5DataLoader(h5_file, keys=keys, batch_size=batch_size, sample_fraction=0.7, sample_seed=4)
    dl2_batch1 = deepcopy(next(iter(dl2)))
    for subbatch1, subbatch1 in zip(dl1_batch1, dl2_batch1):
        assert (subbatch1 == subbatch1).all().all()


def test_len_is_n_batches(h5_file):
    keys = ['key_1', 'key_2']
    dl = H5DataLoader(h5_file, keys=keys, batch_size=9)
    assert len(dl) == 12
