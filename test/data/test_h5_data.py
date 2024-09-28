import h5py
import os
import pytest
import numpy as np
import tempfile

from easyloader.data.h5 import H5Data


@pytest.fixture
def h5_file(scope='module'):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test.h5")

        # Open the HDF5 file in write mode
        with h5py.File(file_path, 'w') as h5file:
            # First dimension is consistent across all datasets, other dimensions vary
            h5file.create_dataset("key_1", data=np.random.rand(100, 10))
            h5file.create_dataset("key_2", data=np.random.rand(100, 5, 3))
            h5file.create_dataset("key_3", data=np.random.rand(100, 20))
            h5file.create_dataset("id_key", data=np.random.rand(100, ))

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


def test_can_instantiate(h5_file):
    _ = H5Data(h5_file, keys=['key_1'])


def test_cant_instantiate_without_keys(h5_file):
    with pytest.raises(TypeError):
        _ = H5Data(h5_file)

    with pytest.raises(ValueError):
        _ = H5Data(h5_file, keys=[])


def test_missing_keys_throws_error(h5_file):
    with pytest.raises(ValueError):
        _ = H5Data(h5_file, keys=['sausage'])


@pytest.mark.parametrize(
    'sample_seed', (8675309, None)
)
def test_not_sampled_if_not_asked(h5_file, sample_seed):
    keys = ['key1', 'key2']
    data = H5Data(h5_file, keys=keys, sample_seed=sample_seed)
    h5 = h5py.File(h5_file)
    arrays = data.


    expected_arrays = [h5[key][:] for key in keys]
    for key, expected_array in zip(data.arrays, expected_data):
        assert (array == original_array).all()


@pytest.mark.parametrize(
    'sample_seed', (8675309, None)
)
def test_sampled_correct_length_and_ordered(df, sample_seed):
    # No seed, but a sample fraction.
    original_df = df.copy()
    data = DFData(df, sample_seed=sample_seed, sample_fraction=0.7)
    assert len(data.df) == len(data) == len(original_df) * 0.7
    ixs = list(data.df.index)
    assert all(ixs[i] <= ixs[i + 1] for i in range(len(ixs) - 1))


@pytest.mark.parametrize(
    'seed,consistent',
    (
            (1, True),
            ('sausage', True),
            ((1, 2, 3), True),
            (None, False),
    )
)
def test_sampled_consistent(df, seed, consistent):
    data = DFData(df, sample_seed=seed, sample_fraction=0.7)
    ix_sets = [list(data.df.index)]
    for _ in range(4):
        data = DFData(df, sample_seed=seed, sample_fraction=0.7)
        ixs = list(data.df.index)
        assert all((ixs == ixsc) == consistent for ixsc in ix_sets)
        ix_sets.append(ixs)


def test_shuffle_works(df):
    df_orig = df.copy()
    data = DFData(df, shuffle_seed=8675309)
    assert (data.df.index == df_orig.index).all().all()
    assert (data.df == df_orig).all().all()
    data.shuffle()
    assert not (data.df.index == df_orig.index).all().all()
    assert set(data.df.index) == set(df_orig.index)
    assert len(data.df) == len(df_orig)


def test_shuffle_consistent(df):
    data = DFData(df, shuffle_seed=8675309)
    ix_sets = [list(data.df.index)]
    for _ in range(4):
        data = DFData(df, shuffle_seed=8675309)
        ixs = list(data.df.index)
        assert all((ixs == ixsc) for ixsc in ix_sets)


def test_shuffle_changes_index(df):
    data = DFData(df, shuffle_seed=8675309)
    index_orig = data.index.copy()
    data.shuffle()
    assert data.index != index_orig
    assert sorted(data.index) == sorted(index_orig)


def test_ids_specified(df):
    data = DFData(df, shuffle_seed=8675309, id_column='id')
    assert (data.ids == df['id']).all()


def test_ids_unspecified(df):
    data = DFData(df, shuffle_seed=8675309)
    assert (data.ids == data.df.index).all()


def test_shuffle_changes_ids(df):
    data = DFData(df, shuffle_seed=8675309, id_column='id')
    data.shuffle()
    assert list(data.ids) != sorted(list(data.ids))


def test_duplicate_ixs_okay(df):
    double_df = pd.concat([df, df])
    data = DFData(double_df, shuffle_seed=8675309, id_column='id', sample_fraction=0.7)
    assert len(data.df) == 0.7 * len(double_df)
    data.shuffle()
    assert len(data.df) == 0.7 * len(double_df)
