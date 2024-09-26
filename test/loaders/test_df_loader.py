import numpy as np
import pandas as pd
import pytest

from copy import deepcopy
from unittest.mock import patch

from easyloader.loaders.df import DFDataLoader


@pytest.fixture(scope='session')
def df():
    n_entries = 1000
    data = [
        [*range(0, n_entries)],
        [*range(0, n_entries * 10, 10)],
        [*range(0, n_entries * 100, 100)]
    ]
    df = pd.DataFrame(data).T
    df.columns = ['ones', 'tens', 'hundreds']
    df['id'] = [f'entry_{i}' for i in range(n_entries)]
    return df


def test_can_instantiate(df):
    column_groups = [['ones', 'tens'], ['hundreds']]
    DFDataLoader(df, column_groups=column_groups)


def test_args_passed_to_data_class(df):
    with patch('easyloader.loaders.df.DFData') as MockArrayData:
        sample_fraction = 0.7
        sample_seed = 8675309
        shuffle_seed = 5318008
        id_column = 'id'
        column_groups = [['ones', 'tens'], ['hundreds']]
        DFDataLoader(df, column_groups=column_groups, id_column=id_column, shuffle_seed=shuffle_seed,
                     sample_fraction=sample_fraction, sample_seed=sample_seed)
        MockArrayData.assert_called_once_with(df, id_column=id_column, shuffle_seed=shuffle_seed,
                                              sample_fraction=sample_fraction, sample_seed=sample_seed)


def test_can_iterate(df):
    column_groups = [['ones', 'tens'], ['hundreds']]
    dl = DFDataLoader(df, column_groups=column_groups)
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
def test_iterated_values_correct(df, batch_size):
    column_groups = [['ones', 'tens'], ['hundreds']]
    dl = DFDataLoader(df, column_groups=column_groups, batch_size=batch_size)
    batches_joined = helper_iterate_all_and_concatenate(dl, batch_size)
    for column_group, batch_joined in zip(column_groups, batches_joined):
        assert (df[column_group] == batch_joined).all().all()


def test_ids_set(df):
    column_groups = [['ones', 'tens'], ['hundreds']]
    dl = DFDataLoader(df, column_groups=column_groups)
    assert len(dl.ids) == len(df)
    dl = DFDataLoader(df, column_groups=column_groups, id_column='id')
    assert (dl.ids == df['id']).all()


def test_shuffle_works(df):
    batch_size = 11
    column_groups = [['ones', 'tens'], ['hundreds']]
    dl = DFDataLoader(df, column_groups=column_groups, batch_size=batch_size, shuffle=True)
    batches_joined = helper_iterate_all_and_concatenate(dl, batch_size)
    for column_group, array_out in zip(column_groups, batches_joined):
        assert (df[column_group].sort_values(column_group[0]) == np.sort(array_out, axis=0)).all().all()


def test_shuffle_consistent(df):
    batch_size = 11
    column_groups = [['ones', 'tens'], ['hundreds']]
    dl1 = DFDataLoader(df, column_groups=column_groups, batch_size=batch_size, shuffle=True, shuffle_seed=8675309)
    dl1_batch1 = deepcopy(next(iter(dl1)))
    dl2 = DFDataLoader(df, column_groups=column_groups, batch_size=batch_size, shuffle=True, shuffle_seed=8675309)
    dl2_batch1 = deepcopy(next(iter(dl2)))
    for subbatch1, subbatch1 in zip(dl1_batch1, dl2_batch1):
        assert (subbatch1 == subbatch1).all().all()


def test_sample_works(df):
    batch_size = 11
    column_groups = [['ones', 'tens'], ['hundreds']]
    dl = DFDataLoader(df, column_groups=column_groups, batch_size=batch_size, sample_fraction=0.7)
    batches_joined = helper_iterate_all_and_concatenate(dl, batch_size)
    for array_out in batches_joined:
        assert len(array_out) == len(df) * 0.7


def test_sample_consistent(df):
    batch_size = 11
    column_groups = [['ones', 'tens'], ['hundreds']]
    dl1 = DFDataLoader(df, column_groups=column_groups, batch_size=batch_size, sample_fraction=0.7, sample_seed=4)
    dl1_batch1 = deepcopy(next(iter(dl1)))
    dl2 = DFDataLoader(df, column_groups=column_groups, batch_size=batch_size, sample_fraction=0.7, sample_seed=4)
    dl2_batch1 = deepcopy(next(iter(dl2)))
    for subbatch1, subbatch1 in zip(dl1_batch1, dl2_batch1):
        assert (subbatch1 == subbatch1).all().all()


def test_len_is_n_batches(df):
    dl = DFDataLoader(df, column_groups=[['ones', 'tens'], ['hundreds']], batch_size=99)
    assert len(dl) == 11
