import numpy as np
import pandas as pd
import pytest

from torch.utils.data import DataLoader
from unittest.mock import patch

from easyloader.datasets.df import DFDataset


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
    DFDataset(df, column_groups=column_groups)


def test_args_passed_to_data_class(df):
    with patch('easyloader.datasets.df.DFData') as MockArrayData:
        sample_fraction = 0.7
        sample_seed = 8675309
        id_column = 'id'
        column_groups = [['ones', 'tens'], ['hundreds']]
        DFDataset(df, column_groups=column_groups, id_column=id_column,
                  sample_fraction=sample_fraction, sample_seed=sample_seed)
        MockArrayData.assert_called_once_with(df, id_column=id_column,
                                              sample_fraction=sample_fraction, sample_seed=sample_seed,)


def test_can_get_item(df):
    column_groups = [['ones', 'tens'], ['hundreds']]
    ds = DFDataset(df, column_groups=column_groups)
    entries = ds[5]
    assert isinstance(entries, tuple)
    assert (entries[0] == df[column_groups[0]].iloc[5]).all()
    assert (entries[1] == df[column_groups[1]].iloc[5]).all()


def test_cant_get_out_of_range_item(df):
    with pytest.raises(IndexError):
        column_groups = [['ones', 'tens'], ['hundreds']]
        ds = DFDataset(df, column_groups=column_groups)
        ds[1000000]


def test_can_be_inputted_to_torch_dataloader(df):
    column_groups = [['ones', 'tens'], ['hundreds']]
    ds = DFDataset(df, column_groups=column_groups)
    DataLoader(ds)


def test_column_groups_used(df):
    column_groups = [['ones', 'tens'], ['hundreds']]
    ds = DFDataset(df, column_groups)
    entry = ds[5]
    assert (entry[0] == df[column_groups[0]].iloc[5]).all().all()
    assert (entry[1] == df[column_groups[1]].iloc[5]).all().all()


def test_works_with_torch_dataloader(df):
    column_groups = [['ones', 'tens'], ['hundreds']]
    ds = DFDataset(df, column_groups=column_groups)
    dl = DataLoader(ds, batch_size=10)
    entries = next(iter(dl))
    expected = tuple([df[g].iloc[:10] for g in column_groups])
    assert all(len(entry) == 10 for entry in entries)
    assert isinstance(expected, tuple)
    assert all((entry.numpy() == array).all().all() for entry, array in zip(entries, expected))


def test_shuffle_works_with_torch_dataloader(df):
    column_groups = [['ones', 'tens'], ['hundreds']]
    ds = DFDataset(df, column_groups=column_groups)
    dl = DataLoader(ds, shuffle=True, batch_size=1000000)
    all_entries = next(iter(dl))
    assert all(not (entry.numpy() == df[g]).all().all() for entry, g in zip(all_entries, column_groups))
