import pandas as pd
import pytest

from torch.utils.data import DataLoader

from easyloader.dataset.df import DFDataset


@pytest.fixture(scope='session')
def df():
    n_entries = 100
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


@pytest.mark.parametrize(
    'sample_seed', (8675309, None)
)
def test_not_sampled_if_not_asked(df, sample_seed):
    original_df = df.copy()
    # There should be no sampling, even if we give it a seed.
    data = DFDataset(df, sample_seed=sample_seed)
    assert len(data.df) == len(data) == len(original_df)
    assert (data.df.index == original_df.index).all()


@pytest.mark.parametrize(
    'sample_seed', (8675309, None)
)
def test_sampled_correct_length_and_ordered(df, sample_seed):
    # No seed, but a sample fraction.
    original_df = df.copy()
    data = DFDataset(df, sample_seed=sample_seed, sample_fraction=0.7)
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
    data = DFDataset(df, sample_seed=seed, sample_fraction=0.7)
    ix_sets = [list(data.df.index)]
    for _ in range(4):
        data = DFDataset(df, sample_seed=seed, sample_fraction=0.7)
        ixs = list(data.df.index)
        assert all((ixs == ixsc) == consistent for ixsc in ix_sets)
        ix_sets.append(ixs)


def test_shuffle_works(df):
    df_orig = df.copy()
    data = DFDataset(df, shuffle_seed=8675309)
    assert (data.df.index == df_orig.index).all().all()
    assert (data.df == df_orig).all().all()
    data.shuffle()
    assert not (data.df.index == df_orig.index).all().all()
    assert set(data.df.index) == set(df_orig.index)
    assert len(data.df) == len(df_orig)


def test_shuffle_consistent(df):
    data = DFDataset(df, shuffle_seed=8675309)
    ix_sets = [list(data.df.index)]
    for _ in range(4):
        data = DFDataset(df, shuffle_seed=8675309)
        ixs = list(data.df.index)
        assert all((ixs == ixsc) for ixsc in ix_sets)


def test_shuffle_changes_index(df):
    data = DFDataset(df, shuffle_seed=8675309)
    index_orig = data.index.copy()
    data.shuffle()
    assert data.index != index_orig
    assert sorted(data.index) == sorted(index_orig)


def test_ids_specified(df):
    data = DFDataset(df, shuffle_seed=8675309, id_column='id')
    assert (data.ids == df['id']).all()


def test_ids_unspecified(df):
    data = DFDataset(df, shuffle_seed=8675309)
    assert (data.ids == data.df.index).all()


def test_shuffle_changes_ids(df):
    data = DFDataset(df, shuffle_seed=8675309, id_column='id')
    data.shuffle()
    assert list(data.ids) != sorted(list(data.ids))


def test_duplicate_ixs_okay(df):
    double_df = pd.concat([df, df])
    data = DFDataset(double_df, shuffle_seed=8675309, id_column='id', sample_fraction=0.7)
    assert len(data.df) == 0.7 * len(double_df)
    data.shuffle()
    assert len(data.df) == 0.7 * len(double_df)


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


def test_slice_works(df):
    column_groups = [['ones', 'tens'], ['hundreds']]
    ds = DFDataset(df, column_groups=column_groups)
    slices = ds[:10]
    assert all(len(s) == 10 for s in slices)
    assert all((s == df[g].iloc[:10]).all().all() for s, g in zip(slices, column_groups))


def test_slice_works_sampled(df):
    column_groups = [['ones', 'tens'], ['hundreds']]
    ds = DFDataset(df, column_groups=column_groups, sample_fraction=0.3, sample_seed=8675309)
    slices = ds[:10]
    assert all(len(s) == 10 for s in slices)
    assert all(not (s == df[g].iloc[:10]).all().all() for s, g in zip(slices, column_groups))


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
