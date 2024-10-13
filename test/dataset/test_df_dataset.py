import math
import numpy as np
import pandas as pd
import pytest
import torch

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


def test_can_instantiate_with_no_columns_specified(df):
    ds = DFDataset(df)
    assert ds._column_groups == [[*df.columns]]


def test_can_instantiate_with_single_column_specified(df):
    DFDataset(df, columns='ones')


def test_can_instantiate_with_only_columns_specifed(df):
    DFDataset(df, columns=['ones', 'tens'])


def test_can_instantiate_withcolumn_groups_specified(df):
    DFDataset(df, columns=[['ones', 'tens'], ['hundreds']])


def test_can_instantiate_with_mix_of_columns_and_column_groups(df):
    column_groups = [['ones', 'tens'], 'hundreds']
    DFDataset(df, columns=column_groups)


def test_missing_columns_causes_error(df):
    with pytest.raises(ValueError):
        column_groups = [['ones', 'tens'], ['thousands']]
        DFDataset(df, columns=column_groups)


def test_single_column_gives_single_output(df):
    ds = DFDataset(df, columns='ones')
    assert isinstance(ds[0], np.ndarray)
    assert ds[:10].shape == (10,)
    assert ds[0].shape == ()


def test_ungrouped_columns_give_single_output(df):
    columns = ['ones', 'tens']
    ds = DFDataset(df, columns=columns)
    assert isinstance(ds[0], np.ndarray)
    assert ds[:10].shape == (10, 2)
    assert ds[0].shape == (2, )


def test_grouped_columns_give_grouped_output(df):
    column_groups = [['ones', 'tens']]
    ds = DFDataset(df, columns=column_groups)
    assert isinstance(ds[0], tuple)
    assert len(ds[0]) == 1
    assert ds[0][0].shape == (2,)
    assert len(ds[:10]) == 1
    assert ds[:10][0].shape == (10, 2)

    column_groups = [['ones', 'tens'], 'hundreds']
    ds = DFDataset(df, columns=column_groups)
    assert isinstance(ds[0], tuple)
    assert len(ds[0]) == 2
    assert ds[0][0].shape == (2,)
    assert ds[0][1].shape == ()
    assert len(ds[:10]) == 2
    assert ds[:10][0].shape == (10, 2)
    assert ds[:10][1].shape == (10,)

    column_groups = [['ones', 'tens'], ['hundreds']]
    ds = DFDataset(df, columns=column_groups)
    assert isinstance(ds[0], tuple)
    assert len(ds[0]) == 2
    assert ds[0][0].shape == (2,)
    assert ds[0][1].shape == (1,)
    assert len(ds[:10]) == 2
    assert ds[:10][0].shape == (10, 2)
    assert ds[:10][1].shape == (10, 1)


def test_tuple_columns_handled_correctly(df):
    df[('tens', 'hundreds')] = df['tens'] + df['hundreds']

    ds = DFDataset(df, columns=('tens', 'hundreds'))
    assert ds._column_groups == [('tens', 'hundreds')]
    assert ds._single

    ds = DFDataset(df, columns=[('tens', 'hundreds')])
    assert ds._column_groups == [[('tens', 'hundreds')]]
    assert ds._single

    ds = DFDataset(df, columns=('ones', 'hundreds'))
    assert ds._column_groups == [['ones', 'hundreds']]
    assert ds._single

    ds = DFDataset(df, columns=[('tens', 'hundreds'), 'ones'])
    assert ds._column_groups == [[('tens', 'hundreds'), 'ones']]


@pytest.mark.parametrize(
    'sample_seed', (8675309, None)
)
def test_not_sampled_if_not_asked(df, sample_seed):
    original_df = df.copy()
    # There should be no sampling, even if we give it a seed.
    data = DFDataset(df, sample_seed=sample_seed)
    assert len(data._df) == len(data) == len(original_df)
    assert (data._df.index == original_df.index).all()


@pytest.mark.parametrize(
    'sample_seed', (8675309, None)
)
def test_sampled_correct_length_and_ordered(df, sample_seed):
    # No seed, but a sample fraction.
    original_df = df.copy()
    data = DFDataset(df, sample_seed=sample_seed, sample_fraction=0.7)
    assert len(data._df) == len(data) == len(original_df) * 0.7
    ixs = list(data._df.index)
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
    ix_sets = [list(data._df.index)]
    for _ in range(4):
        data = DFDataset(df, sample_seed=seed, sample_fraction=0.7)
        ixs = list(data._df.index)
        assert all((ixs == ixsc) == consistent for ixsc in ix_sets)
        ix_sets.append(ixs)


@pytest.mark.parametrize('sample_seed', [*range(10)])
@pytest.mark.parametrize('grain_size', [*range(1, 11)])
def test_sample_grained(df, grain_size, sample_seed):
    data = DFDataset(df, grain_size=grain_size, sample_fraction=0.7, sample_seed=sample_seed)
    assert all(data.index[i + 1] == data.index[i] + 1 for i in range(len(data) - 1) if (i + 1) % grain_size != 0)

    # Size can vary depending on whether the final grain is included in the sample.
    n_original_grains = int(math.ceil(len(df) / grain_size))
    n_sampled_grains = int(n_original_grains * 0.7)

    lower = grain_size * (n_sampled_grains - 1)
    upper = grain_size * n_sampled_grains
    assert lower < len(data.index) == len(data) <= upper


def test_shuffle_works(df):
    df_orig = df.copy()
    data = DFDataset(df, shuffle_seed=8675309)
    assert (data._df.index == df_orig.index).all().all()
    assert (data._df == df_orig).all().all()
    data.shuffle()
    assert not (data._df.index == df_orig.index).all().all()
    assert set(data._df.index) == set(df_orig.index)
    assert len(data._df) == len(df_orig)


def test_shuffle_consistent(df):
    data = DFDataset(df, shuffle_seed=8675309)
    ix_sets = [list(data._df.index)]
    for _ in range(4):
        data = DFDataset(df, shuffle_seed=8675309)
        ixs = list(data._df.index)
        assert all((ixs == ixsc) for ixsc in ix_sets)


def test_shuffle_changes_index(df):
    data = DFDataset(df, shuffle_seed=8675309)
    index_orig = data.index.copy()
    data.shuffle()
    assert data.index != index_orig
    assert sorted(data.index) == sorted(index_orig)


@pytest.mark.parametrize('grain_size', [*range(1, 11)])
def test_shuffle_grained(df, grain_size):
    data = DFDataset(df, grain_size=grain_size)

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
                    for ix in range(gix * grain_size, (gix + 1) * grain_size) if ix < len(df)]
        assert expected == data.index


def test_ids_unspecified(df):
    data = DFDataset(df, shuffle_seed=8675309)
    assert (data.ids == data._df.index).all()


def test_ids_specified_as_column(df):
    data = DFDataset(df, shuffle_seed=8675309, ids='id')
    assert (data.ids == df['id']).all()


def test_ids_specified_as_column_bad(df):
    with pytest.raises(ValueError):
        data = DFDataset(df, shuffle_seed=8675309, ids='sausage')


def test_ids_specified_as_list(df):
    ids = [f'ix_{i}' for i in range(len(df))]
    data = DFDataset(df, shuffle_seed=8675309, ids=ids)
    assert data.ids == ids


def test_ids_specified_as_list_wrong_size(df):
    with pytest.raises(ValueError):
        DFDataset(df, shuffle_seed=8675309, ids=[f'ix_{i}' for i in range(len(df) - 1)])


def test_ids_specified_wrong_type(df):
    with pytest.raises(TypeError):
        DFDataset(df, shuffle_seed=8675309, ids=5)


def test_shuffle_changes_ids(df):
    data = DFDataset(df, shuffle_seed=8675309, ids='id')
    data.shuffle()
    assert list(data.ids) != sorted(list(data.ids))


def test_duplicate_ixs_okay(df):
    double_df = pd.concat([df, df])
    data = DFDataset(double_df, shuffle_seed=8675309, ids='id', sample_fraction=0.7)
    assert len(data._df) == 0.7 * len(double_df)
    data.shuffle()
    assert len(data._df) == 0.7 * len(double_df)


def test_can_get_item(df):
    column_groups = [['ones', 'tens'], ['hundreds']]
    ds = DFDataset(df, columns=column_groups)
    entries = ds[5]
    assert isinstance(entries, tuple)
    assert (entries[0] == df[column_groups[0]].iloc[5]).all()
    assert (entries[1] == df[column_groups[1]].iloc[5]).all()


def test_cant_get_out_of_range_item(df):
    with pytest.raises(IndexError):
        column_groups = [['ones', 'tens'], ['hundreds']]
        ds = DFDataset(df, columns=column_groups)
        ds[1000000]


def test_can_be_inputted_to_torch_dataloader(df):
    column_groups = [['ones', 'tens'], ['hundreds']]
    ds = DFDataset(df, columns=column_groups)
    DataLoader(ds)


def test_column_groups_used(df):
    column_groups = [['ones', 'tens'], ['hundreds']]
    ds = DFDataset(df, column_groups)
    entry = ds[5]
    assert (entry[0] == df[column_groups[0]].iloc[5]).all().all()
    assert (entry[1] == df[column_groups[1]].iloc[5]).all().all()


def test_slice_works(df):
    column_groups = [['ones', 'tens'], ['hundreds']]
    ds = DFDataset(df, columns=column_groups)
    slices = ds[:10]
    assert all(len(s) == 10 for s in slices)
    assert all((s == df[g].iloc[:10]).all().all() for s, g in zip(slices, column_groups))


def test_slice_works_sampled(df):
    column_groups = [['ones', 'tens'], ['hundreds']]
    ds = DFDataset(df, columns=column_groups, sample_fraction=0.3, sample_seed=8675309)
    slices = ds[:10]
    assert all(len(s) == 10 for s in slices)
    assert all(not (s == df[g].iloc[:10]).all().all() for s, g in zip(slices, column_groups))


@pytest.mark.parametrize('column_groups,single', (
        ('ones', True),
        (['ones', 'tens'], True),
        ([['ones', 'tens'], 'hundreds'], False),
        ([['ones', 'tens'], ['hundreds']], False),
))
def test_works_with_torch_dataloader(df, column_groups,single):
    ds = DFDataset(df, columns=column_groups)
    dl = DataLoader(ds, batch_size=10)
    entries = next(iter(dl))

    if single:
        expected = df[column_groups].iloc[:10]
        assert len(entries) == 10
        assert isinstance(entries, torch.Tensor)
        assert (entries.numpy() == expected).all().all()

    else:
        expected = tuple([df[g].iloc[:10] for g in column_groups])
        assert all(len(entry) == 10 for entry in entries)
        assert isinstance(entries, list)
        assert all((entry.numpy() == array).all().all() for entry, array in zip(entries, expected))


def test_shuffle_works_with_torch_dataloader(df):
    column_groups = [['ones', 'tens'], ['hundreds']]
    ds = DFDataset(df, columns=column_groups)
    dl = DataLoader(ds, shuffle=True, batch_size=1000000)
    all_entries = next(iter(dl))
    assert all(not (entry.numpy() == df[g]).all().all() for entry, g in zip(all_entries, column_groups))
