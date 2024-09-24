import pytest
import pandas as pd

from easyloader.data.df import DFData


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
    _ = DFData(df)


@pytest.mark.parametrize(
    'sample_seed', (8675309, None)
)
def test_not_sampled_if_not_asked(df, sample_seed):
    original_df = df.copy()
    # There should be no sampling, even if we give it a seed.
    data = DFData(df, sample_seed=sample_seed)
    assert len(data.df) == len(data) == len(original_df)
    assert (data.df.index == original_df.index).all()


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


def test_shuffle_changes_index(arrays):
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
