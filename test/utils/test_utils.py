import pytest

from easyloader.utils.batch import get_n_batches


@pytest.mark.parametrize('data_length,batch_size,expected_n_batches',
                         (
                                 (100, 20, 5),
                                 (100, 50, 2),
                                 (0, 1, 0),
                         )
)
def test_get_n_batches_no_remainder(data_length, batch_size, expected_n_batches):
    n_batches = get_n_batches(data_length, batch_size)
    assert n_batches == expected_n_batches


@pytest.mark.parametrize('data_length,batch_size',
                         (
                                 (-1, 20),
                                 (100, -1),
                                 (1, 0),
                                 (0, 0),
                                 (-1, 20),
                                 ('A', 20),
                                 (100, 'A'),
                                 (0, 0),
                         )
)
def test_weird_values_throw_error(data_length, batch_size):
    with pytest.raises(ValueError):
        get_n_batches(data_length, batch_size)


@pytest.mark.parametrize('data_length,batch_size,expected_n_batches',
                         (
                                 (101, 20, 6),
                                 (102, 50, 3),
                         )
)
def test_get_n_batches_with_remainder(data_length, batch_size, expected_n_batches):
    n_batches = get_n_batches(data_length, batch_size)
    assert n_batches == expected_n_batches
