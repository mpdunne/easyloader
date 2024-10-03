import pytest
import random

from easyloader.utils.grains import fix_indices, grab_slices_from_grains


@pytest.mark.parametrize('n_items', (0, 4, 8, 15, 16, 23, 42))
def test_fix_indices(n_items):
    items = [*range(n_items)]
    # Do this combinatorially to check all possibilities.
    for ix_start in range(-n_items * 2, n_items * 2):
        for ix_stop in range(-n_items * 2, n_items * 2):
            ix_start_fixed, ix_stop_fixed = fix_indices(n_items, ix_start, ix_stop)
            assert 0 <= ix_start_fixed < n_items
            assert 0 <= ix_stop_fixed <= n_items
            assert ix_start_fixed <= ix_stop_fixed
            assert items[ix_start:ix_stop] == items[ix_start_fixed:ix_stop_fixed]


def helper_check_grab_slices(grain_index, grain_size, ix_start, ix_stop, result):
    # This should pass if the expected behaviour has occurred.

    # Expand the grains first, then take indices.
    grains_expanded = [ix for gix in grain_index for ix in range(gix * grain_size, (gix + 1) * grain_size)]
    expected_expanded = grains_expanded[ix_start: ix_stop]

    # This is the result using the function that we want to test,
    # i.e. grab_slices_from_grains, which is more computationally efficient.
    result_expanded = [ix for s in result for ix in range(s.start, s.stop)]

    # They should be equal.
    assert expected_expanded == result_expanded

    # The slices should also all be standard and valid.
    assert all(0 <= s.start <= s.stop <= len(grain_index) * grain_size for s in result)


@pytest.mark.parametrize('grain_size,ix_start,ix_stop', (
        (5, 12, 13),
        (5, 12, 15),
        (5, 15, 20),
        (10, 12, 18),
        (10, 12, 18),
        (10, 12, 20),
        (10, 12, 17),
        (10, 10, 20),
        (20, 3, 19),
        (20, 34, 38),
        (20, 0, 20),
        (100, 1, 99),
        (100, 0, 100),
    )
)
@pytest.mark.parametrize('grain_index_size', (40, 80, 150, 160, 230, 420))
def test_grab_slices_from_grains_ixs_in_same_grain(grain_index_size, grain_size, ix_start, ix_stop):
    grain_index = [*range(grain_index_size)]
    rng = random.Random(8675309)
    rng.shuffle(grain_index)
    result = grab_slices_from_grains(grain_index, grain_size, ix_start, ix_stop)
    helper_check_grab_slices(grain_index, grain_size, ix_start, ix_stop, result)
    assert len(result) == 1


@pytest.mark.parametrize('grain_size,ix_start,ix_stop', (
        (5, 12, 17),
        (5, 12, 18),
        (10, 12, 22),
        (10, 12, 22),
        (10, 99, 101),
        (20, 3, 33),
        (20, 34, 49),
        (100, 200, 301),
    )
)
@pytest.mark.parametrize('grain_index_size', (40, 80, 150, 160, 230, 420))
def test_grab_slices_from_grains_ixs_in_consecutive_grains(grain_index_size, grain_size, ix_start, ix_stop):
    grain_index = [*range(grain_index_size)]
    rng = random.Random(8675309)
    rng.shuffle(grain_index)
    result = grab_slices_from_grains(grain_index, grain_size, ix_start, ix_stop)
    helper_check_grab_slices(grain_index, grain_size, ix_start, ix_stop, result)
    assert len(result) == 2


@pytest.mark.parametrize('grain_size,ix_start,ix_stop', (
        (5, 12, 37),
        (5, 12, 38),
        (10, 12, 52),
        (10, 12, 52),
        (10, 99, 401),
        (20, 3, 333),
        (20, 34, 249),
        (100, 200, 3301),
    )
)
@pytest.mark.parametrize('grain_index_size', (40, 80, 150, 160, 230, 420))
def test_grab_slices_from_grains_ixs_in_distant_grains(grain_index_size, grain_size, ix_start, ix_stop):
    grain_index = [*range(grain_index_size)]
    rng = random.Random(8675309)
    rng.shuffle(grain_index)
    result = grab_slices_from_grains(grain_index, grain_size, ix_start, ix_stop)
    helper_check_grab_slices(grain_index, grain_size, ix_start, ix_stop, result)
    assert len(result) > 2


@pytest.mark.parametrize('grain_size,ix_start,ix_stop', (
        (5, -20, -1),
        (5, 12, -1),
        (10, -12, -10),
        (10, 12, -10),
        (10, -99, -23),
        (20, 3, -23),
        (20, -45, -40),
        (100, 200, -40),
    )
)
@pytest.mark.parametrize('grain_index_size', (40, 80, 150, 160, 230, 420))
def test_grab_slices_valid_negative_indices_are_fine(grain_index_size, grain_size, ix_start, ix_stop):
    grain_index = [*range(grain_index_size)]
    rng = random.Random(8675309)
    rng.shuffle(grain_index)
    result = grab_slices_from_grains(grain_index, grain_size, ix_start, ix_stop)
    helper_check_grab_slices(grain_index, grain_size, ix_start, ix_stop, result)


@pytest.mark.parametrize('grain_size,ix_start,ix_stop', (
        (5, None, 0),
        (5, None, 1),
        (5, None, 5),
        (5, None, 7),
        (5, None, -5),
        (5, None, -1),
        (5, 0, None),
        (5, 1, None),
        (5, 5, None),
        (5, 7, None),
        (5, -5, None),
        (5, -1, None),
        (5, None, None),
    )
)
@pytest.mark.parametrize('grain_index_size', (40, 80, 150, 160, 230, 420))
def test_grab_slices_none_handled_correctly(grain_index_size, grain_size, ix_start, ix_stop):
    grain_index = [*range(grain_index_size)]
    rng = random.Random(8675309)
    rng.shuffle(grain_index)
    result = grab_slices_from_grains(grain_index, grain_size, ix_start, ix_stop)
    helper_check_grab_slices(grain_index, grain_size, ix_start, ix_stop, result)


@pytest.mark.parametrize('grain_index_size,grain_size,ix_start,ix_stop', (
        (100, 5, 12, 12),
        (100, 5, 15, 15),
        (100, 5, 15, 14),
        (100, 5, 15, 0),
        (100, 5, 15, 0),
        (100, 5, -90, 20),
        (10, 5, 8, 4),
        (100, 10, 2100, 2199),
        (100, 10, 21000, 21099),
    )
)
def test_grab_slices_bad_ixs_return_empty_slice(grain_index_size, grain_size, ix_start, ix_stop):
    grain_index = [*range(grain_index_size)]
    rng = random.Random(8675309)
    rng.shuffle(grain_index)
    result = grab_slices_from_grains(grain_index, grain_size, ix_start, ix_stop)
    assert len(result) == 1
    assert result[0].stop == result[0].start
    assert result[0].stop is not None and result[0].start is not None


def test_grab_slices_combinatorial():
    # Do a full check in case we've missed anything!
    for n_grains in range(10):
        for seed in range(5):
            grain_index = [*range(n_grains)]
            rng = random.Random(seed)
            rng.shuffle(grain_index)
            for grain_size in (4, 8, 15, 16, 23, 42):
                for ix_start in [*range(-n_grains * 3, n_grains * 3)] + [None]:
                    for ix_stop in [*range(-n_grains * 3, n_grains * 3)] + [None]:
                        result = grab_slices_from_grains(grain_index, grain_size, ix_start, ix_stop)
                        helper_check_grab_slices(grain_index, grain_size, ix_start, ix_stop, result)
