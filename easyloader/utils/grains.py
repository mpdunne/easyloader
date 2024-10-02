import math

from typing import Sequence


def fix_indices(total: int, ix_start: int, ix_stop: int):
    """
    In order to accurately work with indices, it's easiest to make sure they are
    positive and within range. This function adjusts them.

    :param total: The total number of items in the list.
    :param ix_start: The start index.
    :param ix_stop: The stop index.
    :return:
    """
    # Deal with Nones and off-the-edge starts and stops.
    if ix_start is None or ix_start < -total:
        ix_start = 0
    if ix_stop is None or ix_stop > total:
        ix_stop = total

    # If the start index is too big or the stop index is too small, the indices are void.
    if ix_stop <= -total or ix_start >= total:
        return 0, 0

    # Make start and stop both positive.
    if ix_start < 0:
        ix_start = ix_start + total

    if ix_stop < 0:
        ix_stop = ix_stop + total

    if ix_stop < ix_start:
        return 0, 0

    return ix_start, ix_stop


def grab_slices_from_grains(grain_index: Sequence[int], grain_size: int, ix_start: int, ix_stop: int):
    """
    Expands indexed grains by their grain size, and extracts contiguous slices of these between
    the given ix_start and ix_stop. The idea is
    - The grain index is a list of integer indices.
    - The grain size is the size of each grain.
    - The ix_start is the index to start extracting slices
    - The ix_stop is the index to stop extracting slices.
    The result should be a bunch of slices, which, when expanded, are the same as extracting an index from the
    grain_index, and slicing this as in index[ix_start: ix_stop].

    :param grain_index: A list of grain indices.
    :param ix_start: The starting index of the range to extract from the expanded grains.
    :param ix_stop: The stopping index of the range to extract from the expanded grains (non-inclusive).
    :param grain_size: The grain size.
    :return: A list of slice objects that represent the portions of the original slices
             which fall within the range [ix_start, ix_stop).
    :raises ValueError: If ix_start is greater than ix_stop.
    """

    # Convert the slices into standard ones that are within range.
    total = grain_size * len(grain_index)
    ix_start, ix_stop = fix_indices(total, ix_start, ix_stop)

    # Return an empty slice if the stop ix is the same as the start ix.
    if ix_start == ix_stop:
        return [slice(0, 0)]

    # Figure out which are the start and stop grain indices
    gix_start = ix_start // grain_size
    gix_stop = ix_stop // grain_size if (ix_stop % grain_size != 0) else ix_stop // grain_size - 1

    # Calculate the start and stop points within those slices
    start_sub_ix = ix_start % grain_size
    stop_sub_ix = ix_stop % grain_size if ix_stop % grain_size != 0 else grain_size

    # Directly return the slices using the computed indices.
    result = []
    if gix_start == gix_stop:

        # In this case, there will be maximum a single grain. Still return an empty slice though please.
        result += [slice(grain_index[gix_start] * grain_size + start_sub_ix,
                         grain_index[gix_start] * grain_size + stop_sub_ix)]
    else:

        # First grain.
        if gix_start < len(grain_index):
            result += [
                slice(grain_index[gix_start] * grain_size + start_sub_ix, (grain_index[gix_start] + 1) * grain_size)]

        # Middle grains.
        for mid_grain_start_value in grain_index[gix_start + 1: gix_stop]:
            result += [slice(mid_grain_start_value * grain_size, (mid_grain_start_value + 1) * grain_size)]

        # Final grain.
        if gix_stop is None or gix_stop < len(grain_index):
            result += [slice(grain_index[gix_stop] * grain_size, grain_index[gix_stop] * grain_size + stop_sub_ix)]

    return result
