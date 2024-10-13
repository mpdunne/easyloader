from functools import reduce

from typing import List


def append_and_merge_slice(slices: List[slice], next_slice: slice) -> List[slice]:
    """
    Join a slice onto the end of a list. If it follows on from the last slice in the list, just
    make that slice longer.

    :param slices: A list of slices.
    :param next_slice: A new slice to add to the fray.
    :return:
    """
    if slices[-1].stop == next_slice.start:
        slices[-1] = slice(slices[-1].start, next_slice.stop)
    else:
        slices.append(next_slice)
    return slices


def merge_neighbouring_slices(slices: List[slice]) -> List[slice]:
    """
    Given a list of slices, if one slice starts where the previous one stops, merge them.

    :param slices: A list of slices.
    :return: A list of merged slices.
    """
    if not slices:
        return []

    # Use the first slice as the base case, directly modify the list
    return reduce(append_and_merge_slice, slices[1:], [slices[0]])
