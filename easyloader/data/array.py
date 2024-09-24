import numpy as np
import random

from typing import Any, Iterable, Sequence

from easyloader.data.base import EasyData


class ArrayData(EasyData):
    """
    Data class for Array data.
    """

    def __init__(self, arrays: Sequence[np.ndarray],
                 ids: Sequence[Any] = None,
                 sample_fraction: float = None,
                 sample_seed=None,
                 shuffle_seed=None):
        """
        Constructor for the ArrayData class

        :param arrays: The arrays to use for the data loader.
        :param ids: A sequence of IDs for the array data.
        :param sample_fraction: Fraction of the dataset to sample.
        :param sample_seed: Seed for random sampling.
        :param shuffle_seed: The seed to be used for shuffling.
        """

        # Initialize the parent class
        super().__init__(sample_fraction=sample_fraction,
                         sample_seed=sample_seed,
                         shuffle_seed=shuffle_seed)

        array_lengths = [len(arr) for arr in arrays]
        if len(set(array_lengths)) != 1:
            raise ValueError('Arrays must all have the same length')

        self.array_length = array_lengths[0]

        index = [*range(self.array_length)]
        if sample_fraction is not None:
            index = random.sample(index, int(sample_fraction * self.array_length))
            index = sorted(index)
            self.arrays = [arr[index] for arr in arrays]
        else:
            self.arrays = arrays

        self._index = index

        self.arrays = arrays
        self.ids = ids

    def shuffle(self):
        """
        Shuffle the underlying DF.

        :return: None.
        """
        ixs = [*range(self.array_length)]
        self.shuffle_random_state.shuffle(ixs)
        self.arrays = [arr[ixs] for arr in self.arrays]
        self._index = list(np.array(self._index)[ixs])

    def ids(self) -> Iterable:
        """
        The IDs, according to the id_column attribute.

        :return: The IDs
        """
        if self.ids is None:
            return [self.ids[i] for i in self.index]
        else:
            return self.index

    @property
    def index(self):
        return self._index

    def __len__(self):
        return self.array_length
