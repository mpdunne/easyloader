import numpy as np

from typing import Any, Iterable, Sequence

from easyloader.data.base import EasyData
from easyloader.utils.random import Seedable


class ArrayData(EasyData):
    """
    Data class for Array data.
    """

    def __init__(self, arrays: Sequence[np.ndarray],
                 ids: Sequence[Any] = None,
                 sample_fraction: float = None,
                 sample_seed: Seedable = None,
                 shuffle_seed: Seedable = None):
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

        # Check lengths
        array_lengths = [len(arr) for arr in arrays]
        if len(set(array_lengths)) != 1:
            raise ValueError('Arrays must all have the same length')
        array_length = array_lengths[0]

        # Organise the IDs
        index = [*range(array_length)]
        if ids is not None:
            if not len(ids) == array_length:
                raise ValueError('ID list must be the same length as the arrays.')
            self._ids = ids
        else:
            self._ids = index.copy()

        # Perform sampling
        if sample_fraction is not None:
            index = self.sample_random_state.sample(index, int(sample_fraction * array_length))
            index = sorted(index)
            self.arrays = [arr[index] for arr in arrays]
        else:
            self.arrays = arrays
        self._index = index

    def shuffle(self):
        """
        Shuffle the underlying DF.

        :return: None.
        """
        ixs = [*range(len(self.index))]
        self.shuffle_random_state.shuffle(ixs)
        self.arrays = [arr[ixs] for arr in self.arrays]
        self._index = list(np.array(self._index)[ixs])

    @property
    def ids(self) -> Iterable:
        """
        The IDs, according to the id_column attribute.

        :return: The IDs
        """
        return [self._ids[i] for i in self.index]

    @property
    def index(self):
        return self._index

    def __len__(self):
        return len(self.index)
