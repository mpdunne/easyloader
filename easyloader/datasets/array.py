import numpy as np
import random
import torch

from typing import Sequence

from uloops.loaders.common import get_n_batches

from easyloader.datasets.base import EasyDataset
from easyloader.common.array import sample_ixs


class ArrayDataset(EasyDataset):
    """
    Turn a list of numpy arrays into a PyTorch Data Set.

    """

    def __init__(self,
                 arrays: Sequence[np.ndarray],
                 sample_fraction: float = 1.0,
                 sample_seed: int = None,
                 shuffle: bool = False,
                 shuffle_seed: int = None):
        """

        :param arrays: The arrays.
        :param sample_fraction: Fraction of the dataset to sample.
        :param sample_seed: Seed for random sampling.
        :param shuffle: Whether to shuffle the data.
        :param shuffle_seed: The seed to be used for shuffling.
        """

        # Initialize the parent class
        super().__init__(sample_fraction=sample_fraction,
                         sample_seed=sample_seed,
                         shuffle=shuffle,
                         shuffle_seed=shuffle_seed)

        array_lengths = [len(arr) for arr in arrays]
        if len(set(array_lengths)) != 1:
            raise ValueError('Arrays must all have the same length')

        self.data_length = int(sample_fraction * array_lengths[0])

        if shuffle or sample_fraction != 1:
            ixs = sample_ixs(range(self.data_length), n_samples=self.data_length,
                             sample_seed=sample_seed, shuffle=shuffle)
            self.arrays = [arr[ixs] for arr in arrays]
        else:
            ixs = [*range(self.data_length)]
            self.arrays = arrays

        self.index = ixs

    def __len__(self) -> int:
        return self.data_length

    def __getitem__(self, ix: int):
        return tuple([arr[ix] for arr in self.arrays])
