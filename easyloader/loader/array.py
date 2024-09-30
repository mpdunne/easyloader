import numpy as np
import torch
import random

from typing import Any, Sequence

from easyloader.loader.base import EasyDataLoader
from easyloader.data.array import ArrayData
from easyloader.utils.batch import get_n_batches
from easyloader.utils.random import Seedable


class ArrayDataLoader(EasyDataLoader):
    """
    Turn a list of NumPy arrays into a PyTorch Data Loader.

    """

    def __init__(self,
                 arrays: Sequence[np.ndarray],
                 ids: Sequence[Any] = None,
                 batch_size: int = 1,
                 sample_fraction: float = None,
                 shuffle: bool = False,
                 sample_seed: Seedable = None,
                 shuffle_seed: Seedable = None):
        """

        :param arrays: A list of arrays to use for the data loader
        :param batch_size: The batch size.
        :param sample_fraction: Fraction of the dataset to sample.
        :param sample_seed: Seed for random sampling.
        :param shuffle: Whether to shuffle the data.
        :param shuffle_seed: The seed to be used for shuffling.
        """

        # Initialize the parent class
        super().__init__(batch_size=batch_size,
                         sample_fraction=sample_fraction,
                         sample_seed=sample_seed,
                         shuffle=shuffle,
                         shuffle_seed=shuffle_seed)

        self.data = ArrayData(arrays, ids=ids, sample_fraction=sample_fraction,
                              sample_seed=sample_seed, shuffle_seed=shuffle_seed)

    @property
    def index(self):
        """
        The numeric indices of the underlying DF, relative to the inputted one.

        :return: The indices.
        """
        return self.data.index

    @property
    def ids(self):
        """
        The IDs, according to the id_column attribute.

        :return: The IDs
        """
        return self.data.ids

    def __iter__(self):
        if self.shuffle:
            self.data.shuffle()

        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self):
            raise StopIteration

        batch = tuple(
            torch.Tensor(arr[self.i * self.batch_size: (self.i + 1) * self.batch_size])
            for arr in self.data.arrays)

        self.i += 1
        return batch

    def __len__(self) -> int:
        return get_n_batches(len(self.data), self.batch_size)
