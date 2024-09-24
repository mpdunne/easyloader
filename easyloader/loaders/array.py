import numpy as np
import torch

from typing import Any, Sequence

from easyloader.loaders.base import EasyDataLoader
from easyloader.data.array import ArrayData
from easyloader.utils.batch import get_n_batches


class ArrayDataLoader(EasyDataLoader):
    """
    Turn a list of NumPy arrays into a PyTorch Data Loader.

    """

    def __init__(self,
                 arrays: Sequence[np.ndarray],
                 ids: Sequence[Any],
                 batch_size: int = 1,
                 sample_fraction: float = None,
                 sample_seed: int = None,
                 shuffle: bool = False,
                 shuffle_seed: bool = None):
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

    def __iter__(self):
        if self.shuffle:
            self.data.shuffle()

        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.n_batches:
            raise StopIteration
        batch = tuple(torch.Tensor(arr[self.i: self.i + self.batch_size]) for arr in self.data.arrays)
        self.i += 1
        return batch

    def __len__(self) -> int:
        return get_n_batches(self.data, self.batch_size)
