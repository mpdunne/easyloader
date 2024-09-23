import numpy as np
import torch

from typing import Sequence

from easyloader.loaders.base import EasyDataLoader
from easyloader.common.array import sample_ixs


class ArrayDataLoader(EasyDataLoader):
    """
    Turn a list of NumPy arrays into a PyTorch Data Loader.

    """

    def __init__(self,
                 arrays: Sequence[np.ndarray],
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

        array_lengths = [len(arr) for arr in arrays]
        if len(set(array_lengths)) != 1:
            raise ValueError('Arrays must all have the same length.')

        self.data_length = int(sample_fraction * array_lengths[0])

        # Only sample if we're asked to
        if sample_fraction != 1:
            ixs = sample_ixs(range(array_lengths[0]), n_samples=self.data_length, sample_seed=sample_seed)
            arrays = [arr[ixs] for arr in arrays]
        else:
            ixs = [*range(array_lengths[0])]
            arrays = arrays

        self.ixs = ixs
        self.arrays = arrays

        # These will be set after __iter__ has been called
        self.groups = None
        self.index = None

    def __iter__(self):
        if self.shuffle:
            sub_ixs = sample_ixs([*range(len(self.ixs))], n_samples=self.data_length)
            ixs = list(np.array(self.ixs)[sub_ixs])
            groups = [arr[sub_ixs] for arr in self.arrays]
        else:
            ixs = self.ixs
            groups = self.arrays

        self.index = ixs
        self.groups = groups

        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.n_batches:
            raise StopIteration
        batch = tuple(torch.Tensor(g[self.i: self.i + self.batch_size]) for g in self.groups)
        self.i += 1
        return batch

    def __len__(self) -> int:
        return self.n_batches
