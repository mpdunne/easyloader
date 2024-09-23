import numpy as np
import random
import torch

from typing import Sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from uloops.loaders.common import get_n_batches


def sample_ixs(ixs: Sequence[int],
               n_samples: int = None,
               shuffle: bool = False,
               sample_seed: int = None,
               shuffle_seed: int = None) -> Sequence[int]:

    if n_samples != None and n_samples != len(ixs):
        random.seed(sample_seed)
        ixs = random.sample([*range(len(ixs))], n_samples)
        ixs = sorted(ixs)

    if shuffle:
        random.seed(shuffle_seed)
        ixs = random.sample(ixs, len(ixs))

    return ixs


class ArrayDataLoader(DataLoader):
    """
    Turn a pandas data frame into a PyTorch Data Loader.

    """

    def __init__(self,
                 arrays: Sequence[np.ndarray],
                 batch_size: int = 1,
                 sample_fraction: float = 1.0,
                 sample_seed: int = None,
                 shuffle: bool = False):

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

        # Only do any shuffling at iter time.
        self.shuffle = shuffle

        # Calculate number of batches
        self.n_batches = get_n_batches(self.data_length, batch_size)
        self.batch_size = batch_size

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
