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


class ArrayDataset(Dataset):
    """
    Turn a list of numpy arrays into a PyTorch Data Set.

    """

    def __init__(self,
                 arrays: Sequence[np.ndarray],
                 sample_fraction: float = 1.0,
                 sample_seed: int = None,
                 shuffle: bool = False):

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
