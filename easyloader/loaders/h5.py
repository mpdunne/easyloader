import h5py
import random
import numpy as np
import torch

from pathlib import Path
from typing import Sequence, Union
from torch.utils.data import Dataset, DataLoader

from uloops.loaders.common import get_n_batches


from easyloader.loaders.base import EasyDataLoader
from easyloader.common.h5 import check_keys


class H5DataLoader(EasyDataLoader):
    """
    Turn an H5 file into a Torch dataset.
    Using granular weak shuffling.
    https://towardsdatascience.com/reading-h5-files-faster-with-pytorch-datasets-3ff86938cc
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 keys: Sequence[str],
                 allow_missing_keys: bool = False,
                 index_key: str = None,
                 batch_size: int = 1,
                 grain_size: int = 1,
                 sample_fraction: float = None,
                 sample_seed: int = None,
                 shuffle: bool = False,
                 shuffle_seed: bool = None):
        """

        :param df:
        :param column_groups:
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

        if batch_size % grain_size != 0:
            raise ValueError(f'Batch size must be divisible by grain size.')

        data = h5py.File(data_path, "r")
        present_keys = check_keys(data.keys(), keys, allow_missing_keys)

        if index_key is not None and index_key not in data.keys():
            raise KeyError(f'If specified, index_key must be in the H5 dataset\'s keys. {index_key} is missing')

        data_length = len(data[present_keys[0]])

        self.keys = keys
        self.index_key = index_key
        self.data = data

        # Calculate # grains. This is the same calculation as for batch size.
        n_grains = get_n_batches(data_length, grain_size)

        # We sample the fraction using whole grains.
        if sample_fraction != 1:
            sample_size = int(sample_fraction * n_grains)
            rng = random.Random(sample_seed)
            self.sampled_grain_ixs = sorted(rng.sample([*range(n_grains)], sample_size))
        else:
            self.sampled_grain_ixs = [*range(n_grains)]

        # We have len(self.sampled_grain_ixs) grains, and they each have size grain_size, except,
        # possibly, for one. But that one won't change the number of batches so we can ignore it.
        sampled_data_length = len(self.sampled_grain_ixs) * grain_size

        # We use this to work out how many batches there are.
        n_batches = get_n_batches(sampled_data_length, batch_size)

        self.shuffle = shuffle
        self.sampled = sample_fraction != 1 or shuffle

        self.grain_size = grain_size
        self.n_batches = n_batches
        self.n_grains = n_grains
        self.grains_per_batch = batch_size // grain_size
        self.data_length = data_length

    def __iter__(self):
        if self.shuffle:
            self.sampled_grain_ixs = random.sample(self.sampled_grain_ixs, len(self.sampled_grain_ixs))

        # Generate and save the index.
        all_ixs = [*range(self.data_length)]
        grains = [all_ixs[ix * self.grain_size: (ix + 1) * self.grain_size] for ix in self.sampled_grain_ixs]
        ixs = [i for g in grains for i in g]
        if self.index_key is not None:
            self.index = self.data[self.index_key][:][ixs]
        else:
            self.index = ixs

        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.n_batches:
            raise StopIteration

        if self.sampled:
            grain_sample_start_ix = self.i * self.grains_per_batch
            grain_sample_end_ix = (self.i + 1) * self.grains_per_batch
            grain_sample_ixs = self.sampled_grain_ixs[grain_sample_start_ix: grain_sample_end_ix]
            batch = tuple(torch.Tensor(np.concatenate([self.data[k][ix * self.grain_size: (ix + 1) * self.grain_size]
                                                       for ix in grain_sample_ixs], 0)) for k in self.keys)

        else:
            batch_start_ix = self.i * self.batch_size
            batch_end_ix = (self.i + 1) * self.batch_size
            batch = tuple(torch.Tensor(self.data[k][batch_start_ix: batch_end_ix]) for k in self.keys)

        self.i += 1
        return batch

    def __len__(self) -> int:
        return self.n_batches
