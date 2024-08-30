import h5py
import random
import numpy as np
import torch

from pathlib import Path
from typing import Sequence, Union
from torch.utils.data import Dataset, DataLoader

from uloops.loaders.common import get_n_batches


def check_keys(data_keys, requested_keys, allow_missing_keys=False):
    present_keys = []
    missing_keys = []
    for key in requested_keys:
        if key in data_keys:
            present_keys.append(key)
        else:
            missing_keys.append(key)

    if missing_keys and not allow_missing_keys:
        missing_key_string = ', '.join(missing_keys)
        raise KeyError(f'The following keys are missing from the h5 file: {missing_key_string}. '
                       'If you don\'t care, set allow_missing_keys to True.')

    if not present_keys:
        raise KeyError('None of the provided keys are present in the H5 file. Need at least one.')

    return present_keys


class H5Dataset(Dataset):
    """
    EV = Embedding, Value

    """

    def __init__(self,
                 data_path: Union[str, Path],
                 keys: Sequence[str],
                 allow_missing_keys: bool = False,
                 sample_fraction: float = 1.0,
                 sample_seed: int = 100,
                 index_key: str = None,
                 shuffle: bool = False):

        data = h5py.File(data_path, "r")
        present_keys = check_keys(data.keys(), keys, allow_missing_keys)

        if index_key is not None and index_key not in data.keys():
            raise KeyError(f'If specified, index_key must be in the H5 dataset\'s keys. {index_key} is missing')

        data_length = len(data[present_keys[0]])

        self.keys = keys
        self.data = data

        if sample_fraction != 1:
            sample_size = int(sample_fraction * data_length)
            rng = random.Random(sample_seed)
            self.sample = sorted(rng.sample([*range(data_length)], sample_size))
        else:
            self.sample = [*range(data_length)]

        if shuffle:
            self.sample = random.sample(self.sample, len(self.sample))

        if index_key is not None:
            self.index = data[index_key][self.sample]
        else:
            self.index = self.sample

    def __len__(self) -> int:
        return len(self.sample)

    def __getitem__(self, index: int):

        values = []
        for key in self.keys:
            if key in self.data:
                values.append(self.data[key][self.sample[index]])
            else:
                values.append(np.array([], dtype=np.float32))

        return tuple(values)


class H5DataLoader(DataLoader):
    """
    Turn an H5 file into a Torch dataset.
    Using granular weak shuffling.
    https://towardsdatascience.com/reading-h5-files-faster-with-pytorch-datasets-3ff86938cc
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 keys: Sequence[str],
                 allow_missing_keys: bool = False,
                 batch_size: int = 1,
                 sample_fraction: float = 1.0,
                 sample_seed: int = None,
                 grain_size: int = 1,
                 index_key: str = None,
                 shuffle: bool = False):

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
        self.batch_size = batch_size
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
