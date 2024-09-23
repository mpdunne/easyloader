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
