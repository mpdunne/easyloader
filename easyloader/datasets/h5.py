import h5py
import random
import numpy as np

from pathlib import Path
from typing import Sequence, Union

from uloops.loaders.common import get_n_batches

from easyloader.datasets.base import EasyDataset
from easyloader.common.h5 import check_keys


class H5Dataset(EasyDataset):
    """
    EV = Embedding, Value

    """

    def __init__(self,
                 data_path: Union[str, Path],
                 keys: Sequence[str],
                 allow_missing_keys: bool = False,
                 sample_fraction: float = 1.0,
                 sample_seed: int = None,
                 shuffle: bool = False,
                 shuffle_seed: int = None):

        """

        :param data_path:
        :param keys:
        :param allow_missing_keys:
        :param df: The DF to use for the data set.
        :param sample_fraction: Fraction of the dataset to sample.
        :param sample_seed: Seed for random sampling.
        :param shuffle: Whether to shuffle the data.
        :param shuffle_seed: The seed to be used for shuffling.
        """


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
