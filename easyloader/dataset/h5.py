import h5py
import random
import numpy as np

from pathlib import Path
from typing import Sequence, Union

from easyloader.dataset.base import EasyDataset
from easyloader.common.h5 import check_keys


class H5Dataset(EasyDataset):
    """
    Turn a H5 file into a PyTorch Data Set.

    """

    def __init__(self,
                 data_path: Union[str, Path],
                 keys: Sequence[str],
                 id_key: str = None,
                 sample_fraction: float = 1.0,
                 sample_seed: int = None,
                 shuffle: bool = False,
                 shuffle_seed: int = None):

        """
        Constructor for the H5Dataset class.

        :param data_path: The path to the H5 file that you want to load.
        :param keys: The keys that you want to grab.
        :param sample_fraction: Fraction of the dataset to sample.
        :param sample_seed: Seed for random sampling.
        :param shuffle: Whether to shuffle the data.
        :param shuffle_seed: The seed to be used for shuffling.
        """

        # Initialize the parent class
        super().__init__(sample_fraction=sample_fraction,
                         sample_seed=sample_seed)

        self.data = H5Data(data_path, keys, id_key=id_key, sample_fraction=sample_fraction,
                           sample_seed=sample_seed, shuffle_seed=shuffle_seed)

        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):

        values = []
        for key in self.data.keys:
            values.append(self.data[key][index])

        return tuple(values)
