import numpy as np
import torch

from pathlib import Path
from typing import Sequence, Union

from easyloader.loader.base import EasyDataLoader
from easyloader.dataset.h5 import H5Dataset
from easyloader.utils.grains import grab_slices_from_grains


class H5DataLoader(EasyDataLoader):
    """
    Turn an H5 file into a Torch dataset.
    Using granular weak shuffling.
    https://towardsdatascience.com/reading-h5-files-faster-with-pytorch-datasets-3ff86938cc
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 keys: Sequence[str],
                 id_key: str = None,
                 batch_size: int = 1,
                 grain_size: int = 1,
                 sample_fraction: float = None,
                 shuffle: bool = False,
                 sample_seed: int = None,
                 shuffle_seed: bool = None):
        """
        Constructor for the H5DtaLoader class

        :param data_path: The path to the H5 file that you want to use.
        :param keys: The keys that you want to grab.
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

        self.dataset = H5Dataset(data_path, keys=keys, id_key=id_key, grain_size=grain_size, shuffle_seed=shuffle_seed,
                                 sample_fraction=sample_fraction, sample_seed=sample_seed)

    def __next__(self):
        """
        Get the next batch.

        :return: The next batch.
        """
        if self.i >= len(self):
            raise StopIteration

        values = []

        batch_start_ix = self.i * self.batch_size
        batch_end_ix = (self.i + 1) * self.batch_size

        ix_slices = grab_slices_from_grains(self.dataset.grain_index, self.dataset.grain_size, batch_start_ix, batch_end_ix)
        for key in self.dataset.keys:
            values.append(torch.Tensor(np.concatenate([self.dataset.h5[key][ix_slice] for ix_slice in ix_slices])))

        self.i += 1
        return tuple(values)
