import torch

from pathlib import Path
from typing import Sequence, Union

from easyloader.loader.base import EasyDataLoader
from easyloader.data.h5 import H5Data
from easyloader.utils.batch import get_n_batches


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
                 sample_seed: int = None,
                 shuffle: bool = False,
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

        self.data = H5Data(data_path, keys, id_key=id_key, grain_size=grain_size,
                           sample_fraction=sample_fraction, sample_seed=sample_seed, shuffle_seed=shuffle_seed)

    def __iter__(self):
        if self.shuffle:
            self.data.shuffle()

        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self):
            raise StopIteration

        batch_start_ix = self.i * self.batch_size
        batch_end_ix = (self.i + 1) * self.batch_size
        batch = tuple(torch.Tensor(self.data[k][batch_start_ix: batch_end_ix]) for k in self.data.keys)

        self.i += 1
        return batch

    def __len__(self) -> int:
        return get_n_batches(len(self.data), self.batch_size)

