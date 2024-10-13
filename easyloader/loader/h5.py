from pathlib import Path
from typing import Hashable, Sequence, Union

from easyloader.loader.base import EasyDataLoader
from easyloader.dataset.h5 import H5Dataset
from easyloader.utils.random import Seedable


class H5DataLoader(EasyDataLoader):
    """
    Turn an H5 file into a Torch dataset.
    Using granular weak shuffling.
    https://towardsdatascience.com/reading-h5-files-faster-with-pytorch-datasets-3ff86938cc
    """

    dataset: H5Dataset

    def __init__(self,
                 data_path: Union[str, Path],
                 keys: Sequence[str],
                 ids: Union[str, Sequence[Hashable]] = None,
                 batch_size: int = 1,
                 grain_size: int = 1,
                 sample_fraction: float = None,
                 shuffle: bool = False,
                 sample_seed: Seedable = None,
                 shuffle_seed: Seedable = None):
        """
        Constructor for the H5DtaLoader class

        :param data_path: The path to the H5 file that you want to use.
        :param keys: The keys that you want to grab.
        :param ids: Either an ID key or a list of IDs to use for the data points.
        :param batch_size: The batch size.
        :param grain_size: The grain size.
        :param sample_fraction: Fraction of the dataset to sample.
        :param shuffle: Whether to shuffle each time iter is called.
        :param sample_seed: Seed for random sampling.
        :param shuffle_seed: The seed to be used for shuffling.
        """

        # Initialize the parent class
        super().__init__(batch_size=batch_size,
                         sample_fraction=sample_fraction,
                         sample_seed=sample_seed,
                         shuffle=shuffle,
                         shuffle_seed=shuffle_seed)

        self.dataset = H5Dataset(data_path, keys=keys, ids=ids, grain_size=grain_size, shuffle_seed=shuffle_seed,
                                 sample_fraction=sample_fraction, sample_seed=sample_seed)
