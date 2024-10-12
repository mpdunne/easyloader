import numpy as np

from typing import Any, Sequence

from easyloader.loader.base import EasyDataLoader
from easyloader.dataset.array import ArrayDataset
from easyloader.utils.random import Seedable


class ArrayDataLoader(EasyDataLoader):
    """
    Turn a list of NumPy arrays into a PyTorch Data Loader.

    """

    dataset: ArrayDataset

    def __init__(self,
                 arrays: Sequence[np.ndarray],
                 ids: Sequence[Any] = None,
                 batch_size: int = 1,
                 sample_fraction: float = None,
                 shuffle: bool = False,
                 sample_seed: Seedable = None,
                 shuffle_seed: Seedable = None):
        """

        :param arrays: A list of arrays to use for the data loader
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

        self.dataset = ArrayDataset(arrays, ids=ids, sample_fraction=sample_fraction,
                                    sample_seed=sample_seed, shuffle_seed=shuffle_seed)
