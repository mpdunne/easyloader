import torch
import numpy as np

from abc import ABC
from torch.utils.data import DataLoader

from easyloader.dataset.base import EasyDataset
from easyloader.utils.random import Seedable
from easyloader.utils.batch import get_n_batches


class EasyDataLoader(DataLoader, ABC):
    """
    Interface class for EasyLoader dataloaders with common functionality for sampling and indexing.
    """

    # This must be set by the child class.
    dataset: EasyDataset

    def __init__(self,
                 batch_size: int = 1,
                 sample_fraction: float = None,
                 shuffle: bool = False,
                 sample_seed: Seedable = None,
                 shuffle_seed: Seedable = None,):
        """
        Constructor for the EasyDataLoader class Interface.

        :param batch_size: The batch size.
        :param sample_fraction: Fraction of the dataset to sample.
        :param sample_seed: Seed for random sampling.
        :param shuffle_seed: The seed to be used for shuffling.
        :param shuffle: Whether to shuffle the data.
        """

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_fraction = sample_fraction
        self.sample_seed = sample_seed
        self.shuffle_seed = shuffle_seed

    @property
    def index(self):
        """
        The numeric indices of the underlying data, relative to the inputted one.

        :return: The indices.
        """
        return self.dataset.index

    @property
    def ids(self):
        """
        The IDs, according to the id_key attribute.

        :return: The IDs
        """
        return self.dataset.ids

    def __iter__(self):
        """
        Shuffle the underlying data set.

        """
        if self.shuffle:
            self.dataset.shuffle()

        self.i = 0
        return self

    def __len__(self) -> int:
        """
        The length of the loader, i.e. the number of batches.

        :return: The total number of batches.
        """
        return get_n_batches(len(self.dataset), self.batch_size)

    def __next__(self):
        """
        Get the next batch.

        :return: The next batch.
        """
        if self.i >= len(self):
            raise StopIteration

        batch_start_ix = self.i * self.batch_size
        batch_end_ix = (self.i + 1) * self.batch_size

        result = self.dataset[batch_start_ix: batch_end_ix]
        if isinstance(result, np.ndarray):
            batch = torch.Tensor(result)
        else:
            batch = tuple(torch.Tensor(arr) for arr in result)

        self.i += 1
        return batch
