from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

from easyloader.utils.random import get_random_state


class EasyDataLoader(DataLoader, ABC):
    """
    Interface class for EasyLoader dataloaders with common functionality for sampling and indexing.
    """

    def __init__(self,
                 batch_size: int = 1,
                 sample_fraction: float = None,
                 shuffle: bool = False,
                 sample_seed: int = None,
                 shuffle_seed: bool = None,):
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

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
