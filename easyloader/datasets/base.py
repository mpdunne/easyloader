from abc import ABC, abstractmethod
from torch.utils.data import Dataset

from easyloader.utils.random import get_random_state


class EasyDataset(Dataset, ABC):
    """
    Interface class for EasyLoader datasets with common functionality for sampling and indexing.
    """

    def __init__(self, sample_fraction: float = 1.0,
                 sample_seed: int = None,
                 shuffle: bool = False,
                 shuffle_seed: bool = None):
        """
        Constructor for the EasyDataset class Interface.

        :param sample_fraction: Fraction of the dataset to sample.
        :param shuffle: Whether to shuffle the data.
        :param sample_seed: Seed for random sampling.
        :param shuffle_seed: The seed to be used for shuffling.
        """
        self.sample_fraction = sample_fraction
        self.shuffle = shuffle

        self.sample_random_state = get_random_state(sample_seed)
        self.shuffle_random_state = get_random_state(shuffle_seed)

        self.ids = None
        self.index = None

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, ix: int):
        pass
