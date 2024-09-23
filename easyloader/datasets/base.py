from abc import ABC, abstractmethod
from torch.utils.data import Dataset


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
        :param sample_seed: Seed for random sampling.
        :param shuffle: Whether to shuffle the data.
        :param shuffle_seed: The seed to be used for shuffling.
        """
        self.sample_fraction = sample_fraction
        self.sample_seed = sample_seed
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.ids = None
        self.index = None

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, ix: int):
        pass
