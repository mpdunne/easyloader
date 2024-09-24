from abc import ABC, abstractmethod
from torch.utils.data import Dataset

from easyloader.utils.random import get_random_state


class EasyDataset(Dataset, ABC):
    """
    Interface class for EasyLoader datasets with common functionality for sampling and indexing.
    """

    def __init__(self, sample_fraction: float = 1.0,
                 sample_seed: int = None):
        """
        Constructor for the EasyDataset class Interface.

        :param sample_fraction: Fraction of the dataset to sample.
        :param sample_seed: Seed for random sampling.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, ix: int):
        pass
