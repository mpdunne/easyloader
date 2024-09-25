from abc import abstractmethod
from typing import Iterable

from easyloader.utils.random import get_random_state, Seedable


class EasyData:
    """
    Data class for DF data.
    """

    def __init__(self,
                 sample_fraction: float = None,
                 sample_seed: Seedable = None,
                 shuffle_seed: Seedable = None):
        """
        Constructor for the Data Interface.

        :param sample_fraction: Fraction of the dataset to sample.
        :param sample_seed: Seed for random sampling.
        :param shuffle_seed: The seed to be used for shuffling.
        """

        self.sample_random_state = get_random_state(sample_seed)
        self.shuffle_random_state = get_random_state(shuffle_seed)

    @abstractmethod
    def shuffle(self):
        """
        Shuffle the underlying data

        :return: None.
        """
        pass

    def ids(self) -> Iterable:
        """
        The IDs.

        :return: The IDs
        """
        pass

    @abstractmethod
    def index(self):
        pass

    @abstractmethod
    def __len__(self):
        pass
