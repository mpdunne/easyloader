from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Hashable, Sequence, Iterable, Union

from easyloader.utils.random import get_random_state, Seedable


class EasyDataset(Dataset, ABC):
    """
    Interface class for EasyLoader datasets with common functionality for sampling and indexing.
    """

    _ids: Sequence[Hashable]
    _index: Sequence[int]

    def __init__(self, sample_fraction: float = 1.0,
                 sample_seed: Seedable = None,
                 shuffle_seed: Seedable = None):
        """
        Constructor for the EasyDataset class Interface.

        :param sample_fraction: Fraction of the dataset to sample.
        :param sample_seed: Seed for random sampling.
        :param shuffle_seed: Seed for shuffling.
        """

        super().__init__()

        self.sample_random_state = get_random_state(sample_seed)
        self.shuffle_random_state = get_random_state(shuffle_seed)

        self._ids = None
        self._index = None

    @abstractmethod
    def __getitem__(self, ix: Union[int, slice]):
        """
        Get items, either by a single index or by a slice.

        :return: A subset of items.
        """
        pass

    def __len__(self) -> int:
        """
        The length of the sampled data set.

        :return: The length of the sampled data set.
        """
        return len(self.index)

    @property
    def ids(self) -> Iterable:
        """
        The IDs, according to the id_column attribute.

        :return: The IDs
        """
        return [self._ids[i] for i in self.index]

    @property
    def index(self):
        """
        The index of the underlying data, relative to the original.

        :return: The length
        """
        return self._index

    @abstractmethod
    def shuffle(self):
        """
        Shuffle the underlying data.

        """
        pass
